"""
IW_train_ppo.py — PPO + LQR 混合控制 惯性轮摆
==============================================

PPO 负责远距离推动 (|θ_err| > 0.4):
  学到如何从下垂推向 π
LQR 负责近距离抓取 (|θ_err| ≤ 0.4):
  精确平衡, 不飞过

观测: [sin(θ), cos(θ), φ_w, ω_b, ω_w]
动作: 电机力矩 [-50, 50] N·m
目标: θ = π
"""

import mujoco
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from collections import deque
import os
import time

# =====================================================================
# 配置
# =====================================================================
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_SCRIPT_DIR, "scene.xml")
TARGET = np.pi
TORQUE_MAX = 50.0

# PPO 超参数
LR = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
ENT_COEF = 0.01
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
UPDATE_EPOCHS = 10
BATCH_SIZE = 64
N_STEPS = 2048
N_EPISODES = 3000
MAX_EP_STEPS = 2000
SAVE_INTERVAL = 100

# 混合切换阈值
SWITCH_RAD = 1.2       # PPO → LQR 切换
LQR_GAIN_SCALE = 0.8   # LQR 增益缩放

# LQR 增益 (Kθ, Kφ, Kωb, Kωw)
LQR_K = np.array([60.5377, 0.0, 12.5443, -0.6899], dtype=np.float32)

# =====================================================================
# 神经网络
# =====================================================================
OBS_DIM = 5


class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(OBS_DIM, 256), nn.Tanh(),
            nn.Linear(256, 128), nn.Tanh(),
        )
        self.actor_mean = nn.Linear(128, 1)
        self.actor_logstd = nn.Parameter(torch.zeros(1))
        self.critic = nn.Linear(128, 1)

    def forward(self, obs):
        x = self.fc(obs)
        mean = self.actor_mean(x)
        logstd = self.actor_logstd.expand_as(mean)
        value = self.critic(x)
        return mean, logstd, value

    def get_action(self, obs, deterministic=False):
        mean, logstd, value = self.forward(obs)
        if deterministic:
            return torch.tanh(mean) * TORQUE_MAX, None, value
        std = torch.exp(logstd)
        dist = Normal(mean, std)
        raw_action = dist.rsample()
        action = torch.tanh(raw_action) * TORQUE_MAX
        log_prob = dist.log_prob(raw_action)
        log_prob = log_prob - torch.log(1 - torch.tanh(raw_action).pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob, value


def lqr_torque(θ, φ_w, ω_b, ω_w):
    """LQR 控制律"""
    θ_err = TARGET - θ
    return -(LQR_K[0] * (θ - TARGET) + LQR_K[1] * φ_w +
             LQR_K[2] * ω_b + LQR_K[3] * ω_w)


# =====================================================================
# MuJoCo 环境 (混合控制)
# =====================================================================
class InertialWheelEnv:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(MODEL_PATH)
        self.data = mujoco.MjData(self.model)

    def reset(self):
        seed = np.random.randint(0, 2**31)
        rng = np.random.RandomState(seed)
        θ0 = rng.uniform(-0.5, 0.5)
        self.data.jnt("body_joint").qpos[0] = θ0
        self.data.jnt("wheel_joint").qpos[0] = 0.0
        self.data.jnt("body_joint").qvel[0] = rng.uniform(-1.0, 1.0)
        self.data.jnt("wheel_joint").qvel[0] = rng.uniform(-1.0, 1.0)
        mujoco.mj_forward(self.model, self.data)
        return self._obs()

    def _obs(self):
        θ = self.data.jnt("body_joint").qpos[0]
        ω_b = self.data.jnt("body_joint").qvel[0]
        φ_w = self.data.jnt("wheel_joint").qpos[0]
        ω_w = self.data.jnt("wheel_joint").qvel[0]
        return np.array([np.sin(θ), np.cos(θ), φ_w, ω_b, ω_w], dtype=np.float32)

    def step(self, ppo_torque):
        """执行动作: PPO 输出 + LQR 介入"""
        θ = self.data.jnt("body_joint").qpos[0]
        ω_b = self.data.jnt("body_joint").qvel[0]
        φ_w = self.data.jnt("wheel_joint").qpos[0]
        ω_w = self.data.jnt("wheel_joint").qvel[0]
        θ_err = TARGET - θ

        # 混合控制: 近处 LQR 完全接管
        if abs(θ_err) < SWITCH_RAD:
            torque = lqr_torque(θ, φ_w, ω_b, ω_w)
        else:
            torque = float(ppo_torque)

        torque = np.clip(torque, -TORQUE_MAX, TORQUE_MAX)
        self.data.ctrl[0] = torque
        mujoco.mj_step(self.model, self.data)

        obs = self._obs()
        θ = self.data.jnt("body_joint").qpos[0]
        ω_b = self.data.jnt("body_joint").qvel[0]
        ω_w = self.data.jnt("wheel_joint").qvel[0]
        θ_err = TARGET - θ

        # 奖励: 只奖励 PPO 把摆杆推到 π 附近的行为
        if abs(θ_err) < SWITCH_RAD:
            # LQR 接管: 给固定奖励 (不罚刹车动作)
            reward = 10.0 + max(0, 1.0 - abs(θ_err) / SWITCH_RAD) * 20.0
        else:
            # PPO 做 swing-up: 越靠近 π 奖励越高
            reward = 5.0 * (1.0 - min(1.0, abs(θ_err) / np.pi))
            reward -= 0.02 * abs(ω_b) + 0.005 * abs(ω_w)

        done = False
        return obs, reward, done


# =====================================================================
# 训练
# =====================================================================
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}", flush=True)

    policy = ActorCritic().to(device)
    optimizer = optim.Adam(policy.parameters(), lr=LR)
    env = InertialWheelEnv()

    episode_rewards = deque(maxlen=50)
    best_mean_reward = -np.inf
    global_step = 0
    episode = 0

    obs_buf, act_buf, logp_buf, val_buf, rew_buf, done_buf = (
        [], [], [], [], [], []
    )

    obs = env.reset()
    episode_reward = 0
    ep_steps = 0
    total_start = time.time()

    while episode < N_EPISODES:
        steps_this_rollout = 0
        while steps_this_rollout < N_STEPS and episode < N_EPISODES:
            obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(device)
            with torch.no_grad():
                action, log_prob, value = policy.get_action(obs_t)
            torque = action.cpu().numpy()[0, 0]

            next_obs, reward, done = env.step(torque)
            ep_steps += 1
            global_step += 1
            episode_reward += reward
            steps_this_rollout += 1

            obs_buf.append(obs)
            act_buf.append(action.cpu().numpy()[0])
            logp_buf.append(log_prob.cpu().numpy()[0, 0])
            val_buf.append(value.cpu().numpy()[0, 0])
            rew_buf.append(reward)
            done_buf.append(done or ep_steps >= MAX_EP_STEPS)

            obs = next_obs

            if done or ep_steps >= MAX_EP_STEPS:
                episode += 1
                episode_rewards.append(episode_reward)
                obs = env.reset()
                episode_reward = 0
                ep_steps = 0

        # ---- PPO 更新 (GAE) ----
        if len(obs_buf) < 128:
            continue

        with torch.no_grad():
            obs_last = torch.from_numpy(obs).float().unsqueeze(0).to(device)
            _, _, last_val = policy(obs_last)
            last_val = last_val.cpu().numpy()[0, 0]

        advantages = np.zeros(len(rew_buf), dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(len(rew_buf))):
            if t == len(rew_buf) - 1:
                next_val = last_val if not done_buf[t] else 0.0
            else:
                next_val = val_buf[t + 1]
            delta = rew_buf[t] + GAMMA * next_val - val_buf[t]
            last_gae = delta + GAMMA * GAE_LAMBDA * (1.0 - done_buf[t]) * last_gae
            advantages[t] = last_gae
        returns = advantages + np.array(val_buf)

        adv_mean, adv_std = advantages.mean(), advantages.std() + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        obs_arr = np.array(obs_buf)
        act_arr = np.array(act_buf)
        logp_arr = np.array(logp_buf)
        adv_arr = advantages.copy()
        ret_arr = returns.copy()

        obs_buf.clear()
        act_buf.clear()
        logp_buf.clear()
        val_buf.clear()
        rew_buf.clear()
        done_buf.clear()

        idxs = np.arange(len(obs_arr))
        for _ in range(UPDATE_EPOCHS):
            np.random.shuffle(idxs)
            for start in range(0, len(idxs), BATCH_SIZE):
                batch = idxs[start:start + BATCH_SIZE]

                obs_b = torch.from_numpy(obs_arr[batch]).float().to(device)
                act_b = torch.from_numpy(act_arr[batch]).float().to(device)
                old_logp_b = torch.from_numpy(logp_arr[batch]).float().to(device)
                adv_b = torch.from_numpy(adv_arr[batch]).float().to(device)
                ret_b = torch.from_numpy(ret_arr[batch]).float().to(device)

                mean, logstd, values = policy(obs_b)
                std = torch.exp(logstd)
                dist = Normal(mean, std)
                raw_act = torch.atanh(
                    torch.clamp(act_b / TORQUE_MAX, -0.999, 0.999)
                )
                log_prob = dist.log_prob(raw_act)
                log_prob = log_prob - torch.log(
                    1 - torch.tanh(raw_act).pow(2) + 1e-6
                )
                log_prob = log_prob.sum(dim=-1, keepdim=True)

                ratio = torch.exp(log_prob - old_logp_b)
                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv_b
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = nn.MSELoss()(values, ret_b.unsqueeze(1))
                entropy = dist.entropy().mean()
                loss = policy_loss + VF_COEF * value_loss - ENT_COEF * entropy

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRAD_NORM)
                optimizer.step()

        # ---- 日志 ----
        if len(episode_rewards) > 0:
            mean_reward = np.mean(episode_rewards)
            if episode % 10 == 0:
                elapsed = time.time() - total_start
                print(
                    f"[{episode:4d}/{N_EPISODES}]  "
                    f"步数={global_step:6d}  "
                    f"平均奖励={mean_reward:+.2f}  "
                    f"最佳={best_mean_reward:+.2f}  "
                    f"时间={elapsed:.0f}s",
                    flush=True,
                )

            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                torch.save(
                    policy.state_dict(),
                    os.path.join(_SCRIPT_DIR, "ppo_best.pth"),
                )
                print(f"  → 新最佳模型保存! (奖励={mean_reward:+.2f})")

            if episode % SAVE_INTERVAL == 0 and episode > 0:
                torch.save(
                    policy.state_dict(),
                    os.path.join(_SCRIPT_DIR, f"ppo_ep{episode}.pth"),
                )

    torch.save(
        policy.state_dict(),
        os.path.join(_SCRIPT_DIR, "ppo_final.pth"),
    )
    total_time = time.time() - total_start
    print(
        f"\n训练完成! 用时={total_time:.0f}s, "
        f"最终模型已保存到 ppo_final.pth"
    )


if __name__ == "__main__":
    train()
