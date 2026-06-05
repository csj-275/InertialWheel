"""
IW_test_ppo.py — 测试 PPO + LQR 混合策略
=========================================

PPO 做 swing-up (远距离), LQR 做抓取平衡 (近距离)
训练好的 PPO 策略保存为 ppo_best.pth
"""

import mujoco
import mujoco.viewer
import numpy as np
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
try:
    from IW_train_ppo import ActorCritic, MODEL_PATH, SWITCH_RAD, TARGET, TORQUE_MAX, lqr_torque
except ImportError:
    import torch.nn as nn
    from torch.distributions import Normal
    OBS_DIM = 5; TARGET = np.pi; TORQUE_MAX = 50.0; SWITCH_RAD = 0.4
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "scene.xml")

    class ActorCritic(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Sequential(
                nn.Linear(OBS_DIM, 256), nn.Tanh(), nn.Linear(256, 128), nn.Tanh()
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

    def lqr_torque(θ, φ_w, ω_b, ω_w):
        return -(60.5377*(θ-TARGET) + 0*φ_w + 12.5443*ω_b + -0.6899*ω_w)


def test(model_path=None, deterministic=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    savedir = os.path.dirname(__file__) if "__file__" in dir() else "."
    if model_path is None:
        candidates = [
            os.path.join(savedir, "ppo_best.pth"),
            os.path.join(savedir, "ppo_final.pth"),
        ]
        model_path = next((p for p in candidates if os.path.exists(p)), None)
        if model_path is None:
            print("未找到训练好的模型, 使用纯 LQR")
            model_path = None
        else:
            print(f"加载模型: {model_path}")

    policy = ActorCritic().to(device)
    use_ppo = model_path is not None and os.path.exists(model_path)

    if use_ppo:
        try:
            state_dict = torch.load(model_path, map_location=device, weights_only=True)
            policy.load_state_dict(state_dict)
            policy.eval()
            print("PPO 模型加载成功, 使用 PPO+LQR 混合控制")
        except Exception as e:
            print(f"PPO 加载失败 ({e}), 使用纯 LQR")
            use_ppo = False
    else:
        print("使用纯 LQR 控制")

    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    # 从下垂开始
    data.jnt("body_joint").qpos[0] = 0.0
    data.jnt("wheel_joint").qpos[0] = 0.0
    data.jnt("body_joint").qvel[0] = 0.0
    data.jnt("wheel_joint").qvel[0] = 0.0
    mujoco.mj_forward(model, data)

    step = 0
    lqr_steps = 0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and step < 50000:
            θ = data.jnt("body_joint").qpos[0]
            ω_b = data.jnt("body_joint").qvel[0]
            φ_w = data.jnt("wheel_joint").qpos[0]
            ω_w = data.jnt("wheel_joint").qvel[0]
            θ_err = TARGET - θ

            # 混合控制
            if abs(θ_err) < SWITCH_RAD:
                torque = lqr_torque(θ, φ_w, ω_b, ω_w)
                label = "LQR"
                lqr_steps += 1
            elif use_ppo:
                obs = np.array([np.sin(θ), np.cos(θ), φ_w, ω_b, ω_w], dtype=np.float32)
                with torch.no_grad():
                    obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(device)
                    mean, _, _ = policy(obs_t)
                    if deterministic:
                        torque = torch.tanh(mean) * TORQUE_MAX
                    else:
                        _, logstd, _ = policy(obs_t)
                        dist = Normal(mean, torch.exp(logstd))
                        torque = torch.tanh(dist.sample()) * TORQUE_MAX
                    torque = torque.cpu().numpy()[0, 0]
                label = "PPO"
            else:
                # 纯 LQR: 虽然远距离效果不好但不会飞
                torque = lqr_torque(θ, φ_w, ω_b, ω_w)
                label = "LQR"

            torque = np.clip(torque, -TORQUE_MAX, TORQUE_MAX)
            data.ctrl[0] = torque
            mujoco.mj_step(model, data)

            if step % 50 == 0:
                print(
                    f"[{label}][{step:4d}] θ:{θ:+.3f}(err:{θ_err:+.3f})  "
                    f"ω_b:{ω_b:+.2f}  ω_w:{ω_w:+.0f}  τ:{torque:+6.1f}"
                )

            viewer.sync()
            step += 1

    print(f"\n测试完成, 共 {step} 步, LQR 介入 {lqr_steps} 步")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--stochastic", action="store_true")
    args = parser.parse_args()
    test(model_path=args.model, deterministic=not args.stochastic)
