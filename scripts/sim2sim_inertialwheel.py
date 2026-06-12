"""
sim2sim_inertialwheel.py — 将 Isaac Lab 训练的策略部署到 MuJoCo 仿真

用法:
    # 用最新一次训练的模型
    python scripts/sim2sim_inertialwheel.py

    # 指定特定训练记录
    python scripts/sim2sim_inertialwheel.py --run 2026-06-05_11-52-51

物理对齐说明:
    MuJoCo 的坐标/惯量/阻尼已经与 Isaac Lab 对齐。
    如果 wheel 力矩方向反了，设置 --action_sign -1。
"""

import argparse
import os
import time

import numpy as np
import torch

# MuJoCo
import mujoco
import mujoco.viewer

# =============================================================
# 解析参数
# =============================================================
parser = argparse.ArgumentParser(description="Sim2Sim: deploy Isaac Lab policy to MuJoCo")
parser.add_argument("--run", type=str, default=None,
                    help="训练记录文件夹名 (如 2026-06-05_11-52-51)，默认自动选最新的")
parser.add_argument("--action_sign", type=int, default=-1, choices=[1, -1],
                    help="力矩符号修正 (Default -1: MuJoCo wheel axis 方向与 USD 相反)")
parser.add_argument("--action_scale", type=float, default=50.0,
                    help="策略输出 [-1,1] → 实际力矩的缩放系数 (训练时 scale=50)")
parser.add_argument("--xml", type=str,
                    default="./Mujoco/inertial_wheel/scene.xml",
                    help="MuJoCo 场景 XML")
parser.add_argument("--max_steps", type=int, default=4000,
                    help="最大仿真步数 (0=无限)")
parser.add_argument("--no_render", action="store_true",
                    help="不打开渲染窗口，仅跑仿真")
args = parser.parse_args()

# =============================================================
# 自动找最新的策略
# =============================================================
log_root = "logs/rsl_rl/inertialwheel"
if args.run is None:
    runs = sorted(os.listdir(log_root))
    if not runs:
        raise FileNotFoundError(f"没有找到任何训练记录在 {log_root}")
    args.run = runs[-1]

policy_path = os.path.join(log_root, args.run, "exported", "policy.pt")
if not os.path.exists(policy_path):
    # 尝试找 onnx
    policy_path = os.path.join(log_root, args.run, "exported", "policy.onnx")

print(f"[Info] 加载策略: {policy_path}")
print(f"[Info] 力矩符号: {'正' if args.action_sign == 1 else '反'}")

# =============================================================
# 加载策略
# =============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
if policy_path.endswith(".onnx"):
    import onnxruntime as ort
    session = ort.InferenceSession(policy_path)
    def policy(obs_np):
        inputs = {session.get_inputs()[0].name: obs_np.astype(np.float32)}
        return session.run(None, inputs)[0]
    policy_device = "cpu"
else:
    policy_jit = torch.jit.load(policy_path, map_location=device)
    policy_jit.eval()
    def policy(obs_np):
        obs_t = torch.from_numpy(obs_np).float().to(device)
        with torch.inference_mode():
            a = policy_jit(obs_t)
        return a.cpu().numpy()
    policy_device = device

# =============================================================
# MuJoCo 环境
# =============================================================
XML_PATH = os.path.abspath(args.xml)
model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

# 确认关节和 actuator 索引
body_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "body_joint")
wheel_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "wheel_joint")
wheel_actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "wheel_joint")
print(f"[Info] body_joint_id={body_joint_id}, wheel_joint_id={wheel_joint_id}, wheel_act_id={wheel_actuator_id}")
assert wheel_actuator_id >= 0, "找不到 wheel_joint actuator"

# 关节状态索引
body_qpos_adr = model.jnt_qposadr[body_joint_id]
body_dof_adr = model.jnt_dofadr[body_joint_id]
wheel_qpos_adr = model.jnt_qposadr[wheel_joint_id]
wheel_dof_adr = model.jnt_dofadr[wheel_joint_id]

print(f"[Info] body_qpos_adr={body_qpos_adr}, body_dof_adr={body_dof_adr}")
print(f"[Info] wheel_qpos_adr={wheel_qpos_adr}, wheel_dof_adr={wheel_dof_adr}")

# =============================================================
# 观测函数: 匹配 Isaac Lab 的训练观测
#
# Isaac Lab 观测 (5维):
#   [sin(body_pos), cos(body_pos), wheel_pos, body_vel, wheel_vel]
# =============================================================
def get_obs():
    body_q = data.qpos[body_qpos_adr]       # body_joint 角度
    body_v = data.qvel[body_dof_adr]         # body_joint 速度
    wheel_q = data.qpos[wheel_qpos_adr]      # wheel_joint 角度
    wheel_v = data.qvel[wheel_dof_adr]       # wheel_joint 速度

    obs = np.array([
        np.sin(body_q),
        np.cos(body_q),
        wheel_q,
        body_v,
        wheel_v,
    ], dtype=np.float32)
    return obs

def reset_to_pose(body_angle=0.0, body_vel=0.0, wheel_angle=0.0, wheel_vel=0.0):
    """复位到指定位置 (摆杆下垂)"""
    data.qpos[body_qpos_adr] = body_angle
    data.qpos[wheel_qpos_adr] = wheel_angle
    data.qvel[body_dof_adr] = body_vel
    data.qvel[wheel_dof_adr] = wheel_vel
    mujoco.mj_forward(model, data)

# =============================================================
# 主循环
# =============================================================
def run_simulation(render=True, max_steps=4000):
    # 初始复位: 摆杆下垂
    np.random.seed(0)
    init_body = np.random.uniform(-0.3, 0.3)
    init_wheel = np.random.uniform(-0.3, 0.3)
    init_body_v = np.random.uniform(-0.3, 0.3)
    init_wheel_v = np.random.uniform(-0.3, 0.3)
    reset_to_pose(init_body, init_body_v, init_wheel, init_wheel_v)

    step = 0
    total_reward = 0.0

    if render:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                # --- 策略推理 ---
                obs = get_obs()
                action = policy(obs[np.newaxis])  # [1, 5] -> [1, 1]
                torque = float(action[0, 0]) * args.action_scale * args.action_sign

                # --- 施加动作 ---
                data.ctrl[wheel_actuator_id] = torque

                # --- MuJoCo 步进 ---
                mujoco.mj_step(model, data)

                # --- 打印状态 ---
                if step % 50 == 0:
                    body_q = data.qpos[body_qpos_adr]
                    wheel_q = data.qpos[wheel_qpos_adr]
                    body_v = data.qvel[body_dof_adr]
                    wheel_v = data.qvel[wheel_dof_adr]
                    up_reward = (np.cos(body_q - np.pi) + 1) / 2
                    print(f"step={step:5d}  body_q={body_q:+.3f}  wheel_q={wheel_q:+.3f}  "
                          f"body_v={body_v:+.3f}  wheel_v={wheel_v:+.3f}  "
                          f"torque={torque:+.2f}  upright={up_reward:.3f}")

                # --- 同步 viewer ---
                viewer.sync()

                step += 1
                if 0 < max_steps <= step:
                    print(f"\n[Info] 达到最大步数 {max_steps}")
                    break
    else:
        # 无渲染模式（快速跑）
        while step < max_steps:
            obs = get_obs()
            action = policy(obs[np.newaxis])
            torque = float(action[0, 0]) * args.action_scale * args.action_sign
            data.ctrl[wheel_actuator_id] = torque
            mujoco.mj_step(model, data)

            if step % 100 == 0:
                body_q = data.qpos[body_qpos_adr]
                up_reward = (np.cos(body_q - np.pi) + 1) / 2
                print(f"step={step:5d}  body_q={body_q:+.3f}  upright={up_reward:.3f}  torque={torque:+.2f}")

            step += 1

    return step

if __name__ == "__main__":
    print("=" * 60)
    print(f"实验: {args.run}")
    print(f"策略: {policy_path}")
    print(f"XML:  {args.xml}")
    print(f"渲染: {'开' if not args.no_render else '关'}")
    print("=" * 60)
    print()

    total_steps = run_simulation(render=not args.no_render, max_steps=args.max_steps)
    print(f"\n[完成] 共仿真 {total_steps} 步")
