"""
IW_test_lqr.py — LQR + 共振Swing-up (v4 — 稳定版)
========================================

为什么 PID 难控制?
  惯性轮摆是欠驱动系统: 1电机 → 2自由度 (摆杆 + 轮子)
  PID 的 θ_err→τ 映射经过积分器级联 + RHP零点, 相位滞后太大.

控制策略:
  STAGE 1 — 共振泵能 (|θ_err| > 0.4):
     以摆杆自然频率 ωₙ ≈ 4.45 rad/s 做正弦力矩驱动
     τ = A·sin(ωₙ·t) - b·ω_w
     靠共振逐渐把能量泵入摆杆

  STAGE 2 — LQR 平衡 (|θ_err| ≤ 0.4):
     τ = -K @ (x - x_target)

状态: x = [θ, φ_w, ω_b, ω_w]^T

参考文献: Spong, M. "Swing up control of the inertia wheel pendulum"
"""

import mujoco
import mujoco.viewer
import numpy as np
from scipy.linalg import solve_continuous_are
import time
# =====================================================================
model = mujoco.MjModel.from_xml_path("./Mujoco/inertial_wheel/scene.xml")
data = mujoco.MjData(model)

TARGET = np.pi
SWITCH_RAD = 0.4                  # 共振 → LQR 切换阈值 (rad)
TORQUE_MAX = 50.0                 # 力矩限幅 (N·m)
SWING_AMP = 30.0                  # 共振驱动振幅 (N·m)
BRAKE_GAIN = 0.01                 # 轮子刹车 (防止无限加速)

# =====================================================================
# 1. 线性化 (θ=π 竖直向上)
# =====================================================================
nv = model.nv
gear = model.actuator("wheel_joint").gear[0]

data.jnt("body_joint").qpos[0] = TARGET
data.jnt("wheel_joint").qpos[0] = 0.0
mujoco.mj_forward(model, data)

M_mat = np.zeros((nv, nv))
mujoco.mj_fullM(model, M_mat, data.qM)
M_inv = np.linalg.inv(M_mat)

bias_0 = data.qfrc_bias[:nv].copy()
eps = 1e-6
dbias_dq = np.zeros((nv, nv))
for i in range(nv):
    data.qpos[i] += eps
    mujoco.mj_forward(model, data)
    dbias_dq[:, i] = (data.qfrc_bias[:nv] - bias_0) / eps
    data.qpos[i] -= eps

data.qpos[0] = TARGET
mujoco.mj_forward(model, data)
D_lin = np.diag([model.dof_damping[0], model.dof_damping[1]])

A = np.zeros((4, 4))
A[0, 2] = 1.0
A[1, 3] = 1.0
A[2:4, 0:2] = -M_inv @ dbias_dq
A[2:4, 2:4] = -M_inv @ D_lin
B = np.zeros((4, 1))
B[2:4, 0] = M_inv @ [0, gear]

poles = np.linalg.eigvals(A)
print("开环极点 (θ=π):", poles.round(4))
print("不稳定极点数:", sum(1 for p in poles if p.real > 1e-6))

# =====================================================================
# 2. LQR 设计
# =====================================================================
Q = np.diag([500.0, 2.0, 20.0, 0.05])
R = np.array([[100.0]])  # gear=1.0 → R 放大 20x
P = solve_continuous_are(A, B, Q, R)
K = np.linalg.inv(R) @ B.T @ P

print("LQR K =", K.flatten().round(2))
print("闭环极点:", np.linalg.eigvals(A - B @ K).round(4))
print("=" * 60)

# =====================================================================
# 3. 共振频率计算
# =====================================================================
# 在 θ=0 (下垂) 处线性化, 找摆杆自然频率
data.qpos[0] = 0.0
mujoco.mj_forward(model, data)
M_h = np.zeros((nv, nv))
mujoco.mj_fullM(model, M_h, data.qM)
M_h_inv = np.linalg.inv(M_h)

bias_h = data.qfrc_bias[:nv].copy()
dbias_h = np.zeros((nv, nv))
for i in range(nv):
    data.qpos[i] += eps
    mujoco.mj_forward(model, data)
    dbias_h[:, i] = (data.qfrc_bias[:nv] - bias_h) / eps
    data.qpos[i] -= eps

A_h = np.zeros((4, 4))
A_h[0, 2] = 1.0
A_h[1, 3] = 1.0
A_h[2:4, 0:2] = -M_h_inv @ dbias_h
A_h[2:4, 2:4] = -M_h_inv @ D_lin
eigs_h = np.linalg.eigvals(A_h)
omega_n = max(abs(e.imag) for e in eigs_h)
period = 2 * np.pi / omega_n
print(f"摆杆自然频率: ωₙ = {omega_n:.3f} rad/s")
print(f"周期: T = {period:.3f} s ({period/model.opt.timestep:.0f} 步)")
print("=" * 60)

data.qpos[0] = TARGET
mujoco.mj_forward(model, data)

# =====================================================================
# 4. 主仿真
# =====================================================================
data.qpos[0] = 0.0
data.qpos[1] = 0.0
data.qvel[:] = 0.0
mujoco.mj_forward(model, data)

step = 0

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        θ = data.jnt("body_joint").qpos[0]
        ω_b = data.jnt("body_joint").qvel[0]
        φ_w = data.jnt("wheel_joint").qpos[0]
        ω_w = data.jnt("wheel_joint").qvel[0]
        θ_err = TARGET - θ
        t = step * model.opt.timestep

        if abs(θ_err) > SWITCH_RAD:
            # Stage 1: 共振泵能
            # τ = A·sin(ωₙ·t) - b·ω_w
            # 靠共振把能量泵入摆杆
            torque = SWING_AMP * np.sin(omega_n * t) - BRAKE_GAIN * ω_w
            torque = np.clip(torque, -TORQUE_MAX, TORQUE_MAX)
            label = "SWING"

        else:
            # Stage 2: LQR 精确平衡
            torque = -(K[0, 0] * (θ - TARGET) +
                       K[0, 1] * φ_w +
                       K[0, 2] * ω_b +
                       K[0, 3] * ω_w)
            torque = np.clip(torque, -TORQUE_MAX, TORQUE_MAX)
            label = "LQR "

        data.ctrl[0] = torque
        mujoco.mj_step(model, data)

        if step % 20 == 0:
            print(f"[{label}][{step:5d}] θ:{θ:+.3f}(err:{θ_err:+.3f})  "
                  f"ω_b:{ω_b:+.2f}  ω_w:{ω_w:+.0f}  τ:{torque:+6.1f}")
        time.sleep(0.01)  # 降低输出频率, 视觉上更连贯
        viewer.sync()
        step += 1
