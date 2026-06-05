"""
IW_test01.py — 极点配置控制器

状态: x = [body_pos_err, body_vel, wheel_pos, wheel_vel]
控制: tau = -(k1*x1 + k2*x2 + k3*x3 + k4*x4) = -K @ x
"""

import mujoco
import mujoco.viewer
import numpy as np
from scipy.signal import place_poles

model = mujoco.MjModel.from_xml_path("scene.xml")
data = mujoco.MjData(model)

target = np.pi  # body_joint 期望角度 (竖直向上)

# ========== 在竖直向上平衡点线性化 ==========
nv = model.nv  # 2自由度: body_joint, wheel_joint
gear = model.actuator("wheel_joint").gear[0]

# 设置到目标构型
data.jnt("body_joint").qpos[0] = target
data.jnt("wheel_joint").qpos[0] = 0.0
data.jnt("body_joint").qvel[0] = 0.0
data.jnt("wheel_joint").qvel[0] = 0.0
mujoco.mj_forward(model, data)

# 质量矩阵 M (2×2)
M = np.zeros((nv, nv))
mujoco.mj_fullM(model, M, data.qM)
M_inv = np.linalg.inv(M)

# 平衡点处重力矩: τ_gravity = -M @ qacc_smooth_forward (qvel=0 时)
qacc_0 = data.qacc_smooth_forward[:nv].copy()
tau_grav_0 = -M @ qacc_0

# 重力刚度矩阵 ∂τ_gravity/∂q (有限差分)
eps = 1e-6
dG_dq = np.zeros((nv, nv))
for i in range(nv):
    data.qpos[i] += eps
    mujoco.mj_forward(model, data)
    # 用当前构型的 M 重新计算重力矩
    M_pert = np.zeros((nv, nv))
    mujoco.mj_fullM(model, M_pert, data.qM)
    tau_pert = -M_pert @ data.qacc_smooth_forward[:nv]
    dG_dq[:, i] = (tau_pert - tau_grav_0) / eps
    data.qpos[i] -= eps

# 恢复构型
data.jnt("body_joint").qpos[0] = target
mujoco.mj_forward(model, data)

# 阻尼矩阵
D = np.diag([model.dof_damping[0], model.dof_damping[1]])

# ========== 构建状态空间矩阵 ==========
# 状态: x = [body_pos_err, wheel_pos, body_vel, wheel_vel]ᵀ
# d/dt [q; q̇] = [[0, I], [-M⁻¹∂τ_grav/∂q, -M⁻¹D]] [q; q̇] + [[0], [M⁻¹B_g]] u
A = np.zeros((4, 4))
A[0, 2] = 1.0                      # d(body_err)/dt = body_vel
A[1, 3] = 1.0                      # d(wheel_pos)/dt = wheel_vel
A[2:4, 0:2] = -M_inv @ dG_dq       # 重力刚度 → 加速度
A[2:4, 2:4] = -M_inv @ D           # 阻尼 → 加速度

B = np.zeros((4, 1))
B[2:4, 0:1] = M_inv @ np.array([[0], [gear]])

# ========== 能控性检查 ==========
Ctrb = np.hstack([np.linalg.matrix_power(A, i) @ B for i in range(4)])
rank = np.linalg.matrix_rank(Ctrb, tol=1e-8)
print(f"能控性矩阵秩: {rank} (满秩 4 方可控)")
assert rank == 4, "系统不完全可控!"

# ========== 极点配置 ==========
# body 极点: -2±2j   (阻尼 0.7, 自然频率 2.8 rad/s)
# wheel 速度: -6     (快速衰减)
# wheel 位置: -0.5   (慢速, 避免 windup)
poles = np.array([-2+2j, -2-2j, -6, -0.5])
K = place_poles(A, B, poles).gain_matrix  # shape (1, 4)
k1, k2, k3, k4 = K.flatten()
print(f"控制增益: k1={k1:+.4f} (body_err), k2={k2:+.4f} (wheel_pos), "
      f"k3={k3:+.4f} (body_vel), k4={k4:+.4f} (wheel_vel)")
print(f"闭环极点: {np.linalg.eigvals(A - B @ K)}")

# ========== 仿真循环 ==========
max_torque = 100.0

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        # 读取状态
        body_pos = data.jnt("body_joint").qpos[0]
        body_vel = data.jnt("body_joint").qvel[0]
        wheel_pos = data.jnt("wheel_joint").qpos[0]
        wheel_vel = data.jnt("wheel_joint").qvel[0]

        # 构造状态向量 (body_pos_err = target - body_pos)
        x1 = target - body_pos
        x2 = body_vel
        x3 = wheel_pos
        x4 = wheel_vel

        # 控制律: tau = -(k1*x1 + k2*x2 + k3*x3 + k4*x4)
        torque = -(k1 * x1 + k2 * x2 + k3 * x3 + k4 * x4)
        torque = np.clip(torque, -max_torque, max_torque)

        data.ctrl[0] = torque

        print(f"θ_body: {body_pos:.3f},  ω_body: {body_vel:.3f},  "
              f"φ_wheel: {wheel_pos:.1f},  ω_wheel: {wheel_vel:.1f},  "
              f"τ: {torque:.2f}")

        mujoco.mj_step(model, data)
        viewer.sync()
