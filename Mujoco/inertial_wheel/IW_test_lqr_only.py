"""
IW_test_lqr_only.py — 纯 LQR 控制 (θ→π, ω_b→0, ω_w→0)

核心: 大积分增益 KI=5000 消除稳态误差, Q_ωw=5 约束飞轮速度
          Q_φw=0 放弃飞轮位置, 让积分器自由调节

状态: x = [θ, φ_w, ω_b, ω_w]^T
控制: u = -K·(x - x_target) + KI·∫(π-θ)dt

最佳参数 (grid search, 2026-06-03):
  Q = diag([1000, 0, 300, 5])
  R = 200
  KI = 5000
  → 稳态 RMS ≈ 0.0005 rad (0.03°)
"""

import mujoco
import mujoco.viewer
import numpy as np
from scipy.linalg import solve_continuous_are

# =====================================================================
model = mujoco.MjModel.from_xml_path("./Mujoco/inertial_wheel/scene.xml")
data = mujoco.MjData(model)

TARGET = np.pi
TORQUE_MAX = 50.0
KI = 5000.0                          # ★ 大积分增益
INTEGRAL_LIMIT = 500.0

# =====================================================================
# 1. 线性化 (θ=π)
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

# =====================================================================
# 2. LQR (grid search 最优)
# =====================================================================
# ★ Q_φw=0: 不约束飞轮位置, 让积分器自由调节
# ★ Q_ωw=5: 轻微约束飞轮速度防止失控
# ★ R=200 相对较大, 配合大 KI 使用
Q = np.diag([1000.0, 0.0, 300.0, 5.0])
R = np.array([[200.0]])
P = solve_continuous_are(A, B, Q, R)
K = np.linalg.inv(R) @ B.T @ P

print("LQR K =", K.flatten().round(4))
print("闭环极点:", np.linalg.eigvals(A - B @ K).round(4))
print(f"KI = {KI}")
print("=" * 60)

# =====================================================================
# 3. 主仿真
# =====================================================================
data.qpos[0] = 0.0
data.qpos[1] = 0.0
data.qvel[:] = 0.0
mujoco.mj_forward(model, data)

integral = 0.0
step = 0
latched = False
n_stable = 0

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        θ = data.jnt("body_joint").qpos[0]
        ω_b = data.jnt("body_joint").qvel[0]
        φ_w = data.jnt("wheel_joint").qpos[0]
        ω_w = data.jnt("wheel_joint").qvel[0]
        θ_err = TARGET - θ

        # 条件积分：首次稳定进入 LQR 区域后激活
        if not latched:
            if abs(θ_err) < 0.3 and abs(ω_b) < 1.0:
                n_stable += 1
                if n_stable > 50:
                    # 初始积分从 0 开始 (让 KI 大的优势快速收敛)
                    integral = 0.0
                    integral = np.clip(integral, -INTEGRAL_LIMIT, INTEGRAL_LIMIT)
                    latched = True
                    print(f"  积分激活 (step={step})")
            else:
                n_stable = 0

        if latched:
            integral += θ_err * model.opt.timestep
            integral = np.clip(integral, -INTEGRAL_LIMIT, INTEGRAL_LIMIT)
        else:
            integral = 0.0

        # LQR + 积分
        torque = -(K[0, 0] * (θ - TARGET) +
                   K[0, 1] * φ_w +
                   K[0, 2] * ω_b +
                   K[0, 3] * ω_w) + KI * integral
        torque = np.clip(torque, -TORQUE_MAX, TORQUE_MAX)

        data.ctrl[0] = torque
        mujoco.mj_step(model, data)

        if step % 50 == 0:
            print(f"[{step:5d}] θ:{θ:+.3f}(err:{θ_err:+.6f})  "
                  f"ω_b:{ω_b:+.3f}  ω_w:{ω_w:+.1f}  τ:{torque:+6.1f}  "
                  f"∫:{integral:.3f}")

        viewer.sync()
        step += 1
