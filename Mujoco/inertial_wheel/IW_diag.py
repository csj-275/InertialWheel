"""
快速诊断: 积分收敛速度和方向
"""

import mujoco
import numpy as np
from scipy.linalg import solve_continuous_are

MODEL_PATH = "./scene.xml"
TARGET = np.pi

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)
DT = model.opt.timestep
nv = model.nv
gear = model.actuator("wheel_joint").gear[0]

# ---------- 线性化 ----------
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

# 检查线性化在 θ=π 时的作用力
print(f"dbias_dq:\n{dbias_dq}")
print(f"\nA:\n{A.round(4)}")
print(f"\nB: {B.flatten().round(6)}")

# 检查 LQR 方向: u = -Kx, 当 x=[1,0,0,0]时, u应该为正还是负
Q_test = np.diag([5000, 0, 300, 5])
R_test = np.array([[200]])
P = solve_continuous_are(A, B, Q_test, R_test)
K = np.linalg.inv(R_test) @ B.T @ P
print(f"\nLQR K = {K.flatten().round(4)}")
print(f"u when x=[+1, 0,0,0] = {-K[0,0]:+.4f}")
print(f"u when x=[π+0.1, 0,0,0] = {-K[0,0]*0.1:+.4f}")
print(f"  -> 如果θ > π (前倾), 需要负力矩(向后?)")
print(f"  -> 正力矩应该产生什么影响? 检查B...")
print(f"  正 τ → body acc = {B[2,0]:.4f}, wheel acc = {B[3,0]:.4f}")

# 开环测试: 正力矩使 body 如何加速?
data.qpos[0] = TARGET
data.qpos[1] = 0
data.qvel[:] = 0
mujoco.mj_forward(model, data)
data.ctrl[0] = 1.0  # +1 Nm
mujoco.mj_step(model, data)
print(f"\n开环step: 施加 +1 Nm, 一帧后: body_acc≈{data.qacc[0]:+.6f}, wheel_acc≈{data.qacc[1]:+.6f}")

# ============ 积分方向测试 + 大 KI ============
def run_fast(Q_diag, R_val, KI, integral_sign=1.0, duration=30.0):
    Q = np.diag(Q_diag)
    R = np.array([[R_val]])
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.inv(R) @ B.T @ P

    data.qpos[0] = 0.0
    data.qpos[1] = 0.0
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    integral = 0.0
    nsteps = int(duration / DT)
    err_buf = np.zeros(nsteps)

    latched = False
    n_stable = 0

    for t in range(nsteps):
        θ = data.jnt("body_joint").qpos[0]
        ω_b = data.jnt("body_joint").qvel[0]
        φ_w = data.jnt("wheel_joint").qpos[0]
        ω_w = data.jnt("wheel_joint").qvel[0]
        θ_err = TARGET - θ

        if not latched:
            if abs(θ_err) < 0.3 and abs(ω_b) < 1.0:
                n_stable += 1
                if n_stable > 50:
                    integral = -(K[0,0]*θ_err + K[0,1]*φ_w + K[0,2]*ω_b + K[0,3]*ω_w) / (KI*integral_sign)
                    integral = np.clip(integral, -500, 500)
                    latched = True
            else:
                n_stable = 0

        if latched:
            integral += θ_err * DT * integral_sign
            integral = np.clip(integral, -500, 500)
        else:
            integral = 0.0

        torque = -(K[0,0]*θ_err + K[0,1]*φ_w + K[0,2]*ω_b + K[0,3]*ω_w) + KI * integral
        torque = np.clip(torque, -50, 50)

        data.ctrl[0] = torque
        mujoco.mj_step(model, data)
        err_buf[t] = abs(θ_err)

    ss = int(nsteps - 5/DT)
    return np.mean(err_buf[ss:]), err_buf


# 测试积分方向和增益
print("\n" + "=" * 60)
print("积分方向和大 KI 测试")
print("=" * 60)

tests = [
    ("基准  KI=32  sign=+1", [5000, 0, 300, 5], 200, 32, 1.0),
    ("大 KI=100 sign=+1", [5000, 0, 300, 5], 200, 100, 1.0),
    ("大 KI=300 sign=+1", [5000, 0, 300, 5], 200, 300, 1.0),
    ("大 KI=500 sign=+1", [5000, 0, 300, 5], 200, 500, 1.0),
    ("KI=32  sign=-1", [5000, 0, 300, 5], 200, 32, -1.0),
    ("KI=100 sign=-1", [5000, 0, 300, 5], 200, 100, -1.0),
]

for name, Qd, Rv, KIv, sign in tests:
    rms_e, err = run_fast(Qd, Rv, KIv, sign, duration=30.0)
    # Convergence check
    early = np.mean(err[2000:3000])
    mid = np.mean(err[7000:8000])
    late = np.mean(err[12000:13000]) if len(err) > 13000 else rms_e
    print(f"  {name:30s}  RMS_5s={rms_e:.6f}  err_4-6s={mid:.6f}  err_24-26s={late:.6f}")

# 长时间验证最佳
print("\n" + "=" * 60)
print("长时间验证 (120s) 最佳参数")
print("=" * 60)
rms, err120 = run_fast([5000, 0, 300, 5], 200, 500, 1.0, duration=120.0)
for t in [10, 20, 30, 60, 90, 120]:
    idx = int(t/0.002)
    val = np.mean(err120[max(0,idx-500):idx])
    print(f"  t={t:3d}s  err_mean={val:.6f} rad ({val*180/np.pi:.4f}°)")
print(f"  最后5s RMS={rms:.8f} rad")
