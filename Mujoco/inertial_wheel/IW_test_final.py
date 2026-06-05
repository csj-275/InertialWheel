"""
IW_test_final.py — 最终长时间验证, 无 viewer
"""

import mujoco
import numpy as np
from scipy.linalg import solve_continuous_are

MODEL_PATH = "./scene.xml"
TARGET = np.pi
KI = 5000.0

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)
DT = model.opt.timestep
nv = model.nv
gear = model.actuator("wheel_joint").gear[0]

# 线性化
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

Q = np.diag([1000.0, 0.0, 300.0, 5.0])
R = np.array([[200.0]])
P = solve_continuous_are(A, B, Q, R)
K = np.linalg.inv(R) @ B.T @ P

# Reset
data.qpos[0] = 0.0
data.qpos[1] = 0.0
data.qvel[:] = 0.0
mujoco.mj_forward(model, data)

integral = 0.0
nsteps = int(300.0 / DT) + 1
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
                integral = 0.0
                latched = True
                print(f"Latched at t={t*DT:.2f}s")
        else:
            n_stable = 0

    if latched:
        integral += θ_err * DT
    else:
        integral = 0.0

    torque = -(K[0, 0] * (θ - TARGET) +
               K[0, 1] * φ_w +
               K[0, 2] * ω_b +
               K[0, 3] * ω_w) + KI * integral
    torque = np.clip(torque, -50, 50)

    data.ctrl[0] = torque
    mujoco.mj_step(model, data)
    err_buf[t] = abs(θ_err)

# 结果
print("\n" + "=" * 60)
print(f"总仿真时间: {nsteps*DT:.0f}s")

for t in [2, 5, 10, 20, 30, 60, 120, 180, 240, 300]:
    idx = int(t / DT)
    if idx < len(err_buf):
        ve = np.mean(err_buf[max(0,idx-500):idx])
        deg = ve * 180 / np.pi
        print(f"  t={t:3d}s  err={ve:.8f} rad  ({deg:.5f}°)")

ss = int(nsteps - 5/DT)
rms = np.sqrt(np.mean(err_buf[ss:]**2))
mean_e = np.mean(err_buf[ss:])
max_e = np.max(err_buf[ss:])
print(f"\n最后 5s: RMS={rms:.10f} rad  Mean={mean_e:.10f} rad  Max={max_e:.10f} rad")
print(f"         RMS={rms*180/np.pi:.6f}°")
print(f"\n校验: 稳态角度误差 ≈ {mean_e*180/np.pi:.4f}°")
