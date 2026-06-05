"""
IW_tune_lqr.py — LQR + 积分调优 (v4: 抗积分饱和)

需要在长时间仿真中保持稳定。
"""

import mujoco
import numpy as np
from scipy.linalg import solve_continuous_are
from itertools import product

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

GRAV_STIFFNESS = dbias_dq[0, 0]
print(f"重力刚度: {GRAV_STIFFNESS:.4f} Nm/rad")

# 估算重力补偿所需积分: 重力矩 ≈ GRAV_STIFFNESS * (θ - π)
# 稳态时 τ ≈ KI*∫, 所以 ∫ss ≈ GRAV_STIFFNESS * θ_err / KI
# 当 KI=5000, err=0.0005: ∫ss ≈ 0.156 * 0.0005 / 5000 ≈ 1.56e-8 (几乎为零)
# 当 KI=1000, err=0.002: ∫ss ≈ 0.156 * 0.002 / 1000 ≈ 3.12e-7
# 实际 LQR 反馈已经补偿大部分重力矩, 积分只需要补偿非线性项


def run_sim(Q_diag, R_val, KI=16.0, torque_max=50.0,
            duration=300.0, anti_windup=True, integral_limit=20.0, verbose=False):
    """
    Run simulation with optional anti-windup.

    anti_windup: True = conditional integration (stop when saturated)
    """
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
    integral_buf = np.zeros(nsteps)
    torque_buf = np.zeros(nsteps)

    latched = False
    n_stable = 0
    already_warned = False

    for t in range(nsteps):
        θ = data.jnt("body_joint").qpos[0]
        ω_b = data.jnt("body_joint").qvel[0]
        φ_w = data.jnt("wheel_joint").qpos[0]
        ω_w = data.jnt("wheel_joint").qvel[0]
        θ_err = TARGET - θ

        # 条件积分 latch
        if not latched:
            if abs(θ_err) < 0.3 and abs(ω_b) < 1.0:
                n_stable += 1
                if n_stable > 50:
                    integral = 0.0
                    latched = True
                    if verbose:
                        print(f"  ∫ latched t={t*DT:.1f}s")
            else:
                n_stable = 0

        if latched:
            # 抗积分饱和: 饱和时不积分
            if anti_windup:
                torque_raw = -(K[0, 0] * (θ - TARGET) +
                               K[0, 1] * φ_w +
                               K[0, 2] * ω_b +
                               K[0, 3] * ω_w) + KI * integral
                if abs(torque_raw) < torque_max:
                    integral += θ_err * DT
            else:
                integral += θ_err * DT

            # 限幅
            integral = np.clip(integral, -integral_limit, integral_limit)
        else:
            integral = 0.0

        torque = -(K[0, 0] * (θ - TARGET) +
                   K[0, 1] * φ_w +
                   K[0, 2] * ω_b +
                   K[0, 3] * ω_w) + KI * integral
        torque = np.clip(torque, -torque_max, torque_max)

        data.ctrl[0] = torque
        mujoco.mj_step(model, data)
        err_buf[t] = abs(θ_err)
        integral_buf[t] = integral
        torque_buf[t] = torque

        # 检测发散
        if latched and not already_warned and abs(θ_err) > 0.5:
            if verbose:
                print(f"  ⚠ 发散 t={t*DT:.1f}s err={θ_err:.4f} ∫={integral:.3f} τ={torque:.3f}")
            already_warned = True

    nsteps_total = len(err_buf)
    ss_start = int(max(0, nsteps_total - 5.0 / DT))
    ss_err = err_buf[ss_start:]
    rms = np.sqrt(np.mean(ss_err**2))
    mean_err = np.mean(ss_err)

    # 收敛: 最后 5s < 0.01 rad 且没有发散
    late_ok = mean_err < 0.01
    no_diverge = np.max(err_buf[-int(10/DT):]) < 0.1 if latched else False
    conv = late_ok and no_diverge

    # 标注发散时间
    diverge_time = None
    if latched:
        for t in range(len(err_buf)):
            if err_buf[t] > 0.5:
                diverge_time = t * DT
                break

    return {
        "rms": rms, "mean": mean_err,
        "converged": conv,
        "diverge_time": diverge_time,
        "final_int": integral_buf[-1],
        "final_torque": torque_buf[-1],
        "all_err": err_buf,
        "all_int": integral_buf,
        "all_tau": torque_buf,
    }


# =====================================================================
# Step 1: 诊断 KI=5000 发散原因
# =====================================================================
print("\n" + "=" * 60)
print("Step 1: 诊断 KI=5000 长时间发散")
print("=" * 60)

# 不用抗饱和
print("\n  No anti-windup:")
r = run_sim([1000, 0, 300, 5], 200, KI=5000, duration=300.0, anti_windup=False)
for t in [2, 5, 10, 20, 60, 120, 180, 240, 300]:
    idx = int(t / DT)
    if idx < len(r['all_err']):
        ve = np.mean(r['all_err'][max(0,idx-500):idx])
        vi = np.mean(r['all_int'][max(0,idx-500):idx])
        vt = np.mean(r['all_tau'][max(0,idx-500):idx])
        print(f"    t={t:3d}s  err={ve:.6f}  ∫={vi:.3f}  τ={vt:.3f}")
print(f"    Diverged at t={r['diverge_time']:.1f}s, final RMS={r['rms']:.4f}")

# 用抗饱和
print("\n  With anti-windup:")
r2 = run_sim([1000, 0, 300, 5], 200, KI=5000, duration=300.0, anti_windup=True)
for t in [2, 5, 10, 20, 60, 120, 180, 240, 300]:
    idx = int(t / DT)
    if idx < len(r2['all_err']):
        ve = np.mean(r2['all_err'][max(0,idx-500):idx])
        vi = np.mean(r2['all_int'][max(0,idx-500):idx])
        vt = np.mean(r2['all_tau'][max(0,idx-500):idx])
        print(f"    t={t:3d}s  err={ve:.6f}  ∫={vi:.3f}  τ={vt:.3f}")
print(f"    Final RMS={r2['rms']:.8f}, conv={r2['converged']}")

# =====================================================================
# Step 2: 找稳定的最大 KI
# =====================================================================
print("\n" + "=" * 60)
print("Step 2: 搜索稳定且精度高的 KI (with anti-windup, ∫limit=20)")
print("=" * 60)

Qθ_list = [500, 1000, 2000, 5000]
R_list = [50, 100, 200, 400]
KI_list = [500, 1000, 2000, 5000, 10000]

all_results = []
for Qθ, R_val, KI_val in product(Qθ_list, R_list, KI_list):
    r = run_sim([Qθ, 0, 300, 5], R_val, KI=KI_val, duration=120.0, anti_windup=True)

    # 评分: 稳态精度优先, 发散大惩罚
    if r["diverge_time"] is not None:
        score = 1000 + r["diverge_time"]  # 发散: 时间越晚越好
    elif r["converged"]:
        score = r["mean"] * 1000  # 收敛: 看精度
    else:
        score = 10 + r["mean"]  # 其他

    all_results.append((score, Qθ, R_val, KI_val, r))

    err_t = np.mean(r['all_err'][-500:])
    flag = ""
    if r["converged"]:
        flag = " <<< CONV"
    elif r["diverge_time"] is not None and r["diverge_time"] < 120:
        flag = f" diverge@{r['diverge_time']:.0f}s"
    print(f"  Qθ={Qθ:5d} R={R_val:4.0f} KI={KI_val:5.0f}  "
          f"err={err_t:.6f}  ∫={r['final_int']:.2f}  "
          f"conv={r['converged']}{flag}")

all_results.sort(key=lambda x: x[0])

print("\nTop 5:")
for i, (sc, Qθ, Rv, KIv, r) in enumerate(all_results[:5]):
    print(f"  #{i+1}: Qθ={Qθ:5d} R={Rv:4.0f} KI={KIv:5.0f}  "
          f"mean_5s={r['mean']:.8f}  conv={r['converged']}  "
          f"diverge={r['diverge_time']}")

# 额外加: 看看用较小 integral_limit 能否改进
print("\n" + "=" * 60)
print("Step 3: 积分限幅对稳定性的影响 (Qθ=1000, R=200, KI=5000)")
print("=" * 60)

for int_lim in [5, 10, 20, 50, 100]:
    r = run_sim([1000, 0, 300, 5], 200, KI=5000, duration=120.0,
                anti_windup=True, integral_limit=int_lim)
    flag = " CONV" if r["converged"] else f" diverge@{r['diverge_time']}" if r["diverge_time"] else " no_conv"
    print(f"  ∫lim={int_lim:4.0f}  err_5s={r['mean']:.6f}  final_int={r['final_int']:.3f}{flag}")

print("\n" + "=" * 60)
print("Step 4: 最佳参数 600s 长时间验证")
print("=" * 60)

best_score, best_Qθ, best_R, best_KI, best_r = all_results[0]
print(f"候选: Qθ={best_Qθ}, R={best_R}, KI={best_KI}")

r_final = run_sim([best_Qθ, 0, 300, 5], best_R, KI=best_KI,
                  duration=600.0, anti_windup=True, integral_limit=20.0, verbose=True)
for t in [2, 5, 10, 20, 60, 120, 300, 600]:
    idx = int(t / DT)
    if idx < len(r_final['all_err']):
        ve = np.mean(r_final['all_err'][max(0,idx-500):idx])
        vi = np.mean(r_final['all_int'][max(0,idx-500):idx])
        print(f"  t={t:3d}s  err={ve:.8f} rad  ({ve*180/np.pi:.5f}°)  ∫={vi:.3f}")

print(f"\n最终: RMS={r_final['rms']:.8f}  mean={r_final['mean']:.8f}  "
      f"conv={r_final['converged']}  diverge={r_final['diverge_time']}")

Q = np.diag([best_Qθ, 0, 300, 5])
R = np.array([[best_R]])
P = solve_continuous_are(A, B, Q, R)
K = np.linalg.inv(R) @ B.T @ P

print(f"\n{'='*60}")
print(f"最终推荐:")
print(f"  Q = diag([{best_Qθ}, 0, 300, 5])")
print(f"  R = {best_R}")
print(f"  KI = {best_KI}")
print(f"  K = {K.flatten().round(4)}")
print(f"  稳态误差 ≈ {r_final['mean']:.6f} rad ({r_final['mean']*180/np.pi:.4f}°)")

with open("best_lqr_params.txt", "w") as f:
    f.write(f"# LQR params (with anti-windup)\n")
    f.write(f"Q = diag([{best_Qθ}, 0, 300, 5])\n")
    f.write(f"R = {best_R}\n")
    f.write(f"KI = {best_KI}\n")
    f.write(f"Integral_limit = 20.0 (anti-windup)\n")
    f.write(f"Torque_max = 50.0\n")
    f.write(f"K = {K.flatten().round(6).tolist()}\n")
    f.write(f"Steady_Mean = {r_final['mean']:.8f} rad ({r_final['mean']*180/np.pi:.5f} deg)\n")
    f.write(f"Converged = {r_final['converged']}\n")
    f.write(f"Sim_duration = 600s\n")
