"""
sim2sim_inertialwheel.py

Transfer a policy trained in Isaac Lab (Isaac-Inertialwheel-v0) to MuJoCo.

Physics alignment verified (2026-06-11):
  - mass / inertia / center-of-mass: ✓ identical
  - joint positions (localPos0 / origin):     ✓ identical
  - body_joint axis (0,0,1):                  ✓ identical
  - wheel_joint axis: USD (0,0,1) vs MuJoCo (0,0,-1)
        → action sign is flipped by default (--action_sign -1)
  - torque limit: 50 N·m in both
  - body_joint damping: 0 in both

Usage:
    # Latest run, with viewer
    python scripts/sim2sim_inertialwheel.py

    # Specific run, headless
    python scripts/sim2sim_inertialwheel.py --run 2026-06-05_11-52-51 --no_render
"""

import argparse
import os

import numpy as np
import torch

import mujoco
import mujoco.viewer

# =============================================================
# CLI
# =============================================================
parser = argparse.ArgumentParser(description="sim2sim: Isaac Lab → MuJoCo")
parser.add_argument("--run", type=str, default=None,
                    help="Training run folder (default: latest)")
parser.add_argument("--action_sign", type=int, default=-1, choices=[1, -1],
                    help="Torque sign. -1 because MuJoCo wheel axis = (0,0,-1) vs USD (0,0,1).")
parser.add_argument("--action_scale", type=float, default=50.0,
                    help="Scale policy output [-1,1] → torque. Matches JointEffortActionCfg(scale=50).")
parser.add_argument("--xml", type=str, default="./Mujoco/inertial_wheel/scene.xml",
                    help="MuJoCo scene XML")
parser.add_argument("--max_steps", type=int, default=4000,
                    help="Max simulation steps (0 = infinite)")
parser.add_argument("--no_render", action="store_true",
                    help="Disable viewer (headless)")
parser.add_argument("--random_init", action="store_true",
                    help="Random initial pose (default: start at downward θ=0)")
args = parser.parse_args()

# =============================================================
# Locate policy
# =============================================================
log_root = "logs/rsl_rl/inertialwheel"
if args.run is None:
    runs = sorted(os.listdir(log_root))
    if not runs:
        raise FileNotFoundError(f"No training runs found under {log_root}")
    args.run = runs[-1]

policy_path = os.path.join(log_root, args.run, "exported", "policy.pt")
if not os.path.exists(policy_path):
    policy_path = os.path.join(log_root, args.run, "exported", "policy.onnx")
    if not os.path.exists(policy_path):
        raise FileNotFoundError(f"No exported policy found under {log_root}/{args.run}/exported/")

print(f"[Info] Loading policy: {policy_path}")

# =============================================================
# Load policy
# =============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
if policy_path.endswith(".onnx"):
    import onnxruntime as ort
    sess = ort.InferenceSession(policy_path)
    def policy(obs_np):
        inp = {sess.get_inputs()[0].name: obs_np.astype(np.float32)}
        return sess.run(None, inp)[0]
else:
    policy_jit = torch.jit.load(policy_path, map_location=device)
    policy_jit.eval()
    def policy(obs_np):
        obs_t = torch.from_numpy(obs_np).float().to(device)
        with torch.inference_mode():
            return policy_jit(obs_t).cpu().numpy()

# =============================================================
# MuJoCo model
# =============================================================
model = mujoco.MjModel.from_xml_path(os.path.abspath(args.xml))
data = mujoco.MjData(model)

# Joint IDs
body_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "body_joint")
wheel_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "wheel_joint")
wheel_act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "wheel_joint")
assert wheel_act >= 0, "actuator 'wheel_joint' not found"

# State indices
bq_adr = model.jnt_qposadr[body_jid]
bv_adr = model.jnt_dofadr[body_jid]
wq_adr = model.jnt_qposadr[wheel_jid]
wv_adr = model.jnt_dofadr[wheel_jid]

print(f"[Info] body_joint qpos_adr={bq_adr} dof_adr={bv_adr}")
print(f"[Info] wheel_joint qpos_adr={wq_adr} dof_adr={wv_adr}")

# =============================================================
# 5-D observation ─ same pipeline as training
#   [sin(body_q), cos(body_q), wheel_q, body_v, wheel_v]
# =============================================================
def get_obs():
    bq = data.qpos[bq_adr]
    bv = data.qvel[bv_adr]
    wq = data.qpos[wq_adr]
    wv = data.qvel[wv_adr]
    return np.array([np.sin(bq), np.cos(bq), wq, bv, wv], dtype=np.float32)

# =============================================================
# Main loop
# =============================================================
def main():
    # Initial condition: downward (θ=0), small perturbation if random
    rng = np.random.RandomState(42)
    if args.random_init:
        data.qpos[bq_adr] = rng.uniform(-0.5, 0.5)
        data.qvel[bv_adr] = rng.uniform(-0.5, 0.5)
    data.qpos[wq_adr] = data.qvel[wv_adr] = 0.0
    mujoco.mj_forward(model, data)

    step = 0

    if args.no_render:
        while step < args.max_steps or args.max_steps == 0:
            action = policy(get_obs()[np.newaxis])           # [1,5] → [1,1]
            torque = float(action[0, 0]) * args.action_scale * args.action_sign
            data.ctrl[wheel_act] = torque
            mujoco.mj_step(model, data)

            if step % 100 == 0:
                bq = data.qpos[bq_adr]
                upright = (np.cos(bq - np.pi) + 1) / 2
                print(f"step={step:5d}  body_q={bq:+.3f}  upright={upright:.3f}  torque={torque:+.2f}")
            step += 1

        print(f"\n[Done] {step} steps simulated")
        return

    # ── With viewer ──
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            action = policy(get_obs()[np.newaxis])
            torque = float(action[0, 0]) * args.action_scale * args.action_sign
            data.ctrl[wheel_act] = torque
            mujoco.mj_step(model, data)

            if step % 50 == 0:
                bq = data.qpos[bq_adr]
                wq = data.qpos[wq_adr]
                bv = data.qvel[bv_adr]
                wv = data.qvel[wv_adr]
                upright = (np.cos(bq - np.pi) + 1) / 2
                print(f"step={step:5d}  body_q={bq:+.3f}  wheel_q={wq:+.3f}  "
                      f"body_v={bv:+.3f}  wheel_v={wv:+.3f}  "
                      f"torque={torque:+.2f}  upright={upright:.3f}")

            viewer.sync()
            step += 1
            if 0 < args.max_steps <= step:
                print(f"[Done] reached max_steps={args.max_steps}")
                break


if __name__ == "__main__":
    print("=" * 60)
    print(f"  Run:   {args.run}")
    print(f"  XML:   {args.xml}")
    print(f"  Scale: {args.action_scale}  Sign: {args.action_sign}")
    print(f"  View:  {'on' if not args.no_render else 'off'}")
    print("=" * 60)
    main()
