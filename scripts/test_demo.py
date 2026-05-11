# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to create a simple environment with a cartpole. It combines the concepts of
scene, action, observation and event managers to create an environment.

.. code-block:: bash

    ./isaaclab.sh -p scripts/tutorials/03_envs/create_piper_base_env.py --num_envs 32

"""

"""Launch Isaac Sim Simulator first."""


import argparse

import torch
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="PID control demo for InertialWheel.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to spawn.")
parser.add_argument("--target", type=float, default=0.0, help="Target body joint angle in radians (within [-pi, pi]).")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
from isaaclab.envs import ManagerBasedRLEnv
# from isaaclab_tasks.manager_based.manipulation.place.config.agibot.place_toy2box_rmp_rel_env_cfg import RmpFlowAgibotPlaceToy2BoxEnvCfg
# from isaaclab_tasks.manager_based.piper_grab.grab_joint_pos_env_cfg import PiperGrabEnvCfg
from InertialWheel.tasks.manager_based.inertialwheel.inertialwheel_env_cfg import InertialwheelEnvCfg


def wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    """Wrap angle to [-pi, pi]."""
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def main():
    """Main function."""
    # parse the arguments
    env_cfg = InertialwheelEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    # setup base environment
    env = ManagerBasedRLEnv(cfg=env_cfg)
    robot = env.scene["robot"]

    # PID gains
    kp = 100.0   # proportional gain
    ki = 0.5     # integral gain
    kd = 20.0    # derivative gain

    # Target angle for body_joint, clamped to [-pi, pi]
    # target_body_angle = max(-torch.pi, min(torch.pi, torch.tensor(args_cli.target)))
    target_body_angle = torch.pi
    print(f"Target body_joint angle: {target_body_angle:.3f} rad")

    # Joint indices
    body_idx = 0   # body_joint

    # Control dt
    dt = env_cfg.sim.dt * env_cfg.decimation  # 1/60 s

    # Integral term (per environment)
    integral = torch.zeros(args_cli.num_envs, device=env_cfg.sim.device)

    # Torque limit for wheel_joint
    max_torque = 4000.0

    step_count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            # --- Read body_joint state ---
            joint_pos = robot.data.joint_pos      # (num_envs, 2)
            joint_vel = robot.data.joint_vel      # (num_envs, 2)

            body_pos = joint_pos[:, body_idx]     # (num_envs,)
            body_vel = joint_vel[:, body_idx]     # (num_envs,)

            # --- Compute PID error ---
            # Error wrapped to [-pi, pi] for angular correctness
            # error = wrap_to_pi(target_body_angle - body_pos)
            error = target_body_angle - body_pos
            # Integral with anti-windup (clamp integral term)
            integral += error * dt
            integral = torch.clamp(integral, -max_torque / ki, max_torque / ki)

            # Derivative on measurement (negative of velocity)
            derivative = -body_vel

            # --- PID output: torque for wheel_joint ---
            torque = kp * error + ki * integral + kd * derivative

            # Clamp torque to actuator limit
            # torque = torch.clamp(torque, -max_torque, max_torque)

            # --- Step environment ---
            joint_actions = torque.unsqueeze(-1)  # (num_envs, 1)
            _ = env.step(joint_actions)

            # --- Logging ---
            if step_count % 60 == 0:  # log every ~1s
                print(f"[{step_count}] body_pos: {body_pos[0].item():+.3f} rad, "
                      f"target: {target_body_angle:+.3f} rad, "
                      f"error: {error[0].item():+.3f} rad, "
                      f"torque: {torque[0].item():+.1f} Nm")
            step_count += 1

    # close the environment
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
