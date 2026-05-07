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

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on creating a cartpole base environment.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to spawn.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import torch
from isaaclab.envs import ManagerBasedRLEnv
# from isaaclab_tasks.manager_based.manipulation.place.config.agibot.place_toy2box_rmp_rel_env_cfg import RmpFlowAgibotPlaceToy2BoxEnvCfg
# from isaaclab_tasks.manager_based.piper_grab.grab_joint_pos_env_cfg import PiperGrabEnvCfg
from InertialWheel.tasks.manager_based.inertialwheel.inertialwheel_env_cfg import InertialwheelEnvCfg

def main():
    """Main function."""
    # parse the arguments
    env_cfg = InertialwheelEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    # setup base environment
    env = ManagerBasedRLEnv(cfg=env_cfg)
    # robot = env.scene["robot"]
    count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            # reset
            if count % 300 == 0:
                count = 0
                env.reset()
            
            # arm_actions = torch.zeros(env.num_envs, 6) 
            # arm_actions[:, 0] = 0.0
            # gripper_action = 10 * torch.full((env.num_envs, 1), 0)
            # joint_actions = torch.cat([arm_actions, gripper_action], dim=1)

            joint_actions = torch.zeros_like(env.action_manager.action)

            # step the environment
            obs = env.step(joint_actions)
            # update counter
            count += 1

    # close the environment
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
