# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
# created by csj, 2026-05-06
"""Configuration for a simple InertialWheelPendulum robot."""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

##
# Configuration
##

INERTIAL_WHEEL_PENDULUM_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="./usd/InertialWheel.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=5.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0), 
        joint_pos={"body_joint": 0.0, 
                   "wheel_joint": 0.0}
    ),
    # 关节驱动器
    actuators={
        "wheel_actuator": ImplicitActuatorCfg(
            joint_names_expr=["wheel_joint"],
            effort_limit_sim=400.0,
            stiffness=0.0, # 位置控制的刚度，力控设为0
            damping=10.0, # 阻尼
        ),
    },
)
"""Configuration for a simple InertialWheelPendulum robot."""
