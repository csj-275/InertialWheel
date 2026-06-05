# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from . import mdp

##
# Pre-defined configs
##

from InertialWheel.assets.InertialWheel import INERTIAL_WHEEL_PENDULUM_CFG  # isort:skip

##
# Scene definition
##


@configclass
class InertialwheelSceneCfg(InteractiveSceneCfg):
    """Configuration for a InertialWheelPendulum scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    # robot
    robot: ArticulationCfg = INERTIAL_WHEEL_PENDULUM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    joint_effort = mdp.JointEffortActionCfg(
        asset_name="robot",
        joint_names=["wheel_joint"],
        scale=50.0)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # sin/cos encoding avoids the pi/-pi discontinuity
        body_sin_cos = ObsTerm(func=mdp.joint_sin_cos_pos,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["body_joint"])})
        # wheel joint raw position (not wrapped at pi, so fine as raw)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["wheel_joint"])})
        # velocities for all joints
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # reset body_joint: random near downward (0) so agent must learn to swing up
    reset_body_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["body_joint"]),
            "position_range": (-0.5, 0.5),
            "velocity_range": (-0.5, 0.5),
        },
    )

    # reset wheel_joint: random small rotation, small velocity
    reset_wheel_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["wheel_joint"]),
            "position_range": (-0.5, 0.5),
            "velocity_range": (-0.5, 0.5),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # (1) Smooth cos-based reward: 1 at upright (π), 0 at downward (0)
    upright = RewTerm(
        func=mdp.upright_reward_cos,
        weight=10.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["body_joint"])},
    )
    # (2) Penalize body angular velocity
    body_vel = RewTerm(
        func=mdp.joint_vel_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["body_joint"])},
    )
    # (3) Penalize excessive wheel velocity
    wheel_vel = RewTerm(
        func=mdp.joint_vel_l1,
        weight=-0.005,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["wheel_joint"])},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    # Time out only — no fall termination.
    # This is a swing-up task: the pendulum starts near downward and needs
    # the full 360° range to reach upright. Any angle-based termination
    # would cut episodes short before learning can happen.
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


##
# Environment configuration
##


@configclass
class InertialwheelEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: InertialwheelSceneCfg = InertialwheelSceneCfg(num_envs=4096, env_spacing=4.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 5
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation