# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # wrap the joint positions to (-pi, pi)
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
    # compute the reward
    return torch.sum(torch.square(joint_pos - target), dim=1)


def upright_bonus(env: ManagerBasedRLEnv, threshold: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Bonus for keeping the pendulum near upright."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
    err = torch.abs(joint_pos - math.pi)
    return (err < threshold).float().squeeze(-1)


def joint_sin_cos_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Observation term: sin and cos of joint positions (avoids wrap discontinuity)."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    return torch.cat([torch.sin(joint_pos), torch.cos(joint_pos)], dim=-1)


def upright_reward_cos(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Smooth reward in [0, 1]: 1 at upright (θ=target), 0 at downward (θ=target-π).

    Uses cos(θ - target) so the gradient is smooth everywhere
    with no discontinuity at the wrap boundary.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    cos_dev = torch.cos(joint_pos - target)
    return ((cos_dev + 1.0) / 2.0).squeeze(-1)


def body_angle_out_of_range(env: ManagerBasedRLEnv, threshold: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Termination: pendulum has deviated too far from upright.

    Note: this only terminates when the pendulum passes through upright
    and continues past it on the other side — it does NOT terminate
    when starting from the downward position (which is closer to the
    wall than threshold from pi, but the wall is on the other side).

    The check: cos(theta) < cos(threshold). When threshold=2.8 rad (~160 deg),
    cos(2.8) ≈ -0.94, so the pendulum must pass through upright and go
    almost full circle before terminating.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    # Use cos distance from pi so the check is symmetric around upright
    cos_theta = torch.cos(joint_pos)
    cos_threshold = torch.cos(torch.tensor(threshold, device=joint_pos.device))
    return (cos_theta < cos_threshold).squeeze(-1)
