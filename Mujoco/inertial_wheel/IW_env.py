"""
IW_env.py — 惯性轮摆 Gymnasium 环境
====================================

系统: 惯性轮倒立摆 (MuJoCo), 目标 θ=π
动作: 轮子电机力矩 [-50, 50] N·m
观测: [sin(θ), cos(θ), φ_w, ω_b, ω_w]  (归一化)
"""

import math
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco

# ============================================================
MODEL_PATH = "./Mujoco/inertial_wheel/scene.xml"
TARGET = np.pi
TORQUE_MAX = 50.0
DT = 0.002

# ============================================================


class InertialWheelEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode: str | None = None, max_steps: int = 2000):
        super().__init__()
        self.max_steps = max_steps
        self.render_mode = render_mode

        # MuJoCo
        self.model = mujoco.MjModel.from_xml_path(MODEL_PATH)
        self.data = mujoco.MjData(self.model)

        # 动作: 力矩
        self.action_space = spaces.Box(
            low=-TORQUE_MAX, high=TORQUE_MAX, shape=(1,), dtype=np.float32
        )

        # 观测: [sin(θ), cos(θ), φ_w, ω_b, ω_w]
        self.observation_space = spaces.Box(
            low=-np.array([1.0, 1.0, -np.inf, -np.inf, -np.inf]),
            high=np.array([1.0, 1.0, np.inf, np.inf, np.inf]),
            dtype=np.float32,
        )

        self._viewer = None
        self.step_count = 0

    def _get_obs(self):
        θ = self.data.jnt("body_joint").qpos[0]
        ω_b = self.data.jnt("body_joint").qvel[0]
        φ_w = self.data.jnt("wheel_joint").qpos[0]
        ω_w = self.data.jnt("wheel_joint").qvel[0]
        return np.array(
            [math.sin(θ), math.cos(θ), φ_w, ω_b, ω_w], dtype=np.float32
        )

    def _get_info(self):
        θ = self.data.jnt("body_joint").qpos[0]
        ω_b = self.data.jnt("body_joint").qvel[0]
        ω_w = self.data.jnt("wheel_joint").qvel[0]
        return dict(
            theta=θ,
            theta_err=TARGET - θ,
            omega_body=ω_b,
            omega_wheel=ω_w,
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # 从下垂附近随机起始
        θ0 = self.np_random.uniform(-0.5, 0.5)
        self.data.jnt("body_joint").qpos[0] = θ0
        self.data.jnt("wheel_joint").qpos[0] = 0.0
        self.data.jnt("body_joint").qvel[0] = self.np_random.uniform(-1, 1)
        self.data.jnt("wheel_joint").qvel[0] = self.np_random.uniform(-1, 1)
        mujoco.mj_forward(self.model, self.data)
        self.step_count = 0
        return self._get_obs(), self._get_info()

    def step(self, action):
        torque = np.clip(action[0], -TORQUE_MAX, TORQUE_MAX)
        self.data.ctrl[0] = torque
        mujoco.mj_step(self.model, self.data)
        self.step_count += 1

        obs = self._get_obs()
        info = self._get_info()
        θ_err = info["theta_err"]
        ω_b = info["omega_body"]
        ω_w = info["omega_wheel"]

        # ---- 奖励设计 ----
        # 主要: 角度误差 (目标 π)
        angle_cost = abs(θ_err)

        # 次要: 速度惩罚
        vel_cost = 0.05 * abs(ω_b) + 0.01 * abs(ω_w)

        # 动作惩罚 (防止抖动)
        act_cost = 0.001 * abs(torque)

        reward = -(angle_cost + vel_cost + act_cost)

        # 到达 π 附近给予额外奖励
        if abs(θ_err) < 0.2 and abs(ω_b) < 0.5:
            reward += 2.0
        if abs(θ_err) < 0.05 and abs(ω_b) < 0.1:
            reward += 5.0

        # 终止条件
        terminated = False
        truncated = self.step_count >= self.max_steps

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            from mujoco import viewer

            if self._viewer is None:
                self._viewer = viewer.launch_passive(self.model, self.data)
            self._viewer.sync()
        # rgb_array 模式下返回空 (需要离屏渲染)
        return None

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
