import mujoco
import mujoco.viewer
import numpy as np

model = mujoco.MjModel.from_xml_path("./Mujoco/inertial_wheel/scene.xml")
data = mujoco.MjData(model)

# PID gains
Kp = 1000
Ki = 8000
Kd = 200

target = np.pi  # target body_joint angle

integral = 0.0
last_error = 0.0
dt = model.opt.timestep

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        # current state
        pos = data.jnt("body_joint").qpos[0]
        vel = data.jnt("body_joint").qvel[0]

        # angular error, wrapped to [-pi, pi]
        # error = (target - pos + np.pi) % (2 * np.pi) - np.pi
        error = target - pos
        integral += error * dt
        # anti-windup
        integral = np.clip(integral, -1000.0, 1000.0)

        derivative = (error - last_error) / dt
        last_error = error
        print(f"pos: {pos:.3f}, error: {error:.3f}")
        # PID output -> wheel torque
        torque = Kp * error + Ki * integral + Kd * derivative
        data.ctrl[0] = torque

        mujoco.mj_step(model, data)
        viewer.sync()
