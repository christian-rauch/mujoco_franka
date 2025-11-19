#!/usr/bin/env python3
import mujoco
import mujoco.viewer
import cv2
import numpy as np

# "link7" -> "hand" in MJCF corresponds to "panda_link7" -> "panda_hand" in URDF

# See documentation about the camera optical frame convention:
# Note that specifically for cameras, the xyaxes attribute is semantically convenient
# as the X and Y axes correspond to the directions “right” and “up” in pixel space, respectively.

# $ ros2 run tf2_ros tf2_echo panda_link7 panda_hand
# - Translation: [0.000, 0.000, 0.107]
# - Rotation: in Quaternion [0.000, 0.000, -0.383, 0.924]
# - Rotation: in RPY (radian) [0.000, 0.000, -0.785]
# - Rotation: in RPY (degree) [0.000, 0.000, -45.000]
# - Matrix:
# 0.707  0.707  0.000  0.000
# -0.707  0.707 -0.000  0.000
# -0.000  0.000  1.000  0.107
# 0.000  0.000  0.000  1.000

# $ ros2 run tf2_ros tf2_echo panda_link7 camera_color_optical_frame
# - Translation: [0.060, -0.018, 0.160]
# - Rotation: in Quaternion [0.000, -0.000, 0.383, 0.924]
# - Rotation: in RPY (radian) [0.000, -0.000, 0.785]
# - Rotation: in RPY (degree) [0.000, -0.000, 45.000]
# - Matrix:
# 0.707 -0.707 -0.000  0.060
# 0.707  0.707 -0.000 -0.018
# 0.000  0.000  1.000  0.160
# 0.000  0.000  0.000  1.000

def main(args=None):
    m = mujoco.MjModel.from_xml_path("franka_emika_panda/mjx_single_cube.xml")
    d = mujoco.MjData(m)

    camera_name = "end_effector_camera"

    # ready state
    ready_state = np.array([0., -1/4 * np.pi, 0., -3/4 * np.pi, 0., 1/2 * np.pi, 1/4 * np.pi])
    m.qpos0[:7] = -ready_state

    cam_renderer = mujoco.Renderer(m, 720, 1280)

    with mujoco.viewer.launch_passive(m, d) as viewer:
        while viewer.is_running():
            mujoco.mj_step(m, d)
            viewer.sync()
            cam_renderer.update_scene(d, camera=camera_name)
            colour = cam_renderer.render()
            cam_renderer.enable_depth_rendering()
            depth = cam_renderer.render()
            cam_renderer.disable_depth_rendering()

            max_depth = 1 # cutoff-depth in metre
            colour_vis = cv2.cvtColor(colour, cv2.COLOR_RGB2BGR).astype(np.uint8)
            depth_vis = cv2.cvtColor((depth*255)/max_depth, cv2.COLOR_RGB2BGR).astype(np.uint8)

            cv2.imshow("End Effector Camera (RGB-D)", np.hstack((colour_vis, depth_vis)))
            cv2.waitKey(1)


if __name__ == '__main__':
    main()
