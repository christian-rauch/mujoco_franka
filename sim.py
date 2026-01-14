#!/usr/bin/env python3
import mujoco
import mujoco.viewer
import cv2
import numpy as np
import open3d as o3d
import os

# Open3D fails to run on Wayland
os.environ["XDG_SESSION_TYPE"] = "x11"

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
    cam_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)

    # ready state
    ready_state = np.array([0., -1/4 * np.pi, 0., -3/4 * np.pi, 0., 1/2 * np.pi, 1/4 * np.pi])
    m.qpos0[:7] = -ready_state

    width, height = m.cam_resolution[cam_id]
    cam_renderer = mujoco.Renderer(m, height, width)

    # Get camera intrinsics directly from MuJoCo
    # cam_intrinsic stores [fx, fy, ox, oy] where ox, oy are offsets from the center
    intr = m.cam_intrinsic[cam_id]
    fx, fy, ox, oy = intr
    cx = width / 2 + ox
    cy = height / 2 + oy
    intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

    # Open3D Visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window("Point Cloud", width=640, height=480)
    pcd = o3d.geometry.PointCloud()
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.5, origin=[0, 0, 0]
    )
    vis.add_geometry(pcd)
    vis.add_geometry(frame)

    with mujoco.viewer.launch_passive(m, d) as viewer:
        while viewer.is_running():
            mujoco.mj_step(m, d)
            viewer.sync()
            cam_renderer.update_scene(d, camera=camera_name)
            colour = cam_renderer.render()
            cam_renderer.enable_depth_rendering()
            depth = cam_renderer.render()
            cam_renderer.disable_depth_rendering()

            max_depth = 2 # cutoff-depth in metre
            colour_vis = cv2.cvtColor(colour, cv2.COLOR_RGB2BGR).astype(np.uint8)
            depth_vis = cv2.cvtColor((depth*255)/max_depth, cv2.COLOR_RGB2BGR).astype(np.uint8)

            # RGB-D image
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(colour),
                o3d.geometry.Image(depth.astype(np.float32)),
                depth_scale=1.0,
                convert_rgb_to_intensity=False,
            )

            # back-project to point cloud
            new_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)

            # transform point cloud from camera frame to world frame
            T_world_cam = np.eye(4)
            T_world_cam[:3, :3] = d.cam_xmat[cam_id].reshape(3, 3)
            T_world_cam[:3, 3] = d.cam_xpos[cam_id]

            # change from OpenCV to MuJuCo camera frame conventions:
            # OpenCV: X right, Y down, Z forward
            # MuJoCo: X right, Y up, Z back
            # -> rotate around X axis, invert Y and Z axes
            T_world_cam[:3, 1] *= -1
            T_world_cam[:3, 2] *= -1

            new_pcd.transform(T_world_cam)

            # update point cloud data
            pcd.points = new_pcd.points
            pcd.colors = new_pcd.colors
            vis.update_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()

            cv2.imshow("End Effector Camera (RGB-D)", np.hstack((colour_vis, depth_vis)))
            cv2.waitKey(1)

    vis.destroy_window()


if __name__ == '__main__':
    main()
