#!/usr/bin/env python3
import mujoco
import mujoco.viewer
import numpy as np

def SO3_to_so3(R):
    theta = np.arccos((np.trace(R) - 1) / 2)
    if np.isclose(theta, 0):
        return np.array([0, 0, 0])
    v = np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1],
    ])
    vn = np.linalg.norm(v)
    if np.isclose(vn, 0):
        return np.array([0, 0, 0])
    v /= vn
    return v


def main(args=None):
    m = mujoco.MjModel.from_xml_path("franka_emika_panda/mjx_single_cube.xml")
    d = mujoco.MjData(m)

    # site ID of the end-effector
    gripper_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "gripper")

    box_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "box_frame")

    # actuated degrees of freedom, excluding the parallel gripper joint
    dof_indices = m.actuator_trnid[:, 0][:-1]

    # set initial ready state with gripper fully opened
    ready_state = np.array([0., -1/4 * np.pi, 0., -3/4 * np.pi, 0., 1/2 * np.pi, 1/4 * np.pi])
    m.qpos0[:7] = -ready_state
    m.qpos0[7] = -0.08


    with mujoco.viewer.launch_passive(m, d) as viewer:
        while viewer.is_running():

            # get current box pose
            box_pos = d.site_xpos[box_id].copy()
            box_rot = d.site_xmat[box_id].reshape(3, 3).copy()
            box_so3 = SO3_to_so3(box_rot)

            # get current gripper pose
            gripper_pos = d.site_xpos[gripper_id].copy()
            gripper_rot = d.site_xmat[gripper_id].reshape(3, 3).copy()
            gripper_so3 = SO3_to_so3(gripper_rot)

            # current joint state
            qk = d.qpos[dof_indices].copy()

            # full 6 x N Jacobian with rows:
            # 0, 1, 2: translational (vx, vy, vz)
            # 3, 4, 5: rotational (wx, wy, wz)
            J = np.zeros((6, m.nv))

            mujoco.mj_forward(m, d)
            mujoco.mj_jacSite(
                m, d,
                J[0:3], # translational Jacobian
                J[3:6], # rotational Jacobian
                gripper_id,
            )

            # filter by actuated degrees of freedom
            J = J[:, dof_indices]

            # TODO: update joint state to minimise distance of end-effector pose to box pose
            d.qpos[dof_indices] = qk

            mujoco.mj_step(m, d)
            viewer.sync()


if __name__ == '__main__':
    main()
