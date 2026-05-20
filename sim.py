#!/usr/bin/env python3
import mujoco
import mujoco.viewer
import numpy as np


def main(args=None):
    m = mujoco.MjModel.from_xml_path("franka_emika_panda/mjx_single_cube.xml")
    d = mujoco.MjData(m)

    # ready state
    ready_state = np.array([0., -1/4 * np.pi, 0., -3/4 * np.pi, 0., 1/2 * np.pi, 1/4 * np.pi])
    m.qpos0[:7] = -ready_state

    with mujoco.viewer.launch_passive(m, d) as viewer:
        while viewer.is_running():
            mujoco.mj_step(m, d)
            viewer.sync()


if __name__ == '__main__':
    main()
