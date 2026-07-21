import numpy as np

from z_manip.ik.symmetry import expand_symmetry


def test_parallel_gripper_symmetry_expands_valid_se3_family():
    grasp = np.eye(4)
    grasp[:3, 3] = (0.45, -0.08, 0.18)

    family = expand_symmetry(grasp, n_about_axis=8)

    assert family.shape == (8, 4, 4)
    assert np.allclose(family[:, :3, 3], grasp[:3, 3])
    assert np.allclose(np.linalg.det(family[:, :3, :3]), 1.0)
    assert np.allclose(family[:, :3, 2], grasp[:3, 2])
    assert np.allclose(family[4, :3, 0], -grasp[:3, 0], atol=1e-7)
