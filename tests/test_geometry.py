import numpy as np
from vislam.geometry import Exp, Log, make_T, inv_T, gravity_alignment, R_BC_DEFAULT


def test_exp_log_roundtrip():
    w = np.array([0.3, -0.2, 0.5])
    assert np.allclose(Log(Exp(w)), w, atol=1e-9)


def test_inverse_transform():
    T = make_T(Exp([0.1, 0.2, 0.3]), [1, 2, 3])
    assert np.allclose(T @ inv_T(T), np.eye(4), atol=1e-12)


def test_gravity_alignment_levels_the_body():
    R = Exp([0.2, -0.1, 0.4])
    f = R.T @ np.array([0, 0, 9.81])           # static specific force in body frame
    Rw = gravity_alignment(f)
    assert np.allclose(Rw.T @ np.array([0, 0, 1.0]), f / np.linalg.norm(f), atol=1e-9)


def test_camera_extrinsic_is_a_rotation():
    assert np.isclose(np.linalg.det(R_BC_DEFAULT), 1.0) and np.allclose(R_BC_DEFAULT @ R_BC_DEFAULT.T, np.eye(3))
