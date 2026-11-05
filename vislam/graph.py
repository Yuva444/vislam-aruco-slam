"""Pose-graph / landmark SLAM back-end (vectorised, sparse, robust Huber loss) using SciPy.

Variables : keyframe poses X_k in SE(3) and marker poses L_i in SE(3)  (rotvec + translation)
Factors   : prior on X_0 (anchors the gauge) | VIO relative-pose factor between keyframes
            | marker factor  z_ki = T_CM  predicted as  (X_k T_BC)^-1 L_i
Marker re-observation after a gap is what closes loops: the same L_i ties distant keyframes.
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation as Rot
from .geometry import make_T, inv_T


def _vec(T):
    return Rot.from_matrix(T[:, :3, :3]).as_rotvec(), T[:, :3, 3]


def optimise_graph(T_odo, meas, T_BC, T0, verbose=False, huber=2.0, max_nfev=60, odo_k=(0.15, 0.3)):
    K = len(T_odo)
    # --- landmark initialisation: L_i = X_k (kf->frame offset) T_BC T_CM at first sighting
    first = {}
    for k, i, T, Tr in meas:
        first.setdefault(i, T_odo[k] @ Tr @ T_BC @ T)
    ids = sorted(first)
    M = len(ids)
    mid = {i: j for j, i in enumerate(ids)}
    kidx = np.array([m[0] for m in meas], int)
    midx = np.array([mid[m[1]] for m in meas], int)
    Tz = np.array([m[2] for m in meas]).reshape(-1, 4, 4)
    # camera pose in the keyframe's body frame: VIO offset (kf -> observation frame) composed with T_BC
    Tkc = np.array([m[3] @ T_BC for m in meas]).reshape(-1, 4, 4)
    Rkc, tkc = Rot.from_matrix(Tkc[:, :3, :3]) if len(Tkc) else None, Tkc[:, :3, 3]
    z_rot, z_t = Rot.from_matrix(Tz[:, :3, :3]) if len(Tz) else None, Tz[:, :3, 3]
    zd = np.linalg.norm(z_t, axis=1) if len(Tz) else np.zeros(0)
    sp_obs = 0.02 + 0.03 * zd
    sr_obs = np.radians(1.5 + 1.0 * zd)

    # --- odometry measurements (relative pose from the VIO front-end)
    rel = np.array([inv_T(T_odo[k]) @ T_odo[k + 1] for k in range(K - 1)]).reshape(-1, 4, 4)
    rel_rot, rel_t = Rot.from_matrix(rel[:, :3, :3]), rel[:, :3, 3]
    sp_odo = 0.005 + odo_k[0] * np.linalg.norm(rel_t, axis=1)
    sr_odo = 0.005 + odo_k[1] * np.linalg.norm(rel_rot.as_rotvec(), axis=1)
    R0, p0 = Rot.from_matrix(T0[:3, :3]), T0[:3, 3]

    rk, tk = _vec(T_odo)
    rm, tm = _vec(np.array([first[i] for i in ids]).reshape(-1, 4, 4))
    x0 = np.r_[rk.ravel(), tk.ravel(), rm.ravel(), tm.ravel()]

    def unpack(x):
        a = 3 * K
        return (x[:a].reshape(K, 3), x[a:2 * a].reshape(K, 3),
                x[2 * a:2 * a + 3 * M].reshape(M, 3), x[2 * a + 3 * M:].reshape(M, 3))

    def resid(x):
        rk, tk, rm, tm = unpack(x)
        Rk = Rot.from_rotvec(rk)
        out = [(tk[0] - p0) / 1e-3, (R0.inv() * Rk[0]).as_rotvec() / 1e-3]
        if K > 1:
            Ra = Rk[:-1]
            out.append(((Ra.inv().apply(tk[1:] - tk[:-1]) - rel_t) / sp_odo[:, None]).ravel())
            out.append(((rel_rot.inv() * (Ra.inv() * Rk[1:])).as_rotvec() / sr_odo[:, None]).ravel())
        if len(kidx):
            Rc = Rk[kidx] * Rkc
            tc = tk[kidx] + Rk[kidx].apply(tkc)
            Rcm = Rc.inv() * Rot.from_rotvec(rm[midx])
            tcm = Rc.inv().apply(tm[midx] - tc)
            out.append(((tcm - z_t) / sp_obs[:, None]).ravel())
            out.append(((z_rot.inv() * Rcm).as_rotvec() / sr_obs[:, None]).ravel())
        return np.concatenate(out)

    # --- Jacobian sparsity pattern
    nO, nZ = max(K - 1, 0), len(kidx)
    n_res = 6 + 6 * nO + 6 * nZ
    S = lil_matrix((n_res, len(x0)), dtype=int)
    cR = lambda k: [3 * k + j for j in range(3)]
    cT = lambda k: [3 * K + 3 * k + j for j in range(3)]
    mR = lambda m: [6 * K + 3 * m + j for j in range(3)]
    mT = lambda m: [6 * K + 3 * M + 3 * m + j for j in range(3)]
    S[0:3, cT(0)] = 1; S[3:6, cR(0)] = 1
    r = 6
    # layout of residual blocks matches the order of `out` above: [odo_t(3*nO)] [odo_r(3*nO)] [z_t(3*nZ)] [z_r(3*nZ)]
    base_ot, base_or = 6, 6 + 3 * nO
    base_zt, base_zr = 6 + 6 * nO, 6 + 6 * nO + 3 * nZ
    for k in range(nO):
        cols = cR(k) + cT(k) + cR(k + 1) + cT(k + 1)
        S[base_ot + 3 * k:base_ot + 3 * k + 3, cols] = 1
        S[base_or + 3 * k:base_or + 3 * k + 3, cR(k) + cR(k + 1)] = 1
    for n in range(nZ):
        cols = cR(kidx[n]) + cT(kidx[n]) + mR(midx[n]) + mT(midx[n])
        S[base_zt + 3 * n:base_zt + 3 * n + 3, cols] = 1
        S[base_zr + 3 * n:base_zr + 3 * n + 3, cR(kidx[n]) + mR(midx[n])] = 1

    r0 = resid(x0)
    sol = least_squares(resid, x0, jac_sparsity=S.tocsr(), loss="huber", f_scale=huber, x_scale="jac",
                        max_nfev=max_nfev, verbose=2 if verbose else 0)
    rk, tk, rm, tm = unpack(sol.x)
    T_kf = np.array([make_T(Rot.from_rotvec(a).as_matrix(), b) for a, b in zip(rk, tk)])
    mk = {i: make_T(Rot.from_rotvec(rm[mid[i]]).as_matrix(), tm[mid[i]]) for i in ids}
    info = {"cost_initial": float(0.5 * r0 @ r0), "cost_final": float(sol.cost * 2) / 2, "nfev": int(sol.nfev),
            "n_markers": M, "n_marker_obs": nZ, "status": int(sol.status)}
    return T_kf, mk, info
