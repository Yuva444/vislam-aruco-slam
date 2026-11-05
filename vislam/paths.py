"""User-designed walking paths for the simulator: waypoints -> smooth curve, validation, random generation.

The curve is built exactly like the simulator builds the walk (natural cubic spline over chord length), so the
preview in the dashboard is the path the simulated person will really follow."""
import numpy as np
from scipy.interpolate import CubicSpline

ROOM = (6.0, 5.0)

PRESETS = {
    "Rectangle loop": [(1.5, 1.5), (4.5, 1.5), (4.5, 3.5), (1.5, 3.5), (1.5, 1.5)],
    "L-shape": [(1.5, 1.5), (4.5, 1.5), (4.5, 3.8)],
    "Zig-zag": [(1.0, 1.0), (2.5, 3.8), (3.5, 1.0), (5.0, 3.8)],
    "S-curve": [(1.0, 1.0), (2.0, 3.6), (3.2, 1.6), (4.2, 3.8), (5.0, 2.2)],
    "Triangle loop": [(1.5, 1.2), (4.8, 1.8), (3.0, 3.9), (1.5, 1.2)],
}


def path_length(wp):
    wp = np.asarray(wp, float)
    return float(np.linalg.norm(np.diff(wp, axis=0), axis=1).sum()) if len(wp) > 1 else 0.0


def path_curve(wp, n=300):
    """Points (n,2) of the smooth path through the waypoints."""
    wp = np.asarray(wp, float)
    u = np.r_[0, np.cumsum(np.linalg.norm(np.diff(wp, axis=0), axis=1))]
    return CubicSpline(u, wp, bc_type="natural")(np.linspace(0, u[-1], n))


def validate_path(wp, room=ROOM, margin=0.5, min_length=3.0, max_length=60.0):
    """Returns (errors, warnings): errors block the simulation, warnings are advice."""
    errors, warnings = [], []
    try:
        wp = np.asarray(wp, float)
    except (TypeError, ValueError):
        return ["Waypoints must be numbers."], []
    if wp.ndim != 2 or wp.shape[1] != 2 or len(wp) < 2:
        return ["Add at least 2 waypoints (x and y)."], []
    if not np.isfinite(wp).all():
        return ["Some waypoints are empty - fill in or delete those rows."], []
    out = (wp[:, 0] < margin) | (wp[:, 0] > room[0] - margin) | (wp[:, 1] < margin) | (wp[:, 1] > room[1] - margin)
    if out.any():
        errors.append(f"Waypoint(s) {', '.join(str(i + 1) for i in np.where(out)[0])} are too close to a wall or outside the room: "
                      f"keep x in [{margin:g}, {room[0] - margin:g}] and y in [{margin:g}, {room[1] - margin:g}] m.")
    if (np.linalg.norm(np.diff(wp, axis=0), axis=1) < 0.05).any():
        errors.append("Two consecutive waypoints are (almost) identical - remove one.")
    if errors:
        return errors, warnings
    L = path_length(wp)
    if L < min_length:
        errors.append(f"The path is only {L:.1f} m long - make it at least {min_length:g} m so there is something to estimate.")
    if L > max_length:
        warnings.append(f"A {L:.0f} m path will take a while to simulate and analyse.")
    c = path_curve(wp)
    if ((c[:, 0] < 0.25) | (c[:, 0] > room[0] - 0.25) | (c[:, 1] < 0.25) | (c[:, 1] > room[1] - 0.25)).any():
        warnings.append("The smoothed path swings very close to a wall or outside the room between two waypoints - "
                        "add a waypoint there or soften the sharp turn.")
    return errors, warnings


def _crosses(a, b, c, d):
    """True if segment ab properly intersects segment cd."""
    def o(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
    return o(a, b, c) * o(a, b, d) < 0 and o(c, d, a) * o(c, d, b) < 0


def self_intersects(wp):
    wp = np.asarray(wp, float)
    n = len(wp) - 1
    return any(_crosses(wp[i], wp[i + 1], wp[j], wp[j + 1]) for i in range(n) for j in range(i + 2, n))


def random_path(seed, n_points=5, closed=False, room=ROOM, margin=0.9, min_step=1.3, max_step=3.2, tries=2000):
    """Random valid walk inside the room.  Deterministic for a given seed.  closed=True returns to the start."""
    rng = np.random.default_rng(seed)
    for _ in range(tries):
        pts = [rng.uniform([margin, margin], [room[0] - margin, room[1] - margin])]
        for _ in range(n_points - 1):
            for _ in range(60):
                p = rng.uniform([margin, margin], [room[0] - margin, room[1] - margin])
                if min_step <= np.linalg.norm(p - pts[-1]) <= max_step and all(np.linalg.norm(p - q) > 0.8 for q in pts[:-1]):
                    pts.append(p)
                    break
            else:
                break
        if len(pts) < n_points:
            continue
        if closed:
            pts.append(pts[0].copy())
        wp = np.round(np.array(pts), 2)
        errs, warns = validate_path(wp, room, margin=0.5)
        if not errs and not warns and not self_intersects(wp):
            return wp
    raise RuntimeError("could not generate a random path; try fewer waypoints")
