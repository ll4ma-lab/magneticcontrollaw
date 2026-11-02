"""Force field of one magnet command over a grid of object positions."""
import warnings

import numpy as np
import torch

from models import model_call_rotated_wrench

PLANES = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}


def grid_points(plane, offset, half_width, resolution):
    """(resolution², 3) points on an axis-aligned square slice at `offset` along its normal."""
    u = np.linspace(-half_width, half_width, resolution)
    a, b = np.meshgrid(u, u, indexing="ij")
    i, j = PLANES[plane]
    points = np.zeros((a.size, 3))
    points[:, 3 - i - j] = offset
    points[:, i] = a.ravel()
    points[:, j] = b.ravel()
    return points


def force_field(model, control, points, magnet_size=0.0):
    """World wrenches [F; T], shape (N, 6), that `control` applies with the object at each point.

    Rows are NaN inside a magnet's box and where Griffin's model is undefined (the rotation
    axis along the magnet-object line).
    """
    points = np.asarray(points, dtype=float).reshape((-1, 3))
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")  # torch.cross's default-dim deprecation, divide by zero
        if model.use_sphere:
            wrenches = _sphere_field(model, control, points)
        else:
            wrenches = np.array(
                [model.forward_cylinder(p, control).ravel() for p in points]
            ).reshape((-1, 6))
    magnets = np.asarray(model.magnets, dtype=float)
    offsets = np.abs(points[:, None, :] - magnets[None, :, :])
    inside = np.any(np.all(offsets < magnet_size / 2, axis=2), axis=1)
    wrenches[inside] = np.nan
    wrenches[~np.all(np.isfinite(wrenches), axis=1)] = np.nan
    return wrenches


def _sphere_field(model, control, points):
    """MultiMagModel.forward_sphere for many points at once, on the torch path that
    RadiusSysid.forward_acceleration uses."""
    n = len(points)
    if n == 0:
        return np.zeros((0, 6))
    # model_call_rotated_wrench special-cases a batch of 1, and torch.cross picks the wrong
    # dimension for a batch of 3, so pad the batch past both.
    padded = points
    while len(padded) in (1, 3):
        padded = np.vstack([padded, padded[-1:]])
    m = len(padded)
    chosen_magnet_pose = np.asarray(model.magnets[control.mag_num], dtype=float) - padded
    dists = torch.tensor(np.linalg.norm(chosen_magnet_pose, axis=1), dtype=torch.float64)
    model_vals = [
        torch.as_tensor(v, dtype=torch.float64).reshape(-1)
        for v in model.sphere_model.model_call(dists, 1)
    ]
    angles = np.tile(np.array([[control.psi], [control.xi]], dtype=float), (1, m))
    wrenches = model_call_rotated_wrench(
        torch.full((m,), float(control.dipole_strength), dtype=torch.float64),
        torch.tensor(angles, dtype=torch.float64),
        model_vals,
        torch.tensor(-chosen_magnet_pose.T, dtype=torch.float64),
        library=torch,
        zero=torch.zeros(m, dtype=torch.float64),
    )
    return wrenches.reshape((m, 6)).numpy()[:n]
