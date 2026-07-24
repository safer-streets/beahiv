"""Axial <-> Cartesian conversions for a regular hex lattice.

Formulas follow the standard axial-hex reference (Red Blob Games):
POINTY cells have vertical side edges, FLAT cells are the 30-degree
rotation of that. Both share the same cube-coordinate rounding scheme.

FLAT's basis vectors are POINTY's basis vectors rotated +30 degrees
about the origin -- not a relabelling of the same grid. So (q, r) is
only meaningful relative to its own orientation's basis: the same
(q, r, side_length) centres on a *different* point for each orientation,
and the two only agree at the shared origin (0, 0). Don't compare or
reuse q/r (or cell ids) across orientations expecting the same physical
cell.
"""

import math

from .orientation import Orientation

SQRT3 = math.sqrt(3.0)


def axial_to_cartesian(
    q: float,
    r: float,
    side_length: float,
    orientation: Orientation = Orientation.FLAT,
) -> tuple[float, float]:
    """Return the (x, y) centre of hex (q, r) in the same units as side_length."""
    s = side_length
    if orientation == Orientation.POINTY:
        x = s * SQRT3 * (q + r / 2.0)
        y = 1.5 * s * r
    else:
        x = 1.5 * s * q
        y = s * SQRT3 * (r + q / 2.0)
    return x, y


def cartesian_to_axial(
    x: float,
    y: float,
    side_length: float,
    orientation: Orientation = Orientation.FLAT,
) -> tuple[int, int]:
    """Return the integer (q, r) of the hex containing point (x, y)."""
    s = side_length
    if orientation == Orientation.POINTY:
        qf = (SQRT3 / 3.0 * x - y / 3.0) / s
        rf = (2.0 / 3.0 * y) / s
    else:
        qf = (2.0 / 3.0 * x) / s
        rf = (-1.0 / 3.0 * x + SQRT3 / 3.0 * y) / s

    return _round_cube(qf, rf)


def _round_half_up(value: float) -> int:
    """Round half away from zero-bias (round half towards +inf), not banker's rounding."""
    return math.floor(value + 0.5)


def _round_cube(qf: float, rf: float) -> tuple[int, int]:
    cx, cz = qf, rf
    cy = -cx - cz

    rx = _round_half_up(cx)
    ry = _round_half_up(cy)
    rz = _round_half_up(cz)

    x_diff = abs(rx - cx)
    y_diff = abs(ry - cy)
    z_diff = abs(rz - cz)

    if x_diff > y_diff and x_diff > z_diff:
        rx = -ry - rz
    elif y_diff > z_diff:
        ry = -rx - rz
    else:
        rz = -rx - ry

    return int(rx), int(rz)
