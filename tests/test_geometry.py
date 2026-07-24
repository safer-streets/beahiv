import random

import pytest

from beahiv import Orientation, cell_centre, cell_polygon, encode
from beahiv.coords import cartesian_to_axial

from ._geom_helpers import hex_area, point_in_polygon_with_tolerance, polygon_area

RELATIVE_TOLERANCE = 1e-9


def test_area_is_constant_and_matches_formula():
    rng = random.Random(0)
    for orientation in Orientation:
        side_length = rng.randint(1, 100_000)
        expected = hex_area(side_length)
        for _ in range(200):
            q = rng.randint(-10_000, 10_000)
            r = rng.randint(-10_000, 10_000)
            cell_id = encode(q, r, side_length, orientation)
            area = polygon_area(cell_polygon(cell_id))
            assert area == pytest.approx(expected, rel=RELATIVE_TOLERANCE)


def test_coordinate_stability_point_falls_within_returned_polygon():
    rng = random.Random(42)
    for _ in range(500):
        x = rng.uniform(0, 700_000)
        y = rng.uniform(0, 1_300_000)
        side_length = rng.randint(10, 5000)
        orientation = rng.choice(list(Orientation))

        q, r = cartesian_to_axial(x, y, side_length, orientation)
        cell_id = encode(q, r, side_length, orientation)

        polygon = cell_polygon(cell_id)
        centre = cell_centre(cell_id)

        assert point_in_polygon_with_tolerance((x, y), polygon, centre)
