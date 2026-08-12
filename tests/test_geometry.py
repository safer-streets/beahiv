import random

import numpy as np
import pytest

from beahiv import Orientation, cell_polygon, cell_polygons, centroid, encode
from beahiv.coords import cartesian_to_axial

from ._geom_helpers import hex_area, point_in_polygon_with_tolerance, polygon_area

RELATIVE_TOLERANCE = 1e-9


def _vertices(polygon) -> list[tuple[float, float]]:
    """Unwrap a Shapely `Polygon` back to a plain vertex list for the Shapely-free helpers,
    dropping the closing vertex Shapely repeats to close the ring."""
    return list(polygon.exterior.coords)[:-1]


def test_area_is_constant_and_matches_formula():
    rng = random.Random(0)
    for orientation in Orientation:
        side_length = rng.randint(1, 100_000)
        expected = hex_area(side_length)
        for _ in range(200):
            q = rng.randint(-10_000, 10_000)
            r = rng.randint(-10_000, 10_000)
            cell_id = encode(q, r, side_length, orientation)
            area = polygon_area(_vertices(cell_polygon(cell_id)))
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
        centre = centroid(cell_id)

        assert point_in_polygon_with_tolerance((x, y), _vertices(polygon), centre)


def test_cell_polygons_matches_scalar_cell_polygon():
    rng = random.Random(7)
    for orientation in Orientation:
        side_length = rng.randint(1, 100_000)
        cell_ids = [
            encode(rng.randint(-10_000, 10_000), rng.randint(-10_000, 10_000), side_length, orientation)
            for _ in range(200)
        ]

        batch_polygons = cell_polygons(cell_ids)
        scalar_polygons = [cell_polygon(cell_id) for cell_id in cell_ids]

        assert len(batch_polygons) == len(scalar_polygons)
        for batch_polygon, scalar_polygon in zip(batch_polygons, scalar_polygons, strict=True):
            assert _vertices(batch_polygon) == pytest.approx(_vertices(scalar_polygon))


def test_cell_polygons_empty_input_returns_empty_list():
    assert cell_polygons([]) == []
    assert cell_polygons(np.array([], dtype=np.uint64)) == []


def test_cell_polygons_accepts_an_array_of_ids():
    """The ids usually arrive as the uint64 array a batch call produced, not as a list."""
    cell_ids = [encode(q, -3, 250) for q in range(5)]

    assert cell_polygons(np.array(cell_ids, dtype=np.uint64)) == cell_polygons(cell_ids)


def test_cell_polygons_rejects_mixed_orientation():
    cell_ids = [encode(0, 0, 500, Orientation.POINTY), encode(0, 0, 500, Orientation.FLAT)]
    with pytest.raises(ValueError, match="single orientation"):
        cell_polygons(cell_ids)


def test_cell_polygons_rejects_mixed_side_length():
    cell_ids = [encode(0, 0, 500, Orientation.FLAT), encode(0, 0, 250, Orientation.FLAT)]
    with pytest.raises(ValueError, match="single side_length"):
        cell_polygons(cell_ids)
