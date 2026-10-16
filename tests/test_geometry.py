import math
import random

import numpy as np
import pytest

from beahiv import INVALID_CELL_ID, Measure, Orientation, cell_polygon, cell_polygons, centroid, centroids, encode
from beahiv.coords import ORIGIN_X, ORIGIN_Y, cartesian_to_axial

from ._geom_helpers import hex_area, point_in_polygon_with_tolerance, polygon_area
from ._sizes import random_size

RELATIVE_TOLERANCE = 1e-9


def _vertices(polygon) -> list[tuple[float, float]]:
    """Unwrap a Shapely `Polygon` back to a plain vertex list for the Shapely-free helpers,
    dropping the closing vertex Shapely repeats to close the ring."""
    return list(polygon.exterior.coords)[:-1]


def test_area_is_constant_and_matches_formula():
    rng = random.Random(0)
    for orientation in Orientation:
        side_length = random_size(rng)
        expected = hex_area(side_length)
        for _ in range(200):
            q = rng.randint(-10_000, 10_000)
            r = rng.randint(-10_000, 10_000)
            cell_id = encode(q, r, size=side_length, measure=Measure.EDGE_LENGTH, orientation=orientation)
            area = polygon_area(_vertices(cell_polygon(cell_id)))
            assert area == pytest.approx(expected, rel=RELATIVE_TOLERANCE)


def test_coordinate_stability_point_falls_within_returned_polygon():
    rng = random.Random(42)
    for _ in range(500):
        x = rng.uniform(0, 700_000)
        y = rng.uniform(0, 1_300_000)
        side_length = random_size(rng, upto=5000)
        orientation = rng.choice(list(Orientation))

        q, r = cartesian_to_axial(x, y, side_length, orientation)
        cell_id = encode(q, r, size=side_length, measure=Measure.EDGE_LENGTH, orientation=orientation)

        polygon = cell_polygon(cell_id)
        centre = centroid(cell_id).coords[0]

        assert point_in_polygon_with_tolerance((x, y), _vertices(polygon), centre)


def test_cell_polygons_matches_scalar_cell_polygon():
    rng = random.Random(7)
    for orientation in Orientation:
        side_length = random_size(rng)
        cell_ids = [
            encode(
                rng.randint(-10_000, 10_000),
                rng.randint(-10_000, 10_000),
                size=side_length,
                measure=Measure.EDGE_LENGTH,
                orientation=orientation,
            )
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
    cell_ids = [encode(q, -3, size=250, measure=Measure.EDGE_LENGTH) for q in range(5)]

    assert cell_polygons(np.array(cell_ids, dtype=np.uint64)) == cell_polygons(cell_ids)


def test_cell_polygons_gives_none_for_missing_and_invalid_ids():
    """Bad rows become None in place, so the result still lines up with the input row for row."""
    valid = [encode(q, -3, size=250, measure=Measure.EDGE_LENGTH) for q in range(3)]
    cell_ids = [valid[0], INVALID_CELL_ID, valid[1], None, valid[2] | (1 << 62)]

    polygons = cell_polygons(cell_ids)

    assert len(polygons) == len(cell_ids)
    assert [p is None for p in polygons] == [False, True, False, True, True]
    assert polygons[0] == cell_polygon(valid[0])
    assert polygons[2] == cell_polygon(valid[1])
    assert cell_polygons([INVALID_CELL_ID, None]) == [None, None]


def test_centroids_gives_none_for_missing_and_invalid_ids():
    valid = encode(4, -6, size=100, measure=Measure.EDGE_LENGTH)

    for lonlat in (False, True):
        points = centroids([None, valid, INVALID_CELL_ID], lonlat=lonlat)
        assert points[0] is None
        assert points[2] is None
        assert points[1] == centroid(valid, lonlat=lonlat)


def test_cell_polygons_rejects_mixed_orientation():
    cell_ids = [
        encode(0, 0, size=500, measure=Measure.EDGE_LENGTH, orientation=Orientation.POINTY),
        encode(0, 0, size=500, measure=Measure.EDGE_LENGTH, orientation=Orientation.FLAT),
    ]
    with pytest.raises(ValueError, match="single orientation"):
        cell_polygons(cell_ids)


def test_cell_polygons_rejects_mixed_side_length():
    cell_ids = [
        encode(0, 0, size=500, measure=Measure.EDGE_LENGTH, orientation=Orientation.FLAT),
        encode(0, 0, size=250, measure=Measure.EDGE_LENGTH, orientation=Orientation.FLAT),
    ]
    with pytest.raises(ValueError, match="single size"):
        cell_polygons(cell_ids)


def test_edge_to_edge_polygon_has_that_distance_between_parallel_sides():
    """Measured from the outside: opposite edge midpoints of the returned hexagon are `size`
    apart, and the area is the edge-to-edge formula sqrt(3)/2 * d**2, not the side-length one."""
    rng = random.Random(3)
    for orientation in Orientation:
        for _ in range(100):
            d = random_size(rng, Measure.EDGE_TO_EDGE)
            cell_id = encode(
                rng.randint(-10_000, 10_000),
                rng.randint(-10_000, 10_000),
                orientation=orientation,
                size=d,
                measure=Measure.EDGE_TO_EDGE,
            )
            vertices = _vertices(cell_polygon(cell_id))
            midpoints = [
                ((x1 + x2) / 2, (y1 + y2) / 2)
                for (x1, y1), (x2, y2) in zip(vertices, vertices[1:] + vertices[:1], strict=True)
            ]
            for i in range(3):
                assert math.dist(midpoints[i], midpoints[i + 3]) == pytest.approx(d, rel=1e-9)
            assert polygon_area(vertices) == pytest.approx(math.sqrt(3) / 2 * d**2, rel=RELATIVE_TOLERANCE)


def test_edge_to_edge_cell_polygons_matches_scalar():
    cell_ids = [
        encode(q, -q, orientation=Orientation.POINTY, size=300, measure=Measure.EDGE_TO_EDGE) for q in range(-50, 50)
    ]
    for batch, scalar in zip(cell_polygons(cell_ids), (cell_polygon(c) for c in cell_ids), strict=True):
        assert batch is not None and batch.equals_exact(scalar, tolerance=0)


def test_cell_polygons_rejects_mixed_measure():
    cell_ids = [
        encode(0, 0, size=500, measure=Measure.EDGE_LENGTH),
        encode(0, 0, size=500, measure=Measure.EDGE_TO_EDGE),
    ]
    with pytest.raises(ValueError, match="single size measure"):
        cell_polygons(cell_ids)


def test_origin_cell_centroid_is_the_grid_origin():
    for orientation in Orientation:
        for cell_id in (
            encode(0, 0, size=250, measure=Measure.EDGE_LENGTH, orientation=orientation),
            encode(0, 0, orientation=orientation, size=250, measure=Measure.EDGE_TO_EDGE),
        ):
            assert centroid(cell_id).coords[0] == (ORIGIN_X, ORIGIN_Y)
