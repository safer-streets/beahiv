import math
import random

import numpy as np
import pytest

from beahiv import Orientation, SizeMeasure
from beahiv.batch import axial_to_cartesian_batch, cartesian_to_axial_batch
from beahiv.coords import ORIGIN_X, ORIGIN_Y, axial_to_cartesian, cartesian_to_axial, side_length_of


def test_axial_origin_centres_on_the_grid_origin():
    """Every grid -- any side_length, either orientation -- puts axial (0, 0) on the same point,
    scalar and batch alike, and that point is the one the constants name."""
    for orientation in Orientation:
        for s in (1, 202, 100_000 / math.sqrt(3)):
            assert axial_to_cartesian(0, 0, s, orientation) == (ORIGIN_X, ORIGIN_Y)
            x, y = axial_to_cartesian_batch(np.array([0]), np.array([0]), s, orientation)
            assert (x[0], y[0]) == (ORIGIN_X, ORIGIN_Y)
            assert cartesian_to_axial(ORIGIN_X, ORIGIN_Y, s, orientation) == (0, 0)


def test_cartesian_to_axial_inverts_axial_to_cartesian():
    """The origin is added on the way out and must be subtracted on the way in, in both paths."""
    rng = random.Random(0)
    for orientation in Orientation:
        for _ in range(200):
            q, r = rng.randint(-5000, 5000), rng.randint(-5000, 5000)
            s = rng.choice([1, 50, 202, 1000 / math.sqrt(3)])
            x, y = axial_to_cartesian(q, r, s, orientation)
            assert cartesian_to_axial(x, y, s, orientation) == (q, r)
            bq, br = cartesian_to_axial_batch(np.array([x]), np.array([y]), s, orientation)
            assert (int(bq[0]), int(br[0])) == (q, r)


def test_side_length_of():
    assert side_length_of(200, SizeMeasure.SIDE_LENGTH) == 200
    # a regular hexagon's parallel sides are sqrt(3) side lengths apart
    assert side_length_of(200, SizeMeasure.SIDE_TO_SIDE) * math.sqrt(3) == pytest.approx(200, rel=1e-15)
