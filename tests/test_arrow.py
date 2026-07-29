"""pyarrow interop for `latlon_to_cell` / `bng_to_cell`.

Arrow arrays have always worked as *input* (numpy consumes them via the buffer protocol); what is
tested here is that they now come back as Arrow, and that Arrow nulls behave like NaN rather than
blowing up the encode. pyarrow is an extra (`beahiv[arrow]`), in the dev group so these run.
"""

import numpy as np
import pyarrow as pa
import pytest

from beahiv import Orientation, bng_to_cell, latlon_to_cell
from beahiv.cell_id import INVALID_CELL_ID, ORIENTATION_SHIFT

SIDE = 202

# a couple of London/Leeds points, in both CRSs
_LATLON = ([51.5074, 53.8008], [-0.1278, -1.5491])
_BNG = ([530034.0, 429600.0], [180381.0, 434000.0])


@pytest.mark.parametrize(
    ("fn", "a", "b"),
    [(latlon_to_cell, *_LATLON), (bng_to_cell, *_BNG)],
)
def test_arrow_in_arrow_out(fn, a, b):
    """An Arrow array in gives an Arrow array back, with the same ids the numpy path produces."""
    result = fn(pa.array(a), pa.array(b), SIDE, Orientation.FLAT)

    assert isinstance(result, pa.Array)
    assert result.to_pylist() == fn(np.array(a), np.array(b), SIDE, Orientation.FLAT).tolist()


@pytest.mark.parametrize(
    ("fn", "a", "b"),
    [(latlon_to_cell, *_LATLON), (bng_to_cell, *_BNG)],
)
def test_chunked_array_in_arrow_out(fn, a, b):
    """A ChunkedArray works too, and the chunk boundaries make no difference to the result."""
    chunked_a = pa.chunked_array([[a[0]], [a[1]]])
    chunked_b = pa.chunked_array([[b[0]], [b[1]]])

    result = fn(chunked_a, chunked_b, SIDE, Orientation.FLAT)

    assert isinstance(result, pa.Array)
    assert result.to_pylist() == fn(pa.array(a), pa.array(b), SIDE, Orientation.FLAT).to_pylist()


@pytest.mark.parametrize(
    ("fn", "a", "b"),
    [(latlon_to_cell, *_LATLON), (bng_to_cell, *_BNG)],
)
def test_arrow_nulls_become_invalid_cell_id(fn, a, b):
    """A null coordinate arrives as NaN and maps to INVALID_CELL_ID, leaving its neighbours alone."""
    result = fn(pa.array([a[0], None]), pa.array([b[0], None]), SIDE, Orientation.FLAT)

    assert result.to_pylist() == [fn(a[0], b[0], SIDE, Orientation.FLAT), INVALID_CELL_ID]


@pytest.mark.parametrize(
    ("fn", "a", "b"),
    [(latlon_to_cell, *_LATLON), (bng_to_cell, *_BNG)],
)
def test_numpy_and_scalar_returns_are_unchanged(fn, a, b):
    """Only Arrow input gets Arrow back: the numpy and scalar paths keep their existing types."""
    assert isinstance(fn(np.array(a), np.array(b), SIDE, Orientation.FLAT), np.ndarray)
    assert isinstance(fn(list(a), list(b), SIDE, Orientation.FLAT), np.ndarray)
    assert isinstance(fn(a[0], b[0], SIDE, Orientation.FLAT), int)


def test_arrow_return_dtype_is_uint64():
    """Arrow gets uint64 back, and every id fits a signed 64-bit column too.

    The reserved bits sit above the orientation bit, so no id reaches bit 63 and
    a consumer storing these as int64/BIGINT can't misread one as negative.
    """
    result = bng_to_cell(pa.array(_BNG[0]), pa.array(_BNG[1]), SIDE, Orientation.FLAT)

    assert result.type == pa.uint64()
    assert all(cell >> ORIENTATION_SHIFT & 1 for cell in result.to_pylist())  # FLAT sets it
    assert all(cell < 2**63 for cell in result.to_pylist())
