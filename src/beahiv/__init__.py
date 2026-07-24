"""BEAHIV (British Equal-Area Hexagonal, Index, that's Versatile): a fast,
equal-area hexagonal grid for EPSG:27700."""

from .cell_id import CellIndex, decode, encode
from .geo import bng_to_cell, centroid, latlon_to_cell
from .geometry import cell_centre, cell_polygon
from .morton import decode_morton, encode_morton
from .neighbours import NEIGHBOUR_OFFSETS, distance, get_neighbours, k_ring
from .orientation import Orientation
from .polyfill import polyfill

__all__ = [
    "Orientation",
    "CellIndex",
    "encode",
    "decode",
    "cell_centre",
    "cell_polygon",
    "get_neighbours",
    "NEIGHBOUR_OFFSETS",
    "distance",
    "k_ring",
    "latlon_to_cell",
    "centroid",
    "bng_to_cell",
    "encode_morton",
    "decode_morton",
    "polyfill",
]
