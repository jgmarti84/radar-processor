"""
Cache implementation for 2D and 3D radar grids using LRU cache.
"""
from cachetools import LRUCache
import numpy as np


def _nbytes_arr(a) -> int:
    """Calculate byte size of an array or MaskedArray."""
    if isinstance(a, np.ma.MaskedArray):
        base = a.data.nbytes
        m = np.ma.getmaskarray(a)
        return base + (m.nbytes if m is not np.ma.nomask else 0)
    return getattr(a, "nbytes", 0)


def _nbytes_pkg(pkg) -> int:
    """Calculate byte size of a 2D grid cache package."""
    n = 0
    a = pkg.get("arr")
    if a is not None:
        n += _nbytes_arr(a)
    
    # Also count QC fields if present
    qc_dict = pkg.get("qc", {})
    for qc_arr in qc_dict.values():
        if qc_arr is not None:
            n += _nbytes_arr(qc_arr)
    
    # Warped arrays
    arr_warped = pkg.get("arr_warped")
    if arr_warped is not None:
        n += _nbytes_arr(arr_warped)
    
    return n


def _nbytes_pkg3d(pkg) -> int:
    """Calculate byte size of a 3D grid cache package."""
    n = 0
    a3 = pkg.get("arr3d")
    if a3 is not None:
        n += _nbytes_arr(a3)
    return n


# 2D Grid Cache (200 MB limit)
GRID2D_CACHE = LRUCache(maxsize=200 * 1024 * 1024, getsizeof=_nbytes_pkg)

# 3D Grid Cache (600 MB limit) 
GRID3D_CACHE = LRUCache(maxsize=600 * 1024 * 1024, getsizeof=_nbytes_pkg3d)
