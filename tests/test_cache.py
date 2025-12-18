"""
Unit tests for radar_cog_processor.cache module.
"""
import pytest
import numpy as np
from radar_cog_processor.cache import (
    _nbytes_arr,
    _nbytes_pkg,
    _nbytes_pkg3d,
    GRID2D_CACHE,
    GRID3D_CACHE,
)


def test_nbytes_arr_regular():
    """Test byte size calculation for regular arrays."""
    arr = np.ones((100, 100), dtype=np.float32)
    size = _nbytes_arr(arr)
    expected = 100 * 100 * 4  # float32 = 4 bytes
    assert size == expected


def test_nbytes_arr_masked():
    """Test byte size calculation for masked arrays."""
    data = np.ones((100, 100), dtype=np.float32)
    mask = np.zeros((100, 100), dtype=bool)
    arr = np.ma.array(data, mask=mask)
    
    size = _nbytes_arr(arr)
    # Should include both data and mask
    expected = (100 * 100 * 4) + (100 * 100 * 1)  # float32 + bool
    assert size == expected


def test_nbytes_pkg():
    """Test byte size calculation for 2D grid package."""
    arr = np.ma.array(np.ones((100, 100), dtype=np.float32))
    qc_arr = np.ma.array(np.ones((100, 100), dtype=np.float32))
    
    pkg = {
        "arr": arr,
        "qc": {"RHOHV": qc_arr},
        "crs": "EPSG:4326",
        "transform": None,
    }
    
    size = _nbytes_pkg(pkg)
    # Should count main array + QC arrays
    assert size > 0
    assert size >= arr.nbytes


def test_nbytes_pkg3d():
    """Test byte size calculation for 3D grid package."""
    arr3d = np.ones((50, 100, 100), dtype=np.float32)
    
    pkg = {
        "arr3d": arr3d,
        "x": np.linspace(-10000, 10000, 100),
        "y": np.linspace(-10000, 10000, 100),
        "z": np.linspace(0, 10000, 50),
    }
    
    size = _nbytes_pkg3d(pkg)
    assert size == arr3d.nbytes


def test_grid2d_cache_operations():
    """Test basic 2D cache operations."""
    # Clear cache
    GRID2D_CACHE.clear()
    
    # Test set and get
    key = "test_key_2d"
    arr = np.ma.array(np.ones((100, 100), dtype=np.float32))
    pkg = {"arr": arr, "crs": "EPSG:4326"}
    
    GRID2D_CACHE[key] = pkg
    assert key in GRID2D_CACHE
    
    retrieved = GRID2D_CACHE[key]
    np.testing.assert_array_equal(retrieved["arr"], pkg["arr"])
    
    # Clear for other tests
    GRID2D_CACHE.clear()


def test_grid3d_cache_operations():
    """Test basic 3D cache operations."""
    # Clear cache
    GRID3D_CACHE.clear()
    
    # Test set and get
    key = "test_key_3d"
    arr3d = np.ones((50, 100, 100), dtype=np.float32)
    pkg = {
        "arr3d": arr3d,
        "x": np.linspace(-10000, 10000, 100),
        "y": np.linspace(-10000, 10000, 100),
        "z": np.linspace(0, 10000, 50),
    }
    
    GRID3D_CACHE[key] = pkg
    assert key in GRID3D_CACHE
    
    retrieved = GRID3D_CACHE[key]
    np.testing.assert_array_equal(retrieved["arr3d"], pkg["arr3d"])
    
    # Clear for other tests
    GRID3D_CACHE.clear()


def test_cache_eviction():
    """Test that cache respects size limits."""
    GRID2D_CACHE.clear()
    
    # Create arrays that should exceed cache limit when combined
    # Cache limit is 200 MB
    large_arr = np.ones((5000, 5000), dtype=np.float32)  # ~100 MB
    
    pkg1 = {"arr": large_arr.copy()}
    pkg2 = {"arr": large_arr.copy()}
    pkg3 = {"arr": large_arr.copy()}
    
    GRID2D_CACHE["key1"] = pkg1
    GRID2D_CACHE["key2"] = pkg2
    GRID2D_CACHE["key3"] = pkg3  # Should trigger eviction
    
    # At least one key should have been evicted
    assert len(GRID2D_CACHE) < 3
    
    # Clear for other tests
    GRID2D_CACHE.clear()
