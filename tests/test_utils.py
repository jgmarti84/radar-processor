"""
Unit tests for radar_cog_processor.utils module.
"""
import pytest
import numpy as np
import hashlib
from unittest.mock import Mock, MagicMock
from radar_cog_processor.utils import (
    md5_file,
    stable_hash,
    resolve_field,
    colormap_for,
    get_radar_site,
    safe_range_max_m,
    qc_signature,
    grid2d_cache_key,
    grid3d_cache_key,
    collapse_field_3d_to_2d,
)
from radar_cog_processor.constants import FIELD_ALIASES


def test_md5_file(tmp_path):
    """Test MD5 file hashing."""
    test_file = tmp_path / "test.txt"
    test_content = b"Hello, World!"
    test_file.write_bytes(test_content)
    
    result = md5_file(str(test_file))
    expected = hashlib.md5(test_content).hexdigest()
    
    assert result == expected
    assert len(result) == 32  # MD5 hex digest length


def test_stable_hash():
    """Test stable hashing of objects."""
    obj1 = {"b": 2, "a": 1}
    obj2 = {"a": 1, "b": 2}
    
    # Same content should produce same hash
    assert stable_hash(obj1) == stable_hash(obj2)
    
    # Different content should produce different hash
    obj3 = {"a": 1, "b": 3}
    assert stable_hash(obj1) != stable_hash(obj3)


def test_resolve_field():
    """Test field name resolution."""
    # Mock radar object
    radar = Mock()
    radar.fields = {"DBZH": {}, "corrected_reflectivity_horizontal": {}}
    
    # Test with primary alias
    field_name, field_key = resolve_field(radar, "DBZH")
    assert field_name == "DBZH"
    assert field_key == "DBZH"
    
    # Test with secondary alias
    radar.fields = {"corrected_reflectivity_horizontal": {}}
    field_name, field_key = resolve_field(radar, "DBZH")
    assert field_name == "corrected_reflectivity_horizontal"
    assert field_key == "DBZH"
    
    # Test unsupported field
    with pytest.raises(KeyError):
        resolve_field(radar, "UNSUPPORTED_FIELD")


def test_colormap_for():
    """Test colormap retrieval."""
    # Test default colormap
    cmap, vmin, vmax, cmap_key = colormap_for("DBZH")
    assert vmin == -30.0
    assert vmax == 70.0
    assert cmap_key == "grc_th"
    
    # Test with override
    cmap, vmin, vmax, cmap_key = colormap_for("DBZH", override_cmap="grc_th2")
    assert cmap_key == "grc_th2"
    
    # Test different field
    cmap, vmin, vmax, cmap_key = colormap_for("ZDR")
    assert vmin == -5.0
    assert vmax == 10.5


def test_get_radar_site():
    """Test radar site coordinate extraction."""
    radar = Mock()
    radar.latitude = {"data": np.array([[-34.5]])}
    radar.longitude = {"data": np.array([[-58.5]])}
    radar.altitude = {"data": np.array([[100.0]])}
    
    lon, lat, alt = get_radar_site(radar)
    assert lat == -34.5
    assert lon == -58.5
    assert alt == 100.0


def test_safe_range_max_m():
    """Test safe range maximum extraction."""
    radar = Mock()
    
    # Test with valid data
    radar.range = {"data": np.array([0, 1000, 2000, 3000])}
    assert safe_range_max_m(radar) == 3000.0
    
    # Test with masked data
    radar.range = {"data": np.ma.array([0, 1000, 2000, np.nan], mask=[False, False, False, True])}
    assert safe_range_max_m(radar) == 2000.0
    
    # Test with empty data
    radar.range = {"data": np.array([])}
    assert safe_range_max_m(radar) == 240000.0  # default


def test_qc_signature():
    """Test QC signature generation for cache keys."""
    # Mock filter objects
    filter1 = Mock()
    filter1.field = "RHOHV"
    filter1.min = 0.8
    filter1.max = 1.0
    
    filter2 = Mock()
    filter2.field = "DBZH"
    filter2.min = -10.0
    filter2.max = None
    
    filters = [filter1, filter2]
    sig = qc_signature(filters)
    
    # Only RHOHV should be in signature (it's in AFFECTS_INTERP_FIELDS)
    assert len(sig) == 1
    assert sig[0] == ("RHOHV", 0.8, 1.0)


def test_grid2d_cache_key():
    """Test 2D grid cache key generation."""
    key1 = grid2d_cache_key(
        file_hash="abc123",
        product_upper="PPI",
        field_to_use="DBZH",
        elevation=0,
        cappi_height=None,
        volume="01",
        interp="nearest",
        qc_sig=()
    )
    
    key2 = grid2d_cache_key(
        file_hash="abc123",
        product_upper="PPI",
        field_to_use="DBZH",
        elevation=0,
        cappi_height=None,
        volume="01",
        interp="nearest",
        qc_sig=()
    )
    
    # Same parameters should produce same key
    assert key1 == key2
    assert key1.startswith("g2d_")
    
    # Different elevation should produce different key
    key3 = grid2d_cache_key(
        file_hash="abc123",
        product_upper="PPI",
        field_to_use="DBZH",
        elevation=1,
        cappi_height=None,
        volume="01",
        interp="nearest",
        qc_sig=()
    )
    assert key1 != key3


def test_grid3d_cache_key():
    """Test 3D grid cache key generation."""
    key1 = grid3d_cache_key(
        file_hash="abc123",
        field_to_use="DBZH",
        volume="01",
        qc_sig=(),
        grid_res_xy=1200.0,
        grid_res_z=1200.0,
        z_top_m=15000.0
    )
    
    assert key1.startswith("g3d_")
    
    # Different resolution should produce different key
    key2 = grid3d_cache_key(
        file_hash="abc123",
        field_to_use="DBZH",
        volume="01",
        qc_sig=(),
        grid_res_xy=300.0,
        grid_res_z=300.0,
        z_top_m=15000.0
    )
    assert key1 != key2


def test_collapse_field_3d_to_2d_colmax():
    """Test 3D to 2D collapse for COLMAX product."""
    # Create synthetic 3D data
    data3d = np.random.rand(10, 20, 30)  # (z, y, x)
    
    result = collapse_field_3d_to_2d(data3d, "colmax")
    
    assert result.shape == (20, 30)
    # COLMAX should be max along z axis
    expected = data3d.max(axis=0)
    np.testing.assert_array_almost_equal(result.data, expected)


def test_collapse_field_3d_to_2d_cappi():
    """Test 3D to 2D collapse for CAPPI product."""
    data3d = np.random.rand(10, 20, 30)
    z_levels = np.linspace(0, 10000, 10)
    target_height = 5000.0
    
    result = collapse_field_3d_to_2d(
        data3d,
        "cappi",
        z_levels=z_levels,
        target_height_m=target_height
    )
    
    assert result.shape == (20, 30)
    # Should select level closest to target height
    iz = np.abs(z_levels - target_height).argmin()
    np.testing.assert_array_almost_equal(result.data, data3d[iz, :, :])


def test_collapse_field_3d_to_2d_ppi():
    """Test 3D to 2D collapse for PPI product."""
    data3d = np.random.rand(10, 20, 30)
    x_coords = np.linspace(-10000, 10000, 30)
    y_coords = np.linspace(-10000, 10000, 20)
    z_levels = np.linspace(0, 10000, 10)
    elevation_deg = 0.5
    
    result = collapse_field_3d_to_2d(
        data3d,
        "ppi",
        x_coords=x_coords,
        y_coords=y_coords,
        z_levels=z_levels,
        elevation_deg=elevation_deg
    )
    
    assert result.shape == (20, 30)
    assert isinstance(result, np.ma.MaskedArray)
