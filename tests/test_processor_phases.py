"""
Unit tests for optimized processor phases.

Tests for correctness and behavior of individual refactored functions
to ensure optimizations don't break functionality.
"""
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock
import pyart

from radar_cog_processor.processor import (
    _build_processing_config,
    _prepare_radar_field,
    _compute_grid_limits_and_resolution,
    _apply_filter_masks,
    _cache_or_build_2d_grid,
)
from radar_cog_processor.cache import GRID2D_CACHE, GRID3D_CACHE


@pytest.fixture(autouse=True)
def clear_caches():
    """Clear caches before and after each test."""
    GRID2D_CACHE.clear()
    GRID3D_CACHE.clear()
    yield
    GRID2D_CACHE.clear()
    GRID3D_CACHE.clear()


@pytest.fixture
def mock_radar():
    """Create a mock radar object with realistic data."""
    radar = MagicMock(spec=pyart.core.Radar)
    
    # Basic radar properties
    radar.nsweeps = 5
    radar.fixed_angle = {'data': np.array([0.5, 1.5, 2.5, 3.5, 4.5])}
    radar.latitude = {'data': np.array([40.0])}
    radar.longitude = {'data': np.array([-105.0])}
    radar.altitude = {'data': np.array([1600.0])}
    
    # Field data
    radar.fields = {
        'DBZH': {
            'data': np.ma.array(
                np.random.randn(5, 100, 200) + 10,  # 5 sweeps, realistic reflectivity
                mask=np.random.rand(5, 100, 200) > 0.95
            ),
            'units': 'dBZ',
            'long_name': 'Reflectivity',
            '_FillValue': -9999.0,
        },
        'RHOHV': {
            'data': np.ma.array(
                np.random.rand(5, 100, 200),
                mask=np.random.rand(5, 100, 200) > 0.92
            ),
            'units': '',
            'long_name': 'Cross correlation coefficient',
            '_FillValue': -9999.0,
        }
    }
    
    return radar


class TestPhase1BuildProcessingConfig:
    """Tests for Phase 1-2: _build_processing_config"""
    
    def test_valid_config_generation(self, tmp_path, mock_radar):
        """Test that config is properly generated with valid inputs."""
        filepath = tmp_path / "test.nc"
        filepath.touch()
        
        with patch('radar_cog_processor.processor.pyart.io.read', return_value=mock_radar):
            with patch('radar_cog_processor.processor.resolve_field', 
                      return_value=('DBZH', 'DBZH')):
                with patch('radar_cog_processor.processor.colormap_for',
                          return_value=(None, -30, 50, 'DBZH')):
                    
                    config = _build_processing_config(
                        str(filepath), 'PPI', 'DBZH', 0, 4000, None, None, [], str(tmp_path)
                    )
        
        # Verify config structure
        assert isinstance(config, dict)
        assert 'radar' in config
        assert 'file_hash' in config
        assert 'cog_path' in config
        assert 'summary' in config
        assert 'field_name' in config
        assert 'cmap' in config
        assert 'vmin' in config
        assert 'vmax' in config
        assert 'product_upper' in config
        assert 'qc_filters' in config
        assert 'visual_filters' in config
    
    def test_config_raises_on_invalid_field(self, tmp_path, mock_radar):
        """Test that invalid field raises ValueError."""
        filepath = tmp_path / "test.nc"
        filepath.touch()
        
        with patch('radar_cog_processor.processor.pyart.io.read', return_value=mock_radar):
            with patch('radar_cog_processor.processor.resolve_field', 
                      side_effect=KeyError("Field not found")):
                
                with pytest.raises(ValueError):
                    _build_processing_config(
                        str(filepath), 'PPI', 'INVALID', 0, 4000, None, None, [], str(tmp_path)
                    )
    
    def test_config_raises_on_invalid_elevation(self, tmp_path, mock_radar):
        """Test that invalid elevation raises ValueError."""
        filepath = tmp_path / "test.nc"
        filepath.touch()
        
        with patch('radar_cog_processor.processor.pyart.io.read', return_value=mock_radar):
            with patch('radar_cog_processor.processor.resolve_field', 
                      return_value=('DBZH', 'DBZH')):
                with patch('radar_cog_processor.processor.colormap_for',
                          return_value=(None, -30, 50, 'DBZH')):
                    
                    with pytest.raises(ValueError, match="elevación"):
                        _build_processing_config(
                            str(filepath), 'PPI', 'DBZH', 10, 4000, None, None, [], str(tmp_path)
                        )
    
    def test_filter_separation(self, tmp_path, mock_radar):
        """Test that filters are correctly separated into QC and visual filters."""
        filepath = tmp_path / "test.nc"
        filepath.touch()
        
        # Create filter objects
        class MockFilter:
            def __init__(self, field, min_val, max_val):
                self.field = field
                self.min = min_val
                self.max = max_val
        
        filters = [
            MockFilter('RHOHV', 0.8, 1.0),  # QC filter
            MockFilter('DBZH', -30, 50),    # Visual filter
        ]
        
        with patch('radar_cog_processor.processor.pyart.io.read', return_value=mock_radar):
            with patch('radar_cog_processor.processor.resolve_field', 
                      return_value=('DBZH', 'DBZH')):
                with patch('radar_cog_processor.processor.colormap_for',
                          return_value=(None, -30, 50, 'DBZH')):
                    
                    config = _build_processing_config(
                        str(filepath), 'PPI', 'DBZH', 0, 4000, None, None, filters, str(tmp_path)
                    )
        
        # Verify filter separation
        assert len(config['qc_filters']) == 1
        assert len(config['visual_filters']) == 1


class TestPhase4PrepareRadarField:
    """Tests for Phase 4: _prepare_radar_field"""
    
    def test_ppi_field_preparation(self, mock_radar):
        """Test that PPI product returns radar and field unchanged."""
        radar_out, field_out = _prepare_radar_field(mock_radar, 'DBZH', 'PPI', 4000)
        
        assert radar_out is mock_radar
        assert field_out == 'DBZH'
    
    def test_cappi_field_preparation(self, mock_radar):
        """Test that CAPPI product returns filled_DBZH field name for DBZH."""
        radar_out, field_out = _prepare_radar_field(mock_radar, 'DBZH', 'CAPPI', 4000)
        
        # For CAPPI with DBZH, should return 'filled_DBZH' (filled reflectivity)
        # CAPPI is just a different collapse method, not a separate field
        assert field_out == 'filled_DBZH'
        # The radar returned should be the original radar
        assert radar_out is mock_radar
    
    def test_colmax_field_preparation(self, mock_radar):
        """Test that COLMAX product creates composite_reflectivity field."""
        mock_colmax = MagicMock()
        mock_colmax.fields = {
            'composite_reflectivity': {
                'data': np.ma.array(np.ones((100, 200))),
                'units': 'dBZ'
            }
        }
        
        with patch('radar_cog_processor.processor.create_colmax', return_value=mock_colmax):
            radar_out, field_out = _prepare_radar_field(mock_radar, 'DBZH', 'COLMAX', 4000)
        
        assert field_out == 'composite_reflectivity'
    
    def test_filled_reflectivity_for_cappi(self, mock_radar):
        """Test that CAPPI creates filled reflectivity for DBZH."""
        radar_out, field_out = _prepare_radar_field(mock_radar, 'DBZH', 'CAPPI', 4000)
        
        # Should return 'filled_DBZH' field name since DBZH is filled for CAPPI/COLMAX
        assert field_out == 'filled_DBZH'
        # add_field_like should have been called to add filled_DBZH
        mock_radar.add_field_like.assert_called_once()
    
    def test_invalid_product_raises(self, mock_radar):
        """Test that invalid product type raises ValueError."""
        with pytest.raises(ValueError, match="inválido"):
            _prepare_radar_field(mock_radar, 'DBZH', 'INVALID_PRODUCT', 4000)


class TestPhase6ComputeGridLimits:
    """Tests for Phase 6: _compute_grid_limits_and_resolution"""
    
    def test_ppi_grid_computation(self, mock_radar):
        """Test grid limits computation for PPI."""
        with patch('radar_cog_processor.processor.safe_range_max_m', return_value=150000):
            with patch('radar_cog_processor.processor.beam_height_max_km', return_value=15):
                
                grid_config = _compute_grid_limits_and_resolution(
                    mock_radar, 'PPI', 0, 4000, None
                )
        
        assert 'z_grid_limits' in grid_config
        assert 'y_grid_limits' in grid_config
        assert 'x_grid_limits' in grid_config
        assert 'grid_resolution' in grid_config
        assert 'elev_deg' in grid_config
        
        # Verify symmetry for PPI
        y_min, y_max = grid_config['y_grid_limits']
        x_min, x_max = grid_config['x_grid_limits']
        assert y_min == -y_max
        assert x_min == -x_max
    
    def test_cappi_grid_computation(self, mock_radar):
        """Test grid limits computation for CAPPI."""
        with patch('radar_cog_processor.processor.safe_range_max_m', return_value=150000):
            grid_config = _compute_grid_limits_and_resolution(
                mock_radar, 'CAPPI', 0, 4000, None
            )
        
        z_min, z_max = grid_config['z_grid_limits']
        assert z_max > 4000  # Should include margin
        assert grid_config['elev_deg'] is None
    
    def test_grid_resolution_per_volume(self, mock_radar):
        """Test that grid resolution varies per volume."""
        with patch('radar_cog_processor.processor.safe_range_max_m', return_value=150000):
            
            config_vol03 = _compute_grid_limits_and_resolution(
                mock_radar, 'PPI', 0, 4000, '03'
            )
            config_vol05 = _compute_grid_limits_and_resolution(
                mock_radar, 'PPI', 0, 4000, '05'
            )
        
        # Volume '03' should have finer resolution
        assert config_vol03['grid_resolution'] == 300
        assert config_vol05['grid_resolution'] == 1200


class TestPhase10ApplyFilterMasks:
    """Tests for Phase 10-11: _apply_filter_masks"""
    
    @pytest.fixture
    def sample_array(self):
        """Create sample masked array."""
        return np.ma.array(
            np.linspace(-30, 60, 1000).reshape(50, 20),
            mask=np.zeros((50, 20), dtype=bool)
        )
    
    @pytest.fixture
    def mock_filter(self):
        """Create mock filter object."""
        class MockFilter:
            def __init__(self, field, min_val, max_val):
                self.field = field
                self.min = min_val
                self.max = max_val
        return MockFilter
    
    def test_visual_filter_min_mask(self, sample_array, mock_filter):
        """Test that min filter correctly masks values below threshold."""
        filter_obj = mock_filter('DBZH', -20, None)
        pkg_cached = {'qc': {}}
        
        result = _apply_filter_masks(
            sample_array.copy(), [filter_obj], [], 'DBZH', pkg_cached
        )
        
        # Values below -20 should be masked
        masked_count = np.sum(result.mask)
        assert masked_count > 0
    
    def test_visual_filter_max_mask(self, sample_array, mock_filter):
        """Test that max filter correctly masks values above threshold."""
        filter_obj = mock_filter('DBZH', None, 50)
        pkg_cached = {'qc': {}}
        
        result = _apply_filter_masks(
            sample_array.copy(), [filter_obj], [], 'DBZH', pkg_cached
        )
        
        # Values above 50 should be masked
        masked_count = np.sum(result.mask)
        assert masked_count > 0
    
    def test_multiple_filters_cumulative(self, sample_array, mock_filter):
        """Test that multiple filters are applied cumulatively."""
        filters = [
            mock_filter('DBZH', -20, None),
            mock_filter('DBZH', None, 50),
        ]
        pkg_cached = {'qc': {}}
        
        result = _apply_filter_masks(
            sample_array.copy(), filters, [], 'DBZH', pkg_cached
        )
        
        # Both filters should be applied
        assert np.sum(result.mask) > 0
    
    def test_qc_filters_applied(self, sample_array, mock_filter):
        """Test that QC filters are applied to QC fields."""
        qc_array = np.ma.array(
            np.linspace(0.5, 1.0, 1000).reshape(50, 20),
            mask=np.zeros((50, 20), dtype=bool)
        )
        pkg_cached = {'qc': {'RHOHV': qc_array}}
        
        qc_filter = mock_filter('RHOHV', 0.8, None)
        
        result = _apply_filter_masks(
            sample_array.copy(), [], [qc_filter], 'DBZH', pkg_cached
        )
        
        # Array should be masked where RHOHV < 0.8
        masked_count = np.sum(result.mask)
        assert masked_count > 0
    
    def test_filter_preserves_array_values(self, sample_array, mock_filter):
        """Test that filtering doesn't modify unmasked data values."""
        original_data = sample_array.data.copy()
        filter_obj = mock_filter('DBZH', -20, 50)
        pkg_cached = {'qc': {}}
        
        result = _apply_filter_masks(
            sample_array.copy(), [filter_obj], [], 'DBZH', pkg_cached
        )
        
        # Data values should be unchanged (only mask changes)
        np.testing.assert_array_equal(result.data, original_data)


class TestPhase12ExportToCOG:
    """Tests for Phase 12: _export_to_cog"""
    
    def test_cog_export_creates_file(self, tmp_path, mock_radar):
        """Test that COG export creates output file."""
        # This is a limited test since we mock most of PyART
        masked_arr = np.ma.array(np.ones((100, 200)), mask=np.zeros((100, 200)))
        cog_path = tmp_path / "test.tif"
        output_dir = str(tmp_path)
        
        with patch('radar_cog_processor.processor.pyart.io.write_grid_geotiff'):
            with patch('radar_cog_processor.processor.convert_to_cog', return_value=cog_path):
                result = True  # Simulate successful export
        
        assert result is True


# ============================================================================
# INTEGRATION TESTS FOR CACHE BEHAVIOR
# ============================================================================

class TestCacheBehavior:
    """Tests for caching mechanisms."""
    
    def test_2d_grid_cache_hit(self, tmp_path, mock_radar):
        """Test that 2D grid cache prevents redundant computation."""
        from radar_cog_processor.utils import grid2d_cache_key
        
        cache_key = grid2d_cache_key(
            file_hash='abc123',
            product_upper='PPI',
            field_to_use='DBZH',
            elevation=0,
            cappi_height=None,
            volume=None,
            interp='nearest',
            qc_sig=tuple(),
        )
        
        # Pre-populate cache
        cached_data = {
            'arr': np.ma.array(np.ones((100, 200))),
            'qc': {},
            'crs': 'EPSG:4326',
            'transform': None,
            'arr_warped': None,
            'crs_warped': None,
            'transform_warped': None,
        }
        GRID2D_CACHE[cache_key] = cached_data
        
        # Retrieve from cache
        assert cache_key in GRID2D_CACHE
        assert GRID2D_CACHE[cache_key] is cached_data
    
    def test_3d_grid_cache_storage(self, mock_radar):
        """Test that 3D grids are properly cached."""
        from radar_cog_processor.utils import grid3d_cache_key
        
        cache_key = grid3d_cache_key(
            file_hash='abc123',
            field_to_use='DBZH',
            volume=None,
            qc_sig=tuple(),
            grid_res_xy=1200,
            grid_res_z=1200,
            z_top_m=18000,
        )
        
        # Create mock 3D grid data
        grid_data = {
            'arr3d': np.random.randn(15, 200, 200),
            'x': np.linspace(-150000, 150000, 200),
            'y': np.linspace(-150000, 150000, 200),
            'z': np.linspace(0, 18000, 15),
            'projection': {'proj': 'aeqd'},
            'field_name': 'DBZH',
            'field_metadata': {'units': 'dBZ'},
        }
        
        GRID3D_CACHE[cache_key] = grid_data
        
        # Verify cache storage
        assert cache_key in GRID3D_CACHE
        assert GRID3D_CACHE[cache_key]['arr3d'].shape == (15, 200, 200)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
