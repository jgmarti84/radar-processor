"""
Tests for masked array edge cases in radar data processing.

These tests cover real-world scenarios where PyART reads files with 
broken/incomplete masks, containing -inf, +inf, and NaN values.
"""
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
import pyart

from radar_processor.processor import (
    _prepare_radar_field,
    create_colmax,
)


class TestMaskedArrayEdgeCases:
    """Test handling of various masked array edge cases."""
    
    @pytest.fixture
    def mock_radar_with_scalar_mask(self):
        """
        Create mock radar with scalar mask=False (broken mask).
        
        This simulates real-world files where PyART doesn't detect
        invalid values during file reading.
        """
        radar = MagicMock(spec=pyart.core.Radar)
        radar.nsweeps = 3
        radar.fixed_angle = {'data': np.array([0.5, 1.5, 2.5])}
        
        # Create data with invalid values but scalar mask
        data_array = np.random.uniform(-10, 50, (300, 100)).astype(np.float32)
        
        # Inject invalid values at specific locations
        data_array[0:10, 0:5] = -np.inf  # -infinity
        data_array[10:20, 0:5] = np.inf   # +infinity
        data_array[20:30, 0:5] = np.nan   # NaN
        
        # Create masked array with SCALAR mask (broken)
        masked_data = np.ma.array(data_array, mask=False)
        
        radar.fields = {
            'DBZH': {
                'data': masked_data,
                'units': 'dBZ',
                'long_name': 'Reflectivity'
            }
        }
        
        return radar
    
    @pytest.fixture
    def mock_radar_with_array_mask(self):
        """
        Create mock radar with proper boolean array mask.
        
        This is the expected/correct behavior from PyART.
        """
        radar = MagicMock(spec=pyart.core.Radar)
        radar.nsweeps = 3
        radar.fixed_angle = {'data': np.array([0.5, 1.5, 2.5])}
        
        # Create data and proper mask
        data_array = np.random.uniform(-10, 50, (300, 100)).astype(np.float32)
        mask_array = np.zeros((300, 100), dtype=bool)
        
        # Mark some values as masked
        mask_array[0:10, 0:5] = True
        mask_array[10:20, 0:5] = True
        mask_array[20:30, 0:5] = True
        
        # Create properly masked array
        masked_data = np.ma.array(data_array, mask=mask_array)
        
        radar.fields = {
            'DBZH': {
                'data': masked_data,
                'units': 'dBZ',
                'long_name': 'Reflectivity'
            }
        }
        
        return radar
    
    @pytest.fixture
    def mock_radar_with_mixed_issues(self):
        """
        Create mock radar with both scalar mask AND invalid values.
        
        This is the worst-case scenario from real files.
        """
        radar = MagicMock(spec=pyart.core.Radar)
        radar.nsweeps = 3
        radar.fixed_angle = {'data': np.array([0.5, 1.5, 2.5])}
        
        # Create data with various invalid patterns
        data_array = np.random.uniform(-10, 50, (300, 100)).astype(np.float32)
        
        # Mix of different invalid values
        data_array[0:50, 0:10] = -np.inf
        data_array[50:100, 10:20] = np.inf
        data_array[100:150, 20:30] = np.nan
        data_array[150:200, 30:40] = -999.0  # Another common fill value
        
        # Scalar mask (broken)
        masked_data = np.ma.array(data_array, mask=False)
        
        radar.fields = {
            'DBZH': {
                'data': masked_data,
                'units': 'dBZ',
                'long_name': 'Reflectivity'
            }
        }
        
        return radar
    
    def test_scalar_mask_replaced_with_fill_value(self, mock_radar_with_scalar_mask):
        """Test that invalid values are replaced even with scalar mask."""
        # Mock create_colmax to avoid PyART's composite_reflectivity
        mock_colmax = MagicMock()
        mock_colmax.fields = {'composite_reflectivity': {'data': np.ones((100, 100))}}
        
        with patch('radar_processor.processor.create_colmax', return_value=mock_colmax):
            radar_out, field_out = _prepare_radar_field(
                mock_radar_with_scalar_mask, 'DBZH', 'COLMAX', 4000
            )
        
        # Should create filled_DBZH field
        assert 'filled_DBZH' in mock_radar_with_scalar_mask.add_field_like.call_args[0]
        
        # Get the filled data that was added
        filled_data = mock_radar_with_scalar_mask.add_field_like.call_args[0][2]
        
        # Verify no invalid values remain
        assert not np.any(np.isnan(filled_data)), "NaN values should be replaced"
        assert not np.any(np.isinf(filled_data)), "Inf values should be replaced"
        assert np.all(np.isfinite(filled_data)), "All values should be finite"
        
        # Verify that invalid locations now contain -30
        assert filled_data[5, 2] == -30, "Invalid values should be -30"
        assert filled_data[15, 2] == -30, "Invalid values should be -30"
        assert filled_data[25, 2] == -30, "Invalid values should be -30"
    
    def test_array_mask_properly_handled(self, mock_radar_with_array_mask):
        """Test that proper array masks are handled correctly."""
        # Mock create_colmax
        mock_colmax = MagicMock()
        mock_colmax.fields = {'composite_reflectivity': {'data': np.ones((100, 100))}}
        
        with patch('radar_processor.processor.create_colmax', return_value=mock_colmax):
            radar_out, field_out = _prepare_radar_field(
                mock_radar_with_array_mask, 'DBZH', 'COLMAX', 4000
            )
        
        # Get the filled data
        filled_data = mock_radar_with_array_mask.add_field_like.call_args[0][2]
        
        # Verify masked locations are filled with -30
        assert filled_data[5, 2] == -30, "Masked values should be -30"
        assert filled_data[15, 2] == -30, "Masked values should be -30"
        
        # Verify valid data is preserved
        valid_value = filled_data[50, 50]
        assert -10 <= valid_value <= 50, "Valid data should be preserved"
    
    def test_mixed_invalid_values_all_replaced(self, mock_radar_with_mixed_issues):
        """Test that all types of invalid values are replaced."""
        # Mock create_colmax
        mock_colmax = MagicMock()
        mock_colmax.fields = {'composite_reflectivity': {'data': np.ones((100, 100))}}
        
        with patch('radar_processor.processor.create_colmax', return_value=mock_colmax):
            radar_out, field_out = _prepare_radar_field(
                mock_radar_with_mixed_issues, 'DBZH', 'COLMAX', 4000
            )
        
        filled_data = mock_radar_with_mixed_issues.add_field_like.call_args[0][2]
        
        # Verify ALL invalid values replaced
        assert not np.any(np.isnan(filled_data)), "All NaN removed"
        assert not np.any(np.isinf(filled_data)), "All inf removed"
        assert np.all(np.isfinite(filled_data)), "All values finite"
        
        # Check specific invalid regions now contain -30
        assert np.all(filled_data[0:50, 0:10] == -30), "-inf regions filled"
        assert np.all(filled_data[50:100, 10:20] == -30), "+inf regions filled"
        assert np.all(filled_data[100:150, 20:30] == -30), "NaN regions filled"
    
    def test_ppi_product_not_affected(self, mock_radar_with_scalar_mask):
        """Test that PPI product doesn't trigger filling."""
        radar_out, field_out = _prepare_radar_field(
            mock_radar_with_scalar_mask, 'DBZH', 'PPI', 4000
        )
        
        # PPI should not create filled_DBZH
        assert field_out == 'DBZH'
        assert radar_out is mock_radar_with_scalar_mask
        
        # add_field_like should not be called for PPI
        mock_radar_with_scalar_mask.add_field_like.assert_not_called()
    
    def test_non_dbzh_field_not_affected(self, mock_radar_with_scalar_mask):
        """Test that non-DBZH fields don't trigger filling."""
        # Add a different field
        mock_radar_with_scalar_mask.fields['ZDR'] = {
            'data': np.ma.array(np.ones((300, 100))),
            'units': 'dB'
        }
        
        # Mock create_colmax (though it shouldn't be called for non-DBZH)
        mock_colmax = MagicMock()
        mock_colmax.fields = {'composite_reflectivity': {'data': np.ones((100, 100))}}
        
        with patch('radar_processor.processor.create_colmax', return_value=mock_colmax):
            radar_out, field_out = _prepare_radar_field(
                mock_radar_with_scalar_mask, 'ZDR', 'COLMAX', 4000
            )
        
        # Should return composite_reflectivity field after COLMAX
        assert field_out == 'composite_reflectivity'


class TestColmaxWithInvalidData:
    """Test COLMAX creation with various invalid data scenarios."""
    
    @pytest.fixture
    def mock_radar_for_colmax(self):
        """Create radar with filled_DBZH field for COLMAX testing."""
        radar = MagicMock(spec=pyart.core.Radar)
        radar.nsweeps = 5
        radar.nrays = 360
        radar.ngates = 200
        radar.fixed_angle = {'data': np.array([0.5, 1.5, 2.5, 3.5, 4.5])}
        
        # Create clean filled_DBZH data (no inf/nan)
        # Shape: (5 sweeps * 360 rays, 200 gates) = (1800, 200)
        clean_data = np.random.uniform(-20, 50, (1800, 200)).astype(np.float32)
        
        # Add some -30 values (no-data areas)
        clean_data[0:100, 0:10] = -30
        
        radar.fields = {
            'filled_DBZH': {
                'data': clean_data,
                'units': 'dBZ',
                'long_name': 'Reflectivity'
            }
        }
        
        return radar
    
    @pytest.fixture
    def mock_radar_with_remaining_invalids(self):
        """Create radar where filled_DBZH still has invalid values (shouldn't happen)."""
        radar = MagicMock(spec=pyart.core.Radar)
        radar.nsweeps = 5
        radar.nrays = 360
        radar.ngates = 200
        radar.fixed_angle = {'data': np.array([0.5, 1.5, 2.5, 3.5, 4.5])}
        
        # Data with remaining invalid values
        data = np.random.uniform(-20, 50, (1800, 200)).astype(np.float32)
        data[0:50, 0:10] = np.nan  # Shouldn't happen after our fix
        
        radar.fields = {
            'filled_DBZH': {
                'data': data,
                'units': 'dBZ',
                'long_name': 'Reflectivity'
            }
        }
        
        return radar
    
    def test_colmax_with_clean_data(self, mock_radar_for_colmax):
        """Test that COLMAX works correctly with clean filled data."""
        # Mock PyART's composite_reflectivity function
        mock_composite = MagicMock()
        composite_data = np.ma.array(
            np.random.uniform(-20, 50, (360, 200)),
            mask=np.zeros((360, 200), dtype=bool)
        )
        mock_composite.fields = {
            'composite_reflectivity': {
                'data': composite_data,
                'units': 'dBZ',
                'long_name': 'Composite Reflectivity'
            }
        }
        
        with patch('radar_processor.processor.pyart.retrieve.composite_reflectivity',
                   return_value=mock_composite):
            result = create_colmax(mock_radar_for_colmax)
        
        # Verify result
        assert 'composite_reflectivity' in result.fields
        comp_data = result.fields['composite_reflectivity']['data']
        
        # Should only mask NaN values (not -30)
        assert isinstance(comp_data, np.ma.MaskedArray)
        assert not np.all(comp_data.mask), "Not all should be masked"
    
    def test_colmax_output_has_no_invalids(self, mock_radar_for_colmax):
        """Test that COLMAX output never contains inf or nan in unmasked regions."""
        # Create a realistic composite output with some NaN
        mock_composite = MagicMock()
        data = np.random.uniform(-20, 50, (360, 200)).astype(np.float32)
        data[0:10, 0:5] = np.nan  # Some NaN values from PyART
        
        mock_composite.fields = {
            'composite_reflectivity': {
                'data': data,
                'units': 'dBZ'
            }
        }
        
        with patch('radar_processor.processor.pyart.retrieve.composite_reflectivity',
                   return_value=mock_composite):
            result = create_colmax(mock_radar_for_colmax)
        
        comp_data = result.fields['composite_reflectivity']['data']
        
        # NaN values should be masked
        assert isinstance(comp_data, np.ma.MaskedArray)
        if hasattr(comp_data.mask, '__getitem__'):
            assert np.all(comp_data.mask[0:10, 0:5]), "NaN locations should be masked"
        
        # Unmasked data should have no inf/nan
        unmasked_data = comp_data.compressed()  # Get only unmasked values
        assert not np.any(np.isnan(unmasked_data)), "Unmasked data has no NaN"
        assert not np.any(np.isinf(unmasked_data)), "Unmasked data has no inf"


class TestRealWorldScenarios:
    """Test scenarios based on actual problematic radar files."""
    
    def test_rma1_file_pattern(self):
        """
        Test the specific pattern found in RMA1 files.
        
        Pattern: mask=False (scalar), data contains -inf/+inf/nan
        """
        radar = MagicMock(spec=pyart.core.Radar)
        radar.nsweeps = 15
        radar.fixed_angle = {'data': np.linspace(0.5, 15, 15)}
        
        # Replicate RMA1 pattern: 38% valid, rest invalid
        total_size = 5400 * 652
        valid_count = int(total_size * 0.38)
        
        data = np.full((5400, 652), -np.inf, dtype=np.float32)
        
        # Scatter valid data randomly
        valid_indices = np.random.choice(total_size, valid_count, replace=False)
        data.flat[valid_indices] = np.random.uniform(-25, 58, valid_count)
        
        # Add some +inf and nan
        data[100:200, 0:10] = np.inf
        data[200:300, 0:10] = np.nan
        
        # Scalar mask (broken)
        masked_data = np.ma.array(data, mask=False)
        
        radar.fields = {
            'DBZH': {
                'data': masked_data,
                'units': 'dBZ'
            }
        }
        
        # Process
        mock_colmax = MagicMock()
        mock_colmax.fields = {'composite_reflectivity': {'data': np.ones((360, 652))}}
        
        with patch('radar_processor.processor.create_colmax', return_value=mock_colmax):
            radar_out, field_out = _prepare_radar_field(radar, 'DBZH', 'COLMAX', 4000)
        
        # Verify fix
        filled_data = radar.add_field_like.call_args[0][2]
        
        assert not np.any(np.isinf(filled_data)), "All inf replaced"
        assert not np.any(np.isnan(filled_data)), "All nan replaced"
        
        # Valid data preserved
        valid_data = filled_data[filled_data > -30]
        assert len(valid_data) > 0, "Valid data exists"
        assert np.all((-25 <= valid_data) & (valid_data <= 58)), "Valid range preserved"
    
    def test_performance_with_large_arrays(self):
        """Test that the fix doesn't significantly impact performance."""
        import time
        
        radar = MagicMock(spec=pyart.core.Radar)
        radar.nsweeps = 15
        radar.fixed_angle = {'data': np.linspace(0.5, 15, 15)}
        
        # Large array (realistic size)
        data = np.random.uniform(-20, 50, (5400, 652)).astype(np.float32)
        
        # Add invalid values (10% of data)
        invalid_count = int(5400 * 652 * 0.1)
        invalid_indices = np.random.choice(5400 * 652, invalid_count, replace=False)
        data.flat[invalid_indices] = np.nan
        
        masked_data = np.ma.array(data, mask=False)
        radar.fields = {'DBZH': {'data': masked_data, 'units': 'dBZ'}}
        
        # Time the operation
        mock_colmax = MagicMock()
        mock_colmax.fields = {'composite_reflectivity': {'data': np.ones((360, 652))}}
        
        start = time.time()
        with patch('radar_processor.processor.create_colmax', return_value=mock_colmax):
            _prepare_radar_field(radar, 'DBZH', 'COLMAX', 4000)
        elapsed = time.time() - start
        
        # Should complete quickly (< 100ms for this size)
        assert elapsed < 0.1, f"Processing too slow: {elapsed:.3f}s"
