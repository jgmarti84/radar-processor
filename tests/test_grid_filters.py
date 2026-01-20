"""
Unit tests for GridFilter - post-interpolation grid value filtering.

Tests the GridFilter class for filtering 2D grid values after interpolation.
"""

import pytest
import numpy as np
from radar_grid.filters import GridFilter


class TestGridFilterBasics:
    """Test basic GridFilter functionality."""
    
    @pytest.fixture
    def sample_grid(self):
        """Create a sample 2D grid."""
        return np.array([
            [10.0, 20.0, 30.0, 40.0],
            [15.0, 25.0, 35.0, 45.0],
            [12.0, 22.0, 32.0, 42.0]
        ])
    
    def test_gridfilter_creation(self):
        """Test that GridFilter can be instantiated."""
        gf = GridFilter()
        assert gf is not None
    
    def test_apply_below_basic(self, sample_grid):
        """Test basic apply_below functionality."""
        gf = GridFilter()
        result = gf.apply_below(sample_grid, 15)
        
        # Values < 15 should be NaN
        assert np.isnan(result[0, 0])  # 10
        assert np.isnan(result[2, 0])  # 12
        
        # Values >= 15 should remain
        assert result[0, 1] == 20.0
        assert result[1, 0] == 15.0
    
    def test_apply_below_preserves_original(self, sample_grid):
        """Test that apply_below doesn't modify original."""
        original = sample_grid.copy()
        gf = GridFilter()
        result = gf.apply_below(sample_grid, 15)
        
        # Original should be unchanged
        np.testing.assert_array_equal(sample_grid, original)
        
        # Result should be different
        assert not np.array_equal(result, sample_grid)


class TestGridFilterThresholds:
    """Test threshold-based filtering."""
    
    @pytest.fixture
    def sample_grid(self):
        """Create a test grid with COLMAX-like values."""
        return np.array([
            [8.0, 18.0, 28.0, 38.0, 48.0],
            [12.0, 22.0, 32.0, 42.0, 52.0],
            [10.0, 20.0, 30.0, 40.0, 50.0],
            [14.0, 24.0, 34.0, 44.0, 54.0]
        ])
    
    def test_apply_below_colmax_example(self, sample_grid):
        """Test the exact example from user: filter colmax below 15."""
        gf = GridFilter()
        result = gf.apply_below(sample_grid, 15)
        
        # Values < 15 should be NaN
        assert np.isnan(result[0, 0])  # 8
        # assert np.isnan(result[0, 1])  # 18... wait, this is >= 15
        assert result[0, 1] == 18.0
        
        assert np.isnan(result[1, 0])  # 12
        assert result[1, 1] == 22.0
        
        # All values >= 15 should remain
        assert np.all(result[~np.isnan(result)] >= 15)
    
    def test_apply_above(self, sample_grid):
        """Test excluding values above threshold."""
        gf = GridFilter()
        result = gf.apply_above(sample_grid, 40)
        
        # Values > 40 should be NaN
        assert np.isnan(result[0, 4])  # 48
        assert np.isnan(result[1, 4])  # 52
        assert np.isnan(result[3, 4])  # 54
        
        # Values <= 40 should remain
        assert result[0, 0] == 8.0
        assert result[0, 3] == 38.0
    
    def test_apply_outside_range(self, sample_grid):
        """Test excluding values outside range."""
        gf = GridFilter()
        result = gf.apply_outside_range(sample_grid, 15, 45)
        
        # Values < 15 should be NaN
        assert np.isnan(result[0, 0])  # 8
        assert np.isnan(result[1, 0])  # 12
        
        # Values > 45 should be NaN
        assert np.isnan(result[0, 4])  # 48
        assert np.isnan(result[1, 4])  # 52
        
        # Values in range should remain
        assert result[0, 1] == 18.0  # 18 in [15, 45]
        assert result[0, 3] == 38.0  # 38 in [15, 45]


class TestGridFilterInvalid:
    """Test invalid value filtering."""
    
    def test_apply_invalid_handles_nan(self):
        """Test that apply_invalid handles NaN values."""
        grid = np.array([
            [10.0, np.nan, 30.0],
            [15.0, 25.0, np.nan]
        ])
        
        gf = GridFilter()
        result = gf.apply_invalid(grid)
        
        # NaN values should remain NaN
        assert np.isnan(result[0, 1])
        assert np.isnan(result[1, 2])
        
        # Valid values should remain unchanged
        assert result[0, 0] == 10.0
        assert result[0, 2] == 30.0
        assert result[1, 0] == 15.0
        assert result[1, 1] == 25.0
    
    def test_apply_invalid_handles_inf(self):
        """Test that apply_invalid handles infinity values."""
        grid = np.array([
            [10.0, np.inf, 30.0],
            [15.0, -np.inf, 35.0]
        ])
        
        gf = GridFilter()
        result = gf.apply_invalid(grid)
        
        # Inf values should become NaN
        assert np.isnan(result[0, 1])
        assert np.isnan(result[1, 1])
        
        # Valid values should remain
        assert result[0, 0] == 10.0


class TestGridFilterCustom:
    """Test custom filter functionality."""
    
    def test_apply_custom_simple_function(self):
        """Test apply_custom with a simple function."""
        grid = np.array([
            [5.0, 10.0, 15.0],
            [20.0, 25.0, 30.0]
        ])
        
        # Filter: mark as excluded if value is < 12
        gf = GridFilter()
        result = gf.apply_custom(grid, lambda x: x < 12)
        
        assert np.isnan(result[0, 0])  # 5
        assert np.isnan(result[0, 1])  # 10
        assert result[0, 2] == 15.0    # 15
        assert result[1, 0] == 20.0    # 20
    
    def test_apply_custom_complex_function(self):
        """Test apply_custom with more complex logic."""
        grid = np.array([
            [5.0, 10.0, 15.0, 20.0],
            [25.0, 30.0, 35.0, 40.0]
        ])
        
        # Filter: mark as excluded if value is odd (for integer conversion)
        gf = GridFilter()
        result = gf.apply_custom(grid, lambda x: (x.astype(int) % 2) != 0)
        
        # 5, 15, 25, 35 are odd -> NaN
        assert np.isnan(result[0, 0])  # 5
        assert np.isnan(result[0, 2])  # 15
        assert np.isnan(result[1, 0])  # 25
        assert np.isnan(result[1, 2])  # 35
        
        # 10, 20, 30, 40 are even -> keep
        assert result[0, 1] == 10.0
        assert result[0, 3] == 20.0


class TestGridFilterCustomFillValue:
    """Test using custom fill values."""
    
    def test_apply_below_custom_fill_value(self):
        """Test apply_below with custom fill value."""
        grid = np.array([[5.0, 15.0, 25.0]])
        
        gf = GridFilter()
        result = gf.apply_below(grid, 15, fill_value=-9999)
        
        assert result[0, 0] == -9999  # 5 < 15
        assert result[0, 1] == 15.0   # 15 >= 15
        assert result[0, 2] == 25.0
    
    def test_apply_above_custom_fill_value(self):
        """Test apply_above with custom fill value."""
        grid = np.array([[5.0, 15.0, 25.0]])
        
        gf = GridFilter()
        result = gf.apply_above(grid, 15, fill_value=-999)
        
        assert result[0, 0] == 5.0
        assert result[0, 1] == 15.0    # 15 not > 15
        assert result[0, 2] == -999    # 25 > 15


class TestGridFilterChaining:
    """Test chaining multiple filters."""
    
    def test_chain_multiple_filters(self):
        """Test applying multiple filters in sequence."""
        grid = np.array([
            [5.0, 15.0, 25.0, 35.0],
            [10.0, 20.0, 30.0, 40.0]
        ])
        
        gf = GridFilter()
        
        # Apply multiple filters
        result = grid.copy()
        result = gf.apply_below(result, 12)  # Remove < 12
        result = gf.apply_above(result, 35)  # Remove > 35
        
        # Check results
        # 5 < 12 -> NaN
        assert np.isnan(result[0, 0])
        # 10 < 12 -> NaN
        assert np.isnan(result[1, 0])
        # 15 in [12, 35] -> keep
        assert result[0, 1] == 15.0
        # 40 > 35 -> NaN
        assert np.isnan(result[1, 3])


class TestGridFilterRealWorldScenario:
    """Test realistic radar product filtering."""
    
    def test_colmax_filtering_scenario(self):
        """
        Test the exact scenario the user described:
        Filter a COLMAX product to set values < 15 to NaN.
        """
        # Simulate a COLMAX product
        ny, nx = 20, 20
        colmax = np.random.uniform(5, 50, (ny, nx))
        
        # Some values are already NaN
        colmax[5:8, 5:8] = np.nan
        
        # Apply the filter: exclude values < 15
        gf = GridFilter()
        filtered = gf.apply_below(colmax, 15)
        
        # Check that all non-NaN values are >= 15
        valid_mask = ~np.isnan(filtered)
        if np.any(valid_mask):
            assert np.all(filtered[valid_mask] >= 15)
        
        # Check that NaN values remain NaN
        original_nan_mask = np.isnan(colmax)
        assert np.all(np.isnan(filtered[original_nan_mask]))
    
    def test_combined_filtering_scenario(self):
        """
        Test filtering COLMAX with both min and max values.
        """
        # Simulate COLMAX: values generally 5-45
        ny, nx = 20, 20
        colmax = np.random.uniform(5, 45, (ny, nx))
        
        # Apply range filter: keep only 15-40
        gf = GridFilter()
        filtered = gf.apply_outside_range(colmax, 15, 40)
        
        # All non-NaN values should be in [15, 40]
        valid_mask = ~np.isnan(filtered)
        if np.any(valid_mask):
            assert np.all(filtered[valid_mask] >= 15)
            assert np.all(filtered[valid_mask] <= 40)

