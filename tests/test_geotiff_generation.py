"""
Unit tests for radar_grid GeoTIFF generation functionality.

Tests the geotiff module functions for creating GeoTIFF and COG files
from 2D radar product arrays.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from radar_grid.geometry import GridGeometry
from radar_grid.geotiff import (
    apply_colormap_to_array,
    create_geotiff,
    create_cog,
    save_product_as_geotiff,
    _string_to_resampling
)

try:
    import rasterio
    from rasterio.enums import Resampling
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False


# Test fixtures
@pytest.fixture
def sample_geometry():
    """Create a sample GridGeometry for testing."""
    nz, ny, nx = 10, 100, 100
    grid_shape = (nz, ny, nx)
    grid_limits = (
        (0.0, 10000.0),      # z: 0 to 10 km
        (-50000.0, 50000.0),  # y: -50 to 50 km
        (-50000.0, 50000.0)   # x: -50 to 50 km
    )
    
    # Create minimal sparse data
    n_points = nz * ny * nx
    indptr = np.arange(n_points + 1, dtype=np.int32)
    gate_indices = np.zeros(n_points, dtype=np.int32)
    weights = np.ones(n_points, dtype=np.float32)
    
    return GridGeometry(
        grid_shape=grid_shape,
        grid_limits=grid_limits,
        indptr=indptr,
        gate_indices=gate_indices,
        weights=weights,
        toa=12000.0,
        radar_altitude=100.0
    )


@pytest.fixture
def sample_2d_data():
    """Create sample 2D data array."""
    ny, nx = 100, 100
    # Create a gradient pattern
    y = np.linspace(0, 1, ny)
    x = np.linspace(0, 1, nx)
    xx, yy = np.meshgrid(x, y)
    data = 50 * (xx + yy) / 2  # Values from 0 to 50
    
    # Add some NaN values
    data[45:55, 45:55] = np.nan
    
    return data


class TestColormapApplication:
    """Test colormap application to 2D arrays."""
    
    def test_apply_colormap_basic(self, sample_2d_data):
        """Test basic colormap application."""
        result = apply_colormap_to_array(sample_2d_data, 'viridis')
        
        assert result.shape == (100, 100, 4)
        assert result.dtype == np.uint8
        assert np.all(result >= 0) and np.all(result <= 255)
    
    def test_apply_colormap_with_vmin_vmax(self, sample_2d_data):
        """Test colormap with custom vmin/vmax."""
        result = apply_colormap_to_array(sample_2d_data, 'viridis', vmin=0, vmax=70)
        
        assert result.shape == (100, 100, 4)
        assert result.dtype == np.uint8
    
    def test_apply_colormap_nan_handling(self, sample_2d_data):
        """Test that NaN values become transparent."""
        result = apply_colormap_to_array(sample_2d_data, 'viridis')
        
        # Check that NaN region (45:55, 45:55) has alpha = 0
        nan_region_alpha = result[45:55, 45:55, 3]
        assert np.all(nan_region_alpha == 0)
        
        # Check that valid data has alpha = 255
        valid_region_alpha = result[0:10, 0:10, 3]
        assert np.all(valid_region_alpha == 255)
    
    def test_apply_colormap_custom_cmap(self, sample_2d_data):
        """Test with custom colormap object."""
        cmap = plt.get_cmap('jet')
        result = apply_colormap_to_array(sample_2d_data, cmap)
        
        assert result.shape == (100, 100, 4)
        assert result.dtype == np.uint8
    
    def test_apply_colormap_fill_value(self, sample_2d_data):
        """Test fill_value parameter."""
        # Create data with specific fill value
        data = sample_2d_data.copy()
        data[45:55, 45:55] = -9999.0
        
        result = apply_colormap_to_array(data, 'viridis', fill_value=-9999.0)
        
        # Check that fill value region has alpha = 0
        fill_region_alpha = result[45:55, 45:55, 3]
        assert np.all(fill_region_alpha == 0)


class TestStringToResampling:
    """Test resampling method string conversion."""
    
    @pytest.mark.skipif(not RASTERIO_AVAILABLE, reason="rasterio not available")
    def test_nearest_method(self):
        """Test 'nearest' conversion."""
        result = _string_to_resampling('nearest')
        assert result == Resampling.nearest
    
    @pytest.mark.skipif(not RASTERIO_AVAILABLE, reason="rasterio not available")
    def test_average_method(self):
        """Test 'average' conversion."""
        result = _string_to_resampling('average')
        assert result == Resampling.average
    
    @pytest.mark.skipif(not RASTERIO_AVAILABLE, reason="rasterio not available")
    def test_invalid_method(self):
        """Test invalid method raises ValueError."""
        with pytest.raises(ValueError, match="Invalid resampling method"):
            _string_to_resampling('invalid_method')


@pytest.mark.skipif(not RASTERIO_AVAILABLE, reason="rasterio not available")
class TestGeoTIFFCreation:
    """Test GeoTIFF file creation."""
    
    def test_create_geotiff_basic(self, sample_2d_data, sample_geometry):
        """Test basic GeoTIFF creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'test.tif'
            
            result_path = create_geotiff(
                sample_2d_data,
                sample_geometry,
                radar_lat=40.5,
                radar_lon=-105.0,
                output_path=output_path,
                cmap='viridis'
            )
            
            assert result_path.exists()
            assert result_path.suffix == '.tif'
            
            # Verify with rasterio
            with rasterio.open(result_path) as src:
                assert src.count == 4  # RGBA
                assert src.width == 100
                assert src.height == 100
                assert src.dtypes[0] == 'uint8'
    
    def test_create_geotiff_shape_mismatch(self, sample_geometry):
        """Test that shape mismatch raises ValueError."""
        wrong_shape_data = np.zeros((50, 50))  # Wrong shape
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'test.tif'
            
            with pytest.raises(ValueError, match="does not match geometry"):
                create_geotiff(
                    wrong_shape_data,
                    sample_geometry,
                    radar_lat=40.5,
                    radar_lon=-105.0,
                    output_path=output_path
                )
    
    def test_create_geotiff_web_mercator(self, sample_2d_data, sample_geometry):
        """Test GeoTIFF with Web Mercator projection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'test_webmerc.tif'
            
            result_path = create_geotiff(
                sample_2d_data,
                sample_geometry,
                radar_lat=40.5,
                radar_lon=-105.0,
                output_path=output_path,
                projection='EPSG:3857'
            )
            
            assert result_path.exists()
            
            with rasterio.open(result_path) as src:
                assert src.crs.to_epsg() == 3857
    
    def test_create_geotiff_wgs84(self, sample_2d_data, sample_geometry):
        """Test GeoTIFF with WGS84 projection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'test_wgs84.tif'
            
            result_path = create_geotiff(
                sample_2d_data,
                sample_geometry,
                radar_lat=40.5,
                radar_lon=-105.0,
                output_path=output_path,
                projection='EPSG:4326'
            )
            
            assert result_path.exists()
            
            with rasterio.open(result_path) as src:
                assert src.crs.to_epsg() == 4326


@pytest.mark.skipif(not RASTERIO_AVAILABLE, reason="rasterio not available")
class TestCOGCreation:
    """Test Cloud-Optimized GeoTIFF creation."""
    
    def test_create_cog_basic(self, sample_2d_data, sample_geometry):
        """Test basic COG creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'test.cog'
            
            result_path = create_cog(
                sample_2d_data,
                sample_geometry,
                radar_lat=40.5,
                radar_lon=-105.0,
                output_path=output_path,
                cmap='viridis'
            )
            
            assert result_path.exists()
            
            with rasterio.open(result_path) as src:
                assert src.count == 4  # RGBA
                assert src.driver == 'COG'
                # Check that overviews exist
                assert len(src.overviews(1)) > 0
    
    def test_create_cog_custom_overviews(self, sample_2d_data, sample_geometry):
        """Test COG with custom overview factors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'test.cog'
            
            result_path = create_cog(
                sample_2d_data,
                sample_geometry,
                radar_lat=40.5,
                radar_lon=-105.0,
                output_path=output_path,
                overview_factors=[2, 4],
                resampling_method='nearest'
            )
            
            assert result_path.exists()
            
            with rasterio.open(result_path) as src:
                overviews = src.overviews(1)
                assert len(overviews) == 2
                assert overviews == [2, 4]
    
    def test_create_cog_no_overviews(self, sample_2d_data, sample_geometry):
        """Test COG without overviews."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'test.cog'
            
            result_path = create_cog(
                sample_2d_data,
                sample_geometry,
                radar_lat=40.5,
                radar_lon=-105.0,
                output_path=output_path,
                overview_factors=[]
            )
            
            assert result_path.exists()
            
            with rasterio.open(result_path) as src:
                assert len(src.overviews(1)) == 0
    
    def test_create_cog_invalid_overview_factors(self, sample_2d_data, sample_geometry):
        """Test that invalid overview_factors type raises TypeError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'test.cog'
            
            with pytest.raises(TypeError, match="must be a list"):
                create_cog(
                    sample_2d_data,
                    sample_geometry,
                    radar_lat=40.5,
                    radar_lon=-105.0,
                    output_path=output_path,
                    overview_factors="invalid"
                )


@pytest.mark.skipif(not RASTERIO_AVAILABLE, reason="rasterio not available")
class TestSaveProductAsGeoTIFF:
    """Test convenience function for saving products."""
    
    def test_save_as_geotiff(self, sample_2d_data, sample_geometry):
        """Test saving as standard GeoTIFF."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'product.tif'
            
            result_path = save_product_as_geotiff(
                sample_2d_data,
                sample_geometry,
                radar_lat=40.5,
                radar_lon=-105.0,
                output_path=output_path,
                as_cog=False
            )
            
            assert result_path.exists()
            
            with rasterio.open(result_path) as src:
                assert src.count == 4
    
    def test_save_as_cog(self, sample_2d_data, sample_geometry):
        """Test saving as COG."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'product.cog'
            
            result_path = save_product_as_geotiff(
                sample_2d_data,
                sample_geometry,
                radar_lat=40.5,
                radar_lon=-105.0,
                output_path=output_path,
                as_cog=True,
                overview_factors=[2, 4]
            )
            
            assert result_path.exists()
            
            with rasterio.open(result_path) as src:
                assert src.driver == 'COG'
                assert len(src.overviews(1)) > 0
    
    def test_save_with_custom_params(self, sample_2d_data, sample_geometry):
        """Test saving with custom colormap and value range."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'product.cog'
            
            result_path = save_product_as_geotiff(
                sample_2d_data,
                sample_geometry,
                radar_lat=40.5,
                radar_lon=-105.0,
                output_path=output_path,
                product_type='CAPPI',
                cmap='jet',
                vmin=0,
                vmax=70,
                projection='EPSG:3857',
                as_cog=True,
                overview_factors=[2, 4, 8],
                resampling_method='average'
            )
            
            assert result_path.exists()


# Integration test
@pytest.mark.skipif(not RASTERIO_AVAILABLE, reason="rasterio not available")
def test_full_workflow(sample_2d_data, sample_geometry):
    """Test complete workflow from data to COG."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Simulate a CAPPI product
        cappi_data = sample_2d_data
        
        # Save as COG
        output_path = Path(tmpdir) / 'cappi_3km.cog'
        result_path = save_product_as_geotiff(
            cappi_data,
            sample_geometry,
            radar_lat=40.5,
            radar_lon=-105.0,
            output_path=output_path,
            product_type='CAPPI',
            cmap='viridis',
            vmin=-10,
            vmax=70,
            projection='EPSG:3857',
            as_cog=True,
            overview_factors=[2, 4, 8, 16],
            resampling_method='nearest'
        )
        
        assert result_path.exists()
        
        # Verify the file is valid
        with rasterio.open(result_path) as src:
            assert src.driver == 'COG'
            assert src.count == 4
            assert src.width == 100
            assert src.height == 100
            assert src.crs.to_epsg() == 3857
            
            # Verify overviews
            overviews = src.overviews(1)
            assert len(overviews) == 4
            assert overviews == [2, 4, 8, 16]
            
            # Read and verify data
            data = src.read()
            assert data.shape == (4, 100, 100)
            assert data.dtype == np.uint8
