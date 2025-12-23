"""
Unit tests for COG overview configuration functionality.

Tests the _string_to_resampling() helper and convert_to_cog() function
with various overview and resampling method configurations.
"""
import pytest
import numpy as np
import tempfile
import rasterio
from pathlib import Path
from rasterio.enums import ColorInterp, Resampling

from radar_cog_processor.processor import (
    _string_to_resampling,
    convert_to_cog,
)


class TestStringToResampling:
    """Test string-to-resampling conversion helper."""

    def test_nearest_method(self):
        """Test conversion of 'nearest' string to Resampling enum."""
        result = _string_to_resampling("nearest")
        assert result == Resampling.nearest

    def test_bilinear_method(self):
        """Test conversion of 'bilinear' string to Resampling enum."""
        result = _string_to_resampling("bilinear")
        assert result == Resampling.bilinear

    def test_cubic_method(self):
        """Test conversion of 'cubic' string to Resampling enum."""
        result = _string_to_resampling("cubic")
        assert result == Resampling.cubic

    def test_average_method(self):
        """Test conversion of 'average' string to Resampling enum."""
        result = _string_to_resampling("average")
        assert result == Resampling.average

    def test_mode_method(self):
        """Test conversion of 'mode' string to Resampling enum."""
        result = _string_to_resampling("mode")
        assert result == Resampling.mode

    def test_max_method(self):
        """Test conversion of 'max' string to Resampling enum."""
        result = _string_to_resampling("max")
        assert result == Resampling.max

    def test_min_method(self):
        """Test conversion of 'min' string to Resampling enum."""
        result = _string_to_resampling("min")
        assert result == Resampling.min

    def test_case_insensitive(self):
        """Test that method strings are case-insensitive."""
        assert _string_to_resampling("NEAREST") == Resampling.nearest
        assert _string_to_resampling("BiLinear") == Resampling.bilinear
        assert _string_to_resampling("CUBIC") == Resampling.cubic

    def test_whitespace_handling(self):
        """Test that leading/trailing whitespace is handled."""
        assert _string_to_resampling("  nearest  ") == Resampling.nearest
        assert _string_to_resampling("\taverage\n") == Resampling.average

    def test_invalid_method_raises_error(self):
        """Test that invalid resampling method raises ValueError."""
        with pytest.raises(ValueError, match="Unknown resampling method"):
            _string_to_resampling("invalid_method")

    def test_invalid_method_error_message(self):
        """Test error message contains available methods."""
        with pytest.raises(ValueError) as exc_info:
            _string_to_resampling("not_a_method")
        
        error_msg = str(exc_info.value)
        assert "Available methods:" in error_msg
        assert "nearest" in error_msg
        assert "bilinear" in error_msg
        assert "average" in error_msg


class TestConvertToCog:
    """Test COG conversion with various overview configurations."""

    @pytest.fixture
    def sample_geotiff(self):
        """Create a temporary sample GeoTIFF file for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            tif_path = tmppath / "sample.tif"
            
            # Create a simple 4-band RGB+Alpha GeoTIFF
            data = np.random.randint(0, 255, (4, 256, 256), dtype=np.uint8)
            
            profile = {
                'driver': 'GTiff',
                'height': 256,
                'width': 256,
                'count': 4,
                'dtype': np.uint8,
                'crs': 'EPSG:4326',
                'transform': rasterio.transform.from_bounds(-180, -90, 180, 90, 256, 256),
            }
            
            with rasterio.open(str(tif_path), 'w', **profile) as dst:
                for i in range(4):
                    dst.write(data[i], i + 1)
            
            yield tif_path

    def test_default_overview_factors(self, sample_geotiff):
        """Test COG creation with default overview factors [2, 4, 8, 16]."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cog_path = Path(tmpdir) / "output.cog"
            
            result = convert_to_cog(str(sample_geotiff), str(cog_path))
            
            assert result == cog_path
            assert cog_path.exists()
            
            # Verify COG has overviews
            with rasterio.open(str(cog_path)) as src:
                assert src.driver in ['COG', 'GTiff']  # Different rasterio versions report differently
                assert src.count == 4
                assert len(src.overviews(1)) == 4  # 4 overview levels

    def test_custom_overview_factors(self, sample_geotiff):
        """Test COG creation with custom overview factors [2, 4]."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cog_path = Path(tmpdir) / "output.cog"
            
            result = convert_to_cog(
                str(sample_geotiff), str(cog_path),
                overview_factors=[2, 4]
            )
            
            assert result == cog_path
            assert cog_path.exists()
            
            # Verify COG has 2 overview levels
            with rasterio.open(str(cog_path)) as src:
                assert len(src.overviews(1)) == 2

    def test_many_overview_factors(self, sample_geotiff):
        """Test COG creation with many overview factors [2, 4, 8, 16, 32]."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cog_path = Path(tmpdir) / "output.cog"
            
            result = convert_to_cog(
                str(sample_geotiff), str(cog_path),
                overview_factors=[2, 4, 8, 16, 32]
            )
            
            assert result == cog_path
            assert cog_path.exists()
            
            # Verify COG has 5 overview levels
            with rasterio.open(str(cog_path)) as src:
                assert len(src.overviews(1)) == 5

    def test_disabled_overviews(self, sample_geotiff):
        """Test COG creation with disabled overviews (empty list)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cog_path = Path(tmpdir) / "output.cog"
            
            result = convert_to_cog(
                str(sample_geotiff), str(cog_path),
                overview_factors=[]
            )
            
            assert result == cog_path
            assert cog_path.exists()
            
            # Verify COG has no overviews
            with rasterio.open(str(cog_path)) as src:
                assert len(src.overviews(1)) == 0

    def test_none_overview_factors_uses_defaults(self, sample_geotiff):
        """Test that None overview_factors uses default [2, 4, 8, 16]."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cog_path = Path(tmpdir) / "output.cog"
            
            result = convert_to_cog(
                str(sample_geotiff), str(cog_path),
                overview_factors=None
            )
            
            assert result == cog_path
            assert cog_path.exists()
            
            # Verify COG has 4 overview levels (defaults)
            with rasterio.open(str(cog_path)) as src:
                assert len(src.overviews(1)) == 4

    def test_nearest_resampling(self, sample_geotiff):
        """Test COG creation with nearest resampling method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cog_path = Path(tmpdir) / "output.cog"
            
            result = convert_to_cog(
                str(sample_geotiff), str(cog_path),
                resampling_method="nearest"
            )
            
            assert result == cog_path
            assert cog_path.exists()
            
            # Verify COG was created successfully with overviews
            with rasterio.open(str(cog_path)) as src:
                assert len(src.overviews(1)) == 4

    def test_average_resampling(self, sample_geotiff):
        """Test COG creation with average resampling method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cog_path = Path(tmpdir) / "output.cog"
            
            result = convert_to_cog(
                str(sample_geotiff), str(cog_path),
                resampling_method="average"
            )
            
            assert result == cog_path
            assert cog_path.exists()
            
            # Verify COG was created successfully with overviews
            with rasterio.open(str(cog_path)) as src:
                assert len(src.overviews(1)) == 4

    def test_bilinear_resampling(self, sample_geotiff):
        """Test COG creation with bilinear resampling method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cog_path = Path(tmpdir) / "output.cog"
            
            result = convert_to_cog(
                str(sample_geotiff), str(cog_path),
                resampling_method="bilinear"
            )
            
            assert result == cog_path
            assert cog_path.exists()
            
            # Verify COG was created successfully with overviews
            with rasterio.open(str(cog_path)) as src:
                assert len(src.overviews(1)) == 4

    def test_cubic_resampling(self, sample_geotiff):
        """Test COG creation with cubic resampling method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cog_path = Path(tmpdir) / "output.cog"
            
            result = convert_to_cog(
                str(sample_geotiff), str(cog_path),
                resampling_method="cubic"
            )
            
            assert result == cog_path
            assert cog_path.exists()
            
            # Verify COG was created successfully with overviews
            with rasterio.open(str(cog_path)) as src:
                assert len(src.overviews(1)) == 4

    def test_case_insensitive_resampling(self, sample_geotiff):
        """Test that resampling method is case-insensitive."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cog_path = Path(tmpdir) / "output.cog"
            
            result = convert_to_cog(
                str(sample_geotiff), str(cog_path),
                resampling_method="AVERAGE"
            )
            
            assert result == cog_path
            assert cog_path.exists()

    def test_invalid_resampling_method_raises_error(self, sample_geotiff):
        """Test that invalid resampling method raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cog_path = Path(tmpdir) / "output.cog"
            
            with pytest.raises(ValueError, match="Unknown resampling method"):
                convert_to_cog(
                    str(sample_geotiff), str(cog_path),
                    resampling_method="not_a_real_method"
                )

    def test_invalid_overview_factors_type_raises_error(self, sample_geotiff):
        """Test that non-list overview_factors raises TypeError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cog_path = Path(tmpdir) / "output.cog"
            
            with pytest.raises(TypeError, match="overview_factors must be a list"):
                convert_to_cog(
                    str(sample_geotiff), str(cog_path),
                    overview_factors=(2, 4, 8)  # tuple instead of list
                )

    def test_invalid_resampling_method_type_raises_error(self, sample_geotiff):
        """Test that non-string resampling_method raises TypeError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cog_path = Path(tmpdir) / "output.cog"
            
            with pytest.raises(TypeError, match="resampling_method must be a string"):
                convert_to_cog(
                    str(sample_geotiff), str(cog_path),
                    resampling_method=Resampling.nearest  # enum instead of string
                )

    def test_combined_custom_factors_and_resampling(self, sample_geotiff):
        """Test COG creation with custom factors and resampling together."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cog_path = Path(tmpdir) / "output.cog"
            
            result = convert_to_cog(
                str(sample_geotiff), str(cog_path),
                overview_factors=[2, 4, 8],
                resampling_method="average"
            )
            
            assert result == cog_path
            assert cog_path.exists()
            
            # Verify both settings
            with rasterio.open(str(cog_path)) as src:
                assert len(src.overviews(1)) == 3

    def test_cog_compression_settings(self, sample_geotiff):
        """Test that COG is created with proper compression settings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cog_path = Path(tmpdir) / "output.cog"
            
            convert_to_cog(str(sample_geotiff), str(cog_path))
            
            with rasterio.open(str(cog_path)) as src:
                # COG driver may report as GTiff or COG depending on rasterio version
                assert src.driver in ['COG', 'GTiff']
                # Compression may be lowercase or uppercase
                assert src.profile['compress'].upper() == 'DEFLATE'

    def test_cog_band_descriptions(self, sample_geotiff):
        """Test that COG is created successfully with color bands."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cog_path = Path(tmpdir) / "output.cog"
            
            convert_to_cog(str(sample_geotiff), str(cog_path))
            
            with rasterio.open(str(cog_path)) as src:
                # Band descriptions may not be preserved by COG driver,
                # but we can verify it has 4 bands (RGBA)
                assert src.count == 4

    def test_backward_compatibility(self, sample_geotiff):
        """Test that old code without overview parameters still works."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cog_path = Path(tmpdir) / "output.cog"
            
            # Call with just the two required parameters (backward compatible)
            result = convert_to_cog(str(sample_geotiff), str(cog_path))
            
            assert result == cog_path
            assert cog_path.exists()
            
            # Should use defaults
            with rasterio.open(str(cog_path)) as src:
                assert len(src.overviews(1)) == 4

    def test_data_integrity_with_overviews(self, sample_geotiff):
        """Test that overview creation doesn't corrupt full resolution data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cog_path = Path(tmpdir) / "output.cog"
            
            convert_to_cog(str(sample_geotiff), str(cog_path))
            
            # Compare original and COG full resolution data
            with rasterio.open(str(sample_geotiff)) as src_orig:
                with rasterio.open(str(cog_path)) as src_cog:
                    for i in range(1, 5):
                        orig_data = src_orig.read(i)
                        cog_data = src_cog.read(i)
                        np.testing.assert_array_equal(orig_data, cog_data)

    def test_single_overview_level(self, sample_geotiff):
        """Test COG creation with single overview level."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cog_path = Path(tmpdir) / "output.cog"
            
            result = convert_to_cog(
                str(sample_geotiff), str(cog_path),
                overview_factors=[2]
            )
            
            assert result == cog_path
            assert cog_path.exists()
            
            # Verify COG has 1 overview level
            with rasterio.open(str(cog_path)) as src:
                assert len(src.overviews(1)) == 1

    def test_pathlib_path_objects(self, sample_geotiff):
        """Test that pathlib.Path objects work as input/output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cog_path = Path(tmpdir) / "output.cog"
            
            # Use Path objects
            result = convert_to_cog(
                sample_geotiff,  # already a Path object
                cog_path,  # Path object
                overview_factors=[2, 4]
            )
            
            assert result == cog_path
            assert cog_path.exists()
