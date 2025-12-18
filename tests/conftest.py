"""
Pytest configuration and fixtures.
"""
import pytest
import numpy as np
from pathlib import Path


@pytest.fixture
def sample_radar_data():
    """Create sample radar data for testing."""
    # This would normally be a real radar file, but for testing we create synthetic data
    data = {
        "reflectivity": np.random.rand(360, 500) * 70 - 30,  # -30 to 40 dBZ
        "velocity": np.random.rand(360, 500) * 70 - 35,  # -35 to 35 m/s
        "correlation": np.random.rand(360, 500) * 0.5 + 0.5,  # 0.5 to 1.0
    }
    return data


@pytest.fixture
def output_dir(tmp_path):
    """Create temporary output directory."""
    output = tmp_path / "output"
    output.mkdir()
    return output


@pytest.fixture
def sample_netcdf_file(tmp_path):
    """Create a minimal synthetic radar NetCDF file for testing."""
    # Note: This is a placeholder. In real tests, you'd need actual radar files
    # or use pyart to create synthetic radar objects
    return None  # Placeholder for actual test data


@pytest.fixture(autouse=True)
def clear_caches():
    """Clear caches before and after each test."""
    from radar_cog_processor.cache import GRID2D_CACHE, GRID3D_CACHE
    GRID2D_CACHE.clear()
    GRID3D_CACHE.clear()
    yield
    GRID2D_CACHE.clear()
    GRID3D_CACHE.clear()
