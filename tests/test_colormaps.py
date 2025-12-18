"""
Unit tests for radar_cog_processor.colormaps module.
"""
import pytest
from matplotlib.colors import LinearSegmentedColormap
from radar_cog_processor.colormaps import (
    get_cmap_grc_rain,
    get_cmap_grc_th,
    get_cmap_grc_th2,
    get_cmap_grc_rho,
    get_cmap_grc_zdr,
    get_cmap_grc_zdr2,
    get_cmap_grc_vrad,
)


def test_get_cmap_grc_rain():
    """Test rain colormap generation."""
    cmap = get_cmap_grc_rain()
    assert isinstance(cmap, LinearSegmentedColormap)
    assert cmap.name == 'grc_rain'
    # Test that colormap can map values
    rgba = cmap(0.5)
    assert len(rgba) == 4  # RGBA


def test_get_cmap_grc_th():
    """Test reflectivity colormap generation."""
    cmap = get_cmap_grc_th()
    assert isinstance(cmap, LinearSegmentedColormap)
    assert cmap.name == 'grc_th'
    # Test colormap range
    rgba_min = cmap(0.0)
    rgba_max = cmap(1.0)
    assert rgba_min != rgba_max


def test_get_cmap_grc_th2():
    """Test alternative reflectivity colormap."""
    cmap = get_cmap_grc_th2()
    assert isinstance(cmap, LinearSegmentedColormap)
    assert cmap.name == 'grc_th2'


def test_get_cmap_grc_rho():
    """Test RHOHV correlation colormap."""
    cmap = get_cmap_grc_rho()
    assert isinstance(cmap, LinearSegmentedColormap)
    assert cmap.name == 'grc_rho'


def test_get_cmap_grc_zdr():
    """Test ZDR differential reflectivity colormap."""
    cmap = get_cmap_grc_zdr()
    assert isinstance(cmap, LinearSegmentedColormap)
    assert cmap.name == 'grc_zdr'


def test_get_cmap_grc_zdr2():
    """Test alternative ZDR colormap with smooth transitions."""
    cmap = get_cmap_grc_zdr2()
    assert isinstance(cmap, LinearSegmentedColormap)
    assert cmap.name == 'grc_zdr2'


def test_get_cmap_grc_vrad():
    """Test radial velocity colormap."""
    cmap = get_cmap_grc_vrad()
    assert isinstance(cmap, LinearSegmentedColormap)
    assert cmap.name == 'grc_vrad'


def test_all_colormaps_callable():
    """Test that all colormap functions are callable and return colormaps."""
    colormap_funcs = [
        get_cmap_grc_rain,
        get_cmap_grc_th,
        get_cmap_grc_th2,
        get_cmap_grc_rho,
        get_cmap_grc_zdr,
        get_cmap_grc_zdr2,
        get_cmap_grc_vrad,
    ]
    
    for func in colormap_funcs:
        cmap = func()
        assert isinstance(cmap, LinearSegmentedColormap)
        # Test that colormap can process array of values
        import numpy as np
        values = np.linspace(0, 1, 100)
        colors = cmap(values)
        assert colors.shape == (100, 4)  # RGBA
