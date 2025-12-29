"""
Custom colormap definitions for radar data visualization.

This file has been synchronized with radarlib's colormap definitions
to ensure PNG and COG outputs use identical color scales.

Source: radarlib/src/radarlib/colormaps.py
"""
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, to_rgb


def get_cmap_grc_vrad():
    """Get custom colormap for VRAD (radial velocity) visualization."""
    grc_vrad_data = {
        'red': [(0.0, 0, 0),
                (0.45, 0, 0),
                (0.5, 0.05, 0.05),
                (0.55, 0.5, 0.5),
                (1, 1, 1)],
        'green': [(0.0, 1, 1),
                  (0.45, 0.5, 0.5),
                  (0.5, 0.05, 0.05),
                  (0.55, 0, 0),
                  (1, 0, 0)],
        'blue': [(0.0, 0, 0),
                 (0.45, 0, 0),
                 (0.5, 0.05, 0.05),
                 (0.55, 0, 0),
                 (1, 0, 0)],
    }
    return LinearSegmentedColormap('grc_vrad', grc_vrad_data)


def get_cmap_grc_rho():
    """Get custom colormap for RHOHV (correlation coefficient) visualization."""
    grc_rho_data = {
        'red': [
            (0.0, 0.38, 0.38),
            (0.3, 0.16, 0.16),
            (0.5, 0.27, 0.27),
            (0.7, 0.31, 0.31),
            (0.8, 0.49, 0.49),
            (0.85, 0, 0),
            (0.9, 1, 1),
            (0.95, 1, 1),
            (0.98, 0.67, 0.67),
            (0.995, 0.67, 0.67),
            (1, 1, 1),
        ],
        'green': [
            (0.0, 0, 0),
            (0.3, 0.28, 0.28),
            (0.5, 0.55, 0.55),
            (0.7, 0.78, 0.78),
            (0.8, 0.89, 0.89),
            (0.85, 0.81, 0.81),
            (0.9, 1, 1),
            (0.95, 0.55, 0.55),
            (0.98, 0, 0),
            (0.995, 0, 0),
            (1, 0, 0),
        ],
        'blue': [
            (0.0, 0.57, 0.57),
            (0.3, 0.97, 0.97),
            (0.5, 0.83, 0.83),
            (0.7, 1, 1),
            (0.8, 1, 1),
            (0.85, 0, 0),
            (0.9, 0, 0),
            (0.95, 0.45, 0.45),
            (0.98, 0, 0),
            (0.995, 0, 0),
            (1, 1, 1),
        ],
    }
    return LinearSegmentedColormap('grc_rho', grc_rho_data)


def get_cmap_grc_th():
    """Get custom colormap for reflectivity (TH) visualization."""
    grc_th = {
        'red': [
            (0.0, 1, 1),
            (0.33, 0.95, 0.95),
            (0.4, 0.24, 0.24),
            (0.45, 0.22, 0.22),
            (0.55, 0.04, 0.04),
            (0.63, 0.95, 0.95),
            (0.85, 1, 1),
            (1, 1, 1),
        ],
        'green': [
            (0.0, 1, 1),
            (0.33, 0.97, 0.97),
            (0.4, 0.46, 0.46),
            (0.4, 0.98, 0.98),
            (0.55, 0.62, 0.62),
            (0.63, 1, 1),
            (0.85, 0, 0),
            (1, 0, 0),
        ],
        'blue': [
            (0.0, 1, 1),
            (0.33, 0.95, 0.95),
            (0.4, 0.78, 0.78),
            (0.4, 0.52, 0.52),
            (0.55, 0.27, 0.27),
            (0.63, 0, 0),
            (0.85, 0, 0),
            (1, 1, 1),
        ],
    }
    return LinearSegmentedColormap('grc_th', grc_th)


def get_cmap_grc_th2():
    """Get alternative custom colormap for reflectivity visualization."""
    grc_th2 = {
        'red': [
            (0.0, 1, 1),
            (0.25, 0.95, 0.95),
            (0.5, 0.24, 0.24),
            (0.5, 0.22, 0.22),
            (0.55, 0.04, 0.04),
            (0.63, 0.95, 0.95),
            (0.85, 1, 1),
            (1, 1, 1),
        ],
        'green': [
            (0.0, 1, 1),
            (0.25, 0.97, 0.97),
            (0.5, 0.46, 0.46),
            (0.5, 0.98, 0.98),
            (0.55, 0.62, 0.62),
            (0.63, 1, 1),
            (0.85, 0, 0),
            (1, 0, 0),
        ],
        'blue': [
            (0.0, 1, 1),
            (0.25, 0.95, 0.95),
            (0.5, 0.78, 0.78),
            (0.5, 0.52, 0.52),
            (0.55, 0.27, 0.27),
            (0.63, 0, 0),
            (0.85, 0, 0),
            (1, 1, 1),
        ],
    }
    return LinearSegmentedColormap('grc_th2', grc_th2)


def get_cmap_grc_zdr():
    """Get custom colormap for ZDR (differential reflectivity) visualization.
    
    Uses LinearSegmentedColormap.from_list to interpolate between 7 key colors.
    This ensures smooth color transitions for continuous ZDR data.
    """
    # 7 key colors from radarlib - will be interpolated for smooth gradients
    grc_zdr_colors = ["#b7b7b7", "#0055FF", "#66b3df", "#00FFFF", "#489D39", "#F9EA3C", "#FF078B"]
    return LinearSegmentedColormap.from_list('grc_zdr', grc_zdr_colors, N=256)


def get_cmap_grc_zdr2():
    """Get alternative custom colormap for ZDR visualization with smooth transitions."""
    # This is a smoothed version with more colors for gradual transitions
    hex_colors = [
        '#2c2c2c', '#8a8a8a', '#e6e6e6', '#00FFFF',
        '#94CDFF', '#0055FF', '#489D39', '#F9EA3C',
        '#FF8345', '#FF212C', '#FF078B'
    ]
    
    # Convert hex to RGB (0-1 range)
    rgb_colors = [to_rgb(c) for c in hex_colors]
    n = len(rgb_colors)
    
    # Define equidistant positions
    positions = [i / (n - 1) for i in range(n)]
    
    grc_zdr_data = {
        'red': [(positions[i], rgb_colors[i][0], rgb_colors[i][0]) for i in range(n)],
        'green': [(positions[i], rgb_colors[i][1], rgb_colors[i][1]) for i in range(n)],
        'blue': [(positions[i], rgb_colors[i][2], rgb_colors[i][2]) for i in range(n)],
    }
    
    return LinearSegmentedColormap('grc_zdr2', grc_zdr_data)


def get_cmap_grc_rain():
    """Get custom colormap for rain rate visualization."""
    # Based on radarlib's _th colormap structure
    cmp_rain = {
        'red': [
            (0.0, 1, 1),
            (0.33, 0.95, 0.95),
            (0.4, 0.24, 0.24),
            (0.45, 0.22, 0.22),
            (0.55, 0.04, 0.04),
            (0.63, 0.95, 0.95),
            (0.85, 1, 1),
            (1, 1, 1),
        ],
        'green': [
            (0.0, 1, 1),
            (0.33, 0.97, 0.97),
            (0.4, 0.46, 0.46),
            (0.4, 0.98, 0.98),
            (0.55, 0.62, 0.62),
            (0.63, 1, 1),
            (0.85, 0, 0),
            (1, 0, 0),
        ],
        'blue': [
            (0.0, 1, 1),
            (0.33, 0.95, 0.95),
            (0.4, 0.78, 0.78),
            (0.4, 0.52, 0.52),
            (0.55, 0.27, 0.27),
            (0.63, 0, 0),
            (0.85, 0, 0),
            (1, 1, 1),
        ],
    }
    return LinearSegmentedColormap('grc_rain', cmp_rain)
