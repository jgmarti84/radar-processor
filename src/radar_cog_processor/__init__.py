"""
Radar COG Processor
===================

A Python package for processing meteorological radar NetCDF files into Cloud-Optimized GeoTIFFs (COG).

Main components:
- processor: Core processing functions for radar data to COG conversion
- constants: Field definitions, colormaps, and rendering parameters
- cache: LRU cache implementation for 2D and 3D grids
- utils: Helper functions for colormap generation and common operations
"""

__version__ = "0.1.0"

from .processor import process_radar_to_cog
from .constants import FIELD_ALIASES, FIELD_RENDER, FIELD_COLORMAP_OPTIONS
from .cache import GRID2D_CACHE, GRID3D_CACHE

__all__ = [
    "process_radar_to_cog",
    "FIELD_ALIASES",
    "FIELD_RENDER",
    "FIELD_COLORMAP_OPTIONS",
    "GRID2D_CACHE",
    "GRID3D_CACHE",
]
