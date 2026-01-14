"""
radar_grid - Fast radar gridding with precomputed geometry
"""

from .geometry import GridGeometry, save_geometry, load_geometry
from .compute import compute_grid_geometry
from .interpolate import apply_geometry, apply_geometry_multi
from .utils import get_gate_coordinates, get_field_data, get_available_fields, get_radar_info
from .mpl_visualization import (
    plot_grid_slice,
    plot_grid_multi_level,
    plot_all_fields,
    plot_vertical_cross_section,
    FIELD_CONFIGS,
)
from .filters import GateFilter, create_mask_from_filter
from .products import (
    constant_altitude_ppi,
    constant_elevation_ppi,
    column_max,
    column_min,
    column_mean,
    get_elevation_from_z_level,
    get_beam_height_difference,
    compute_beam_height,
    compute_beam_height_flat,
    EARTH_RADIUS,
    EFFECTIVE_RADIUS_FACTOR,
)
from .geotiff import (
    create_geotiff,
    create_cog,
    save_product_as_geotiff,
    apply_colormap_to_array,
)
# from .products import (
#     constant_elevation_ppi,
#     column_max,
#     column_min,
#     column_mean,
#     get_elevation_from_z_level,
# )

__version__ = "0.1.0"

__all__ = [
    # Core classes
    "GridGeometry",
    # Geometry I/O
    "save_geometry",
    "load_geometry",
    # Computation
    "compute_grid_geometry",
    # Interpolation
    "apply_geometry",
    "apply_geometry_multi",
    # Utilities
    "get_gate_coordinates",
    "get_field_data",
    "get_available_fields",
    "get_radar_info",
    # Visualization
    "plot_grid_slice",
    "plot_grid_multi_level",
    "plot_all_fields",
    "plot_vertical_cross_section",
    "FIELD_CONFIGS",
    # Filters
    "GateFilter",
    "create_mask_from_filter",
    # Products
    "constant_altitude_ppi",
    "constant_elevation_ppi",
    "column_max",
    "column_min",
    "column_mean",
    "get_elevation_from_z_level",
    "get_beam_height_difference",
    "compute_beam_height",
    "compute_beam_height_flat",
    "EARTH_RADIUS",
    "EFFECTIVE_RADIUS_FACTOR",
    # GeoTIFF generation
    "create_geotiff",
    "create_cog",
    "save_product_as_geotiff",
    "apply_colormap_to_array",
]
# """
# radar_grid - Fast radar gridding with precomputed geometry
# """

# from .geometry import GridGeometry, save_geometry, load_geometry
# from .compute import compute_grid_geometry
# from .interpolate import apply_geometry, apply_geometry_multi
# from .utils import get_gate_coordinates, get_field_data, get_available_fields, get_radar_info
# from .mpl_visualization import (
#     plot_grid_slice,
#     plot_grid_multi_level,
#     plot_all_fields,
#     plot_vertical_cross_section,
#     FIELD_CONFIGS,
# )
# from .filters import GateFilter, create_mask_from_filter

# __version__ = "0.1.0"

# __all__ = [
#     # Core classes
#     "GridGeometry",
#     # Geometry I/O
#     "save_geometry",
#     "load_geometry",
#     # Computation
#     "compute_grid_geometry",
#     # Interpolation
#     "apply_geometry",
#     "apply_geometry_multi",
#     # Utilities
#     "get_gate_coordinates",
#     "get_field_data",
#     "get_available_fields",
#     "get_radar_info",
#     # Visualization
#     "plot_grid_slice",
#     "plot_grid_multi_level",
#     "plot_all_fields",
#     "plot_vertical_cross_section",
#     "FIELD_CONFIGS",
#     # Filters
#     "GateFilter",
#     "create_mask_from_filter",
# ]