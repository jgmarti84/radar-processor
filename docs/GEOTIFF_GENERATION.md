# GeoTIFF Generation for radar_grid Products

## Overview

The `radar_grid` module now includes comprehensive support for generating georeferenced GeoTIFF and Cloud-Optimized GeoTIFF (COG) files from 2D radar products such as CAPPI, PPI, and COLMAX.

## Features

- **GeoTIFF Creation**: Convert 2D radar products to georeferenced GeoTIFF images
- **Cloud-Optimized GeoTIFF (COG)**: Generate COG files with pyramid overviews for efficient web serving
- **Colormap Application**: Apply matplotlib colormaps with custom vmin/vmax ranges
- **Projection Support**: Support for Web Mercator (EPSG:3857), WGS84 (EPSG:4326), and other projections
- **Coordinate Transformation**: Automatic conversion from radar-relative Cartesian coordinates to geographic coordinates
- **Transparent No-Data**: Proper handling of NaN/no-data values with alpha channel transparency

## Installation

The GeoTIFF functionality requires the following dependencies:
- `rasterio` >= 1.3.0
- `pyproj` >= 3.0.0
- `matplotlib` >= 3.5.0
- `numpy` >= 1.20.0

These are already included in the radar-processor requirements.

## Quick Start

### Basic Usage

```python
from radar_grid import (
    load_geometry,
    get_field_data,
    apply_geometry,
    constant_altitude_ppi,
    save_product_as_geotiff
)
import pyart

# Load radar data
radar = pyart.io.read('radar_file.nc')
geometry = load_geometry('geometry.npz')

# Get radar location
radar_lat = radar.latitude['data'][0]
radar_lon = radar.longitude['data'][0]

# Generate CAPPI at 3000m
dbzh_data = get_field_data(radar, 'DBZH')
grid = apply_geometry(geometry, dbzh_data)
cappi = constant_altitude_ppi(grid, geometry, altitude=3000.0)

# Save as COG
save_product_as_geotiff(
    cappi,
    geometry,
    radar_lat,
    radar_lon,
    output_path='cappi_3km.cog',
    cmap='viridis',
    vmin=-10,
    vmax=70,
    as_cog=True
)
```

### Advanced Usage

#### Custom Colormaps and Projections

```python
from radar_grid import create_cog, column_max

# Generate COLMAX
colmax = column_max(grid)

# Save with custom settings
create_cog(
    colmax,
    geometry,
    radar_lat=40.5,
    radar_lon=-105.0,
    output_path='colmax.cog',
    cmap='jet',                    # Custom colormap
    vmin=0,
    vmax=70,
    projection='EPSG:3857',        # Web Mercator
    overview_factors=[2, 4, 8, 16], # Pyramid levels
    resampling_method='average'    # Resampling for overviews
)
```

#### PPI at Constant Elevation

```python
from radar_grid import constant_elevation_ppi, create_geotiff

# Generate PPI at 2.0° elevation
ppi = constant_elevation_ppi(grid, geometry, elevation_angle=2.0)

# Save as standard GeoTIFF (not COG)
create_geotiff(
    ppi,
    geometry,
    radar_lat=40.5,
    radar_lon=-105.0,
    output_path='ppi_2deg.tif',
    cmap='viridis',
    vmin=-10,
    vmax=70,
    projection='EPSG:4326'  # WGS84
)
```

## API Reference

### Main Functions

#### `save_product_as_geotiff()`
Convenience function to save any radar product as GeoTIFF or COG.

**Parameters:**
- `product_data` (np.ndarray): 2D array of product data
- `geometry` (GridGeometry): Grid geometry object
- `radar_lat` (float): Radar latitude in degrees
- `radar_lon` (float): Radar longitude in degrees
- `output_path` (str/Path): Output file path
- `product_type` (str): Product type name (default: 'CAPPI')
- `cmap` (str/Colormap): Colormap to apply (default: 'viridis')
- `vmin` (float): Minimum value for colormap scaling
- `vmax` (float): Maximum value for colormap scaling
- `projection` (str): Target projection (default: 'EPSG:3857')
- `as_cog` (bool): Create COG if True (default: True)
- `overview_factors` (list): Overview levels for COG (default: [2, 4, 8, 16])
- `resampling_method` (str): Resampling method (default: 'nearest')

#### `create_cog()`
Create a Cloud-Optimized GeoTIFF with overviews.

**Parameters:** Similar to `save_product_as_geotiff()` but always creates COG.

#### `create_geotiff()`
Create a standard GeoTIFF (without COG optimization).

**Parameters:** Similar to `save_product_as_geotiff()` but without overview options.

#### `apply_colormap_to_array()`
Apply a colormap to convert data values to RGBA.

**Parameters:**
- `data` (np.ndarray): 2D data array
- `cmap` (str/Colormap): Colormap to apply
- `vmin` (float): Minimum value for normalization
- `vmax` (float): Maximum value for normalization
- `fill_value` (float): Value to treat as no-data

**Returns:** RGBA array (ny, nx, 4) with dtype uint8

## Supported Projections

The implementation supports any projection that can be specified as:
- EPSG code (e.g., 'EPSG:3857', 'EPSG:4326')
- PROJ4 string

Common projections:
- **EPSG:3857** - Web Mercator (default, optimal for web mapping)
- **EPSG:4326** - WGS84 geographic coordinates
- **EPSG:32633** - UTM Zone 33N (for specific regions)

## Colormap Options

Any matplotlib colormap can be used. Common radar colormaps:
- `'viridis'` - Default, perceptually uniform
- `'jet'` - Classic rainbow colormap
- `'pyart_NWSRef'` - PyART NWS reflectivity colormap (if available)
- `'plasma'`, `'inferno'`, `'magma'` - Perceptually uniform options

## Overview Levels (COG)

Overview factors determine the pyramid levels for efficient multi-scale display:
- `[2, 4, 8, 16]` - Default, 4 levels (recommended for most use cases)
- `[2, 4]` - Fewer levels for faster generation
- `[2, 4, 8, 16, 32]` - More levels for deep zoom
- `[]` - No overviews (faster generation, larger file access times)

## Resampling Methods

For COG overviews, several resampling methods are available:
- `'nearest'` - Fastest, preserves exact values (default)
- `'average'` - Averages pixel values (good for radar intensity)
- `'bilinear'` - Smooth interpolation
- `'cubic'` - High quality interpolation
- `'mode'` - Most frequent value (good for categorical data)
- `'max'` - Maximum value (good for preserving reflectivity peaks)
- `'min'` - Minimum value

## Coordinate System Details

The radar_grid module uses a Cartesian coordinate system with:
- Origin at radar location
- X-axis pointing East
- Y-axis pointing North
- Z-axis pointing up (altitude)
- Units in meters

The GeoTIFF functions automatically:
1. Convert from radar-relative Cartesian to geographic (lat/lon)
2. Transform to the target projection (e.g., Web Mercator)
3. Generate proper geotransform and CRS metadata

## Examples

See the `examples/geotiff_generation_example.py` file for comprehensive examples including:
1. Basic CAPPI to GeoTIFF
2. CAPPI as Cloud-Optimized GeoTIFF
3. PPI to GeoTIFF
4. COLMAX to GeoTIFF
5. WGS84 projection usage
6. Multiple altitudes batch processing

## Performance Considerations

### COG vs Standard GeoTIFF
- **COG**: Slower to create (~100-300ms extra for overviews) but much faster for web serving and partial reads
- **Standard GeoTIFF**: Faster to create but slower for web applications

### Resampling Methods
- **nearest**: Fastest (~100-150ms for standard overviews)
- **average**: Slower (~200-300ms) but better preserves radar data characteristics
- **cubic**: Slowest but highest quality

### File Sizes
- Standard GeoTIFF: Depends on data and compression
- COG with overviews: ~33% larger (overviews add ~1/3 of original size)
- Compression reduces both significantly (DEFLATE used by default)

## Testing

Run the test suite:
```bash
pytest tests/test_geotiff_generation.py -v
```

Tests cover:
- Colormap application with various parameters
- GeoTIFF creation with different projections
- COG creation with custom overviews
- Shape validation and error handling
- Complete workflow integration

## Troubleshooting

### Import Errors
Ensure all dependencies are installed:
```bash
pip install rasterio pyproj matplotlib numpy
```

### Projection Errors
If projection transformation fails, verify:
- Radar location (lat/lon) is correct
- Target projection is valid EPSG code or PROJ4 string
- PyProj is properly installed

### Memory Issues
For very large grids:
- Reduce overview factors
- Use 'nearest' resampling method
- Process in batches if generating multiple products

## Future Enhancements

Potential additions:
- Support for additional metadata in GeoTIFF tags
- Batch processing utilities
- Custom color scales and legends
- Direct integration with PyART colormaps
- Parallel processing for multiple products

## License

This functionality is part of the radar-processor package and follows the same MIT license.
