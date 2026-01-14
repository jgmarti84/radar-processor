#!/usr/bin/env python
"""
Standalone test script to verify GeoTIFF projection differences.

This script creates two test GeoTIFFs with different projections and
compares their metadata to verify they are actually different.
"""

import numpy as np
from pathlib import Path
import sys

# Add the src directory to path
sys.path.insert(0, '/home/runner/work/radar-processor/radar-processor/src')

from radar_grid.geometry import GridGeometry
from radar_grid.geotiff import create_geotiff, create_cog

# Create test data and geometry
print("=" * 80)
print("GEOTIFF PROJECTION VERIFICATION TEST")
print("=" * 80)

nz, ny, nx = 10, 100, 100
grid_shape = (nz, ny, nx)
grid_limits = (
    (0.0, 10000.0),       # z: 0 to 10 km
    (-50000.0, 50000.0),  # y: -50 to 50 km
    (-50000.0, 50000.0)   # x: -50 to 50 km
)

n_points = nz * ny * nx
geometry = GridGeometry(
    grid_shape=grid_shape,
    grid_limits=grid_limits,
    indptr=np.arange(n_points + 1, dtype=np.int32),
    gate_indices=np.zeros(n_points, dtype=np.int32),
    weights=np.ones(n_points, dtype=np.float32),
    toa=12000.0,
    radar_altitude=100.0
)

# Create sample 2D data (gradient pattern)
data = np.random.rand(ny, nx) * 50

# Test radar location
radar_lat = 40.5
radar_lon = -105.0

output_dir = Path('/tmp/geotiff_test')
output_dir.mkdir(exist_ok=True)

print(f"\nTest Configuration:")
print(f"  Radar location: {radar_lat}°N, {radar_lon}°W")
print(f"  Grid size: {ny} x {nx} pixels")
print(f"  Grid extent: ±50 km from radar")
print(f"  Output directory: {output_dir}")

# Test 1: Create Web Mercator GeoTIFF
print("\n" + "=" * 80)
print("TEST 1: Creating Web Mercator (EPSG:3857) GeoTIFF")
print("=" * 80)

output_webmerc = output_dir / 'test_webmercator.tif'
try:
    create_geotiff(
        data, geometry, radar_lat, radar_lon,
        output_webmerc,
        cmap='viridis',
        vmin=0, vmax=50,
        projection='EPSG:3857'
    )
    print(f"\n✓ Created: {output_webmerc}")
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Create WGS84 GeoTIFF
print("\n" + "=" * 80)
print("TEST 2: Creating WGS84 (EPSG:4326) GeoTIFF")
print("=" * 80)

output_wgs84 = output_dir / 'test_wgs84.tif'
try:
    create_geotiff(
        data, geometry, radar_lat, radar_lon,
        output_wgs84,
        cmap='viridis',
        vmin=0, vmax=50,
        projection='EPSG:4326'
    )
    print(f"\n✓ Created: {output_wgs84}")
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Compare metadata
print("\n" + "=" * 80)
print("METADATA COMPARISON")
print("=" * 80)

try:
    import rasterio
    
    print("\n--- Web Mercator (EPSG:3857) ---")
    with rasterio.open(output_webmerc) as src:
        print(f"CRS: {src.crs}")
        print(f"EPSG: {src.crs.to_epsg()}")
        print(f"Bounds: {src.bounds}")
        print(f"  West:  {src.bounds.left:>15.2f}")
        print(f"  South: {src.bounds.bottom:>15.2f}")
        print(f"  East:  {src.bounds.right:>15.2f}")
        print(f"  North: {src.bounds.top:>15.2f}")
        print(f"Transform: {src.transform}")
        webmerc_crs = src.crs
        webmerc_bounds = src.bounds
    
    print("\n--- WGS84 (EPSG:4326) ---")
    with rasterio.open(output_wgs84) as src:
        print(f"CRS: {src.crs}")
        print(f"EPSG: {src.crs.to_epsg()}")
        print(f"Bounds: {src.bounds}")
        print(f"  West:  {src.bounds.left:>15.6f}")
        print(f"  South: {src.bounds.bottom:>15.6f}")
        print(f"  East:  {src.bounds.right:>15.6f}")
        print(f"  North: {src.bounds.top:>15.6f}")
        print(f"Transform: {src.transform}")
        wgs84_crs = src.crs
        wgs84_bounds = src.bounds
    
    # Analysis
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    
    if webmerc_crs == wgs84_crs:
        print("\n⚠ PROBLEM: Both files have the SAME CRS!")
        print(f"   Both: {webmerc_crs}")
        print("   This indicates a BUG in the projection logic.")
    else:
        print("\n✓ GOOD: Files have different CRS")
        print(f"   Web Mercator: {webmerc_crs}")
        print(f"   WGS84: {wgs84_crs}")
    
    # Check bounds
    bounds_similar = (
        abs(webmerc_bounds.left - wgs84_bounds.left) < 100 and
        abs(webmerc_bounds.right - wgs84_bounds.right) < 100
    )
    
    if bounds_similar:
        print("\n⚠ PROBLEM: Bounds are suspiciously similar!")
        print("   This might indicate the projection transformation didn't work.")
    else:
        print("\n✓ GOOD: Bounds are different")
        print("   Web Mercator bounds should be in meters (millions)")
        print("   WGS84 bounds should be in degrees (~-105, ~40)")
        
        # Verify magnitude
        webmerc_magnitude = max(abs(webmerc_bounds.left), abs(webmerc_bounds.right))
        wgs84_magnitude = max(abs(wgs84_bounds.left), abs(wgs84_bounds.right))
        
        print(f"\n   Web Mercator magnitude: {webmerc_magnitude:,.0f}")
        print(f"   WGS84 magnitude: {wgs84_magnitude:.1f}")
        
        if webmerc_magnitude > 1_000_000 and wgs84_magnitude < 200:
            print("   ✓ Magnitudes are correct!")
        else:
            print("   ⚠ Unexpected magnitudes")
    
    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
    print(f"\nTest files created in: {output_dir}")
    print("You can open these files in QGIS or another GIS viewer to verify")
    print("they display in different locations if projections are wrong.")

except ImportError:
    print("\n⚠ rasterio not available for detailed comparison")
    print("The files were created but cannot be validated.")

except Exception as e:
    print(f"\n✗ Error during comparison: {e}")
    import traceback
    traceback.print_exc()
