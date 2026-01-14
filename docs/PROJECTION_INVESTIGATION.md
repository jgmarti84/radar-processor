# GeoTIFF Projection Investigation

## Issue Report

User reports: "I don't see any change in the generated COG file between the web mercator projection and the WGS84 projection."

## Investigation Summary

### What's Expected vs What User Might See

#### Expected Behavior (CORRECT):
1. **Pixel data (RGBA image)**: IDENTICAL in both Web Mercator and WGS84 files
2. **Metadata (CRS + geotransform)**: DIFFERENT between the two files
   - Web Mercator: CRS = EPSG:3857, bounds in meters (e.g., -11,688,000 to -11,588,000)
   - WGS84: CRS = EPSG:4326, bounds in degrees (e.g., -105.5 to -104.5)

#### What User Might Be Seeing (needs verification):
- If opening files in an image viewer: Files look IDENTICAL (expected - same pixels)
- If opening in GIS without proper projection: Might appear in same location (would be bug)
- If metadata shows same CRS/bounds: DEFINITE BUG

## How GeoTIFF Projection Works

### Important Concept
A GeoTIFF contains:
1. **Pixel data**: The actual image (RGBA values)
2. **Georeferencing**: Metadata telling how to position the image in the world
   - CRS (Coordinate Reference System)
   - Geotransform (maps pixels to coordinates)

When we change the projection, we're changing #2 (metadata), NOT #1 (pixels).

### Two Approaches to Projection
1. **Metadata-only** (what we're doing):
   - Keep pixel grid as-is
   - Change CRS and geotransform to place it correctly in world
   - Fast, no resampling needed
   - Pixel array identical across projections

2. **Full reprojection** (not implemented):
   - Warp/resample the pixel data to fit target projection's coordinate system
   - Slow, requires interpolation
   - Pixel arrays would be different

## Debugging Changes Made

### 1. Added Debug Output (commit fcfa23c)

Added print statements to show:
```python
[DEBUG] Target projection: EPSG:3857
[DEBUG] Target EPSG: 3857
[DEBUG] Is WGS84: False
[DEBUG] Transforming to non-WGS84 projection...
[DEBUG] WGS84 bounds: W=-105.668234, E=-104.331766, S=40.150000, N=41.250000
[DEBUG] Projected bounds: W=-11768234.56, E=-11618234.56, S=4887651.21, N=5037651.21
[DEBUG] Final CRS: EPSG:3857
[DEBUG] Final transform: | 1500.00, 0.00, -11768234.56|
                          | 0.00, -1500.00, 5037651.21|
                          | 0.00, 0.00, 1.00|
```

### 2. Added Verification Script (commit 03a84d8)

Location: `tests/test_projection_verification.py`

This script:
- Creates test GeoTIFFs with both projections
- Compares their metadata
- Reports if there's a bug

## How to Verify

### Method 1: Run Verification Script
```bash
cd /home/runner/work/radar-processor/radar-processor
python tests/test_projection_verification.py
```

Look for output like:
```
✓ GOOD: Files have different CRS
   Web Mercator: EPSG:3857
   WGS84: EPSG:4326

✓ GOOD: Bounds are different
   Web Mercator magnitude: 11,688,000
   WGS84 magnitude: 105.5
   ✓ Magnitudes are correct!
```

### Method 2: Inspect with gdalinfo
```bash
# Check Web Mercator file
gdalinfo /tmp/geotiff_test/test_webmercator.tif | grep -A 5 "Coordinate System"

# Check WGS84 file
gdalinfo /tmp/geotiff_test/test_wgs84.tif | grep -A 5 "Coordinate System"
```

### Method 3: Open in QGIS
1. Open both files in QGIS
2. Check the CRS in layer properties
3. They should have different CRS
4. If you reproject the layer, the position should match

## Possible Bugs to Check

### Bug 1: Both files have same CRS
**Symptom**: Debug output shows same CRS for both
**Cause**: Logic error in `if target_proj.to_epsg() != 4326:` condition
**Fix**: Need to check if `to_epsg()` is returning unexpected value

### Bug 2: Both files have same bounds
**Symptom**: Debug output shows same bounds (in same units)
**Cause**: Transformation not being applied
**Fix**: Check transformer is working correctly

### Bug 3: User expectation mismatch
**Symptom**: Files "look the same" but metadata is different
**Cause**: User expects pixels to be different (full reprojection)
**Reality**: Current implementation only changes metadata
**Solution**: Either:
  - Clarify this is expected behavior
  - Implement full reprojection (more complex)

## Code Logic Flow

```
Input: 2D array + GridGeometry + radar location + projection

Step 1: Apply colormap to get RGBA image (SAME for all projections)

Step 2: Get grid bounds in meters (relative to radar)
  e.g., x: [-50000, 50000], y: [-50000, 50000]

Step 3: Transform to WGS84 (degrees)
  Using Azimuthal Equidistant projection centered on radar
  Result: e.g., lon: [-105.668, -104.332], lat: [40.150, 41.250]

Step 4: Check target projection
  If target == WGS84:
    Use bounds from Step 3 directly
    CRS = 'EPSG:4326'
  
  If target != WGS84 (e.g., Web Mercator):
    Transform bounds from WGS84 to target projection
    Result: e.g., x: [-11768234, -11618234], y: [4887651, 5037651] (meters)
    CRS = target_proj object

Step 5: Create geotransform from bounds
  Maps pixel coordinates to world coordinates

Step 6: Write file
  - RGBA data (SAME in all projections)
  - CRS (DIFFERENT)
  - Transform (DIFFERENT)
```

## Next Steps

1. User should run the verification script and share output
2. If debug shows different metadata: Expected behavior, clarify to user
3. If debug shows same metadata: There's a bug, investigate `pyproj.CRS.to_epsg()` behavior
4. If user wants pixels to be different: Need to implement full reprojection (separate feature)

## Related Files

- Main implementation: `src/radar_grid/geotiff.py`
  - `create_geotiff()` lines 263-288
  - `create_cog()` lines 463-485
- Test script: `tests/test_projection_verification.py`
- Example usage: `examples/geotiff_generation_example.py`
