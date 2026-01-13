#!/usr/bin/env python
"""
Simple validation script to test the GeoTIFF generation module.

This script can be run without actual radar data to validate that
the module structure and API are correct.
"""

import sys
import traceback

def test_imports():
    """Test that all required imports work."""
    print("Testing imports...")
    try:
        # Test main imports
        from radar_grid import (
            create_geotiff,
            create_cog,
            save_product_as_geotiff,
            apply_colormap_to_array
        )
        print("  ✓ Main functions imported successfully")
        
        # Test that they are callable
        assert callable(create_geotiff)
        assert callable(create_cog)
        assert callable(save_product_as_geotiff)
        assert callable(apply_colormap_to_array)
        print("  ✓ All functions are callable")
        
        return True
    except Exception as e:
        print(f"  ✗ Import failed: {e}")
        traceback.print_exc()
        return False


def test_colormap_function():
    """Test the colormap application function."""
    print("\nTesting colormap application...")
    try:
        import numpy as np
        from radar_grid import apply_colormap_to_array
        
        # Create test data
        data = np.random.rand(50, 50) * 100
        data[20:30, 20:30] = np.nan  # Add some NaN values
        
        # Apply colormap
        rgba = apply_colormap_to_array(data, 'viridis', vmin=0, vmax=100)
        
        # Validate output
        assert rgba.shape == (50, 50, 4), f"Expected shape (50, 50, 4), got {rgba.shape}"
        assert rgba.dtype == np.uint8, f"Expected dtype uint8, got {rgba.dtype}"
        assert np.all(rgba >= 0) and np.all(rgba <= 255), "Values out of range [0, 255]"
        
        # Check that NaN region has alpha = 0
        assert np.all(rgba[20:30, 20:30, 3] == 0), "NaN region should be transparent"
        
        print("  ✓ Colormap application works correctly")
        return True
    except Exception as e:
        print(f"  ✗ Colormap test failed: {e}")
        traceback.print_exc()
        return False


def test_geotiff_api():
    """Test that the GeoTIFF API is correctly structured."""
    print("\nTesting GeoTIFF API structure...")
    try:
        import inspect
        from radar_grid.geotiff import create_geotiff, create_cog, save_product_as_geotiff
        
        # Check function signatures
        sig_geotiff = inspect.signature(create_geotiff)
        sig_cog = inspect.signature(create_cog)
        sig_save = inspect.signature(save_product_as_geotiff)
        
        # Verify key parameters exist
        geotiff_params = list(sig_geotiff.parameters.keys())
        required_params = ['data', 'geometry', 'radar_lat', 'radar_lon', 'output_path']
        for param in required_params:
            assert param in geotiff_params, f"Missing parameter: {param}"
        
        print("  ✓ All required parameters present")
        
        # Check optional parameters
        optional_params = ['cmap', 'vmin', 'vmax', 'projection']
        for param in optional_params:
            assert param in geotiff_params, f"Missing optional parameter: {param}"
        
        print("  ✓ Optional parameters present")
        
        # Check COG-specific parameters
        cog_params = list(sig_cog.parameters.keys())
        assert 'overview_factors' in cog_params, "Missing COG parameter: overview_factors"
        assert 'resampling_method' in cog_params, "Missing COG parameter: resampling_method"
        
        print("  ✓ COG-specific parameters present")
        
        return True
    except Exception as e:
        print(f"  ✗ API structure test failed: {e}")
        traceback.print_exc()
        return False


def test_geometry_integration():
    """Test that the module integrates properly with GridGeometry."""
    print("\nTesting GridGeometry integration...")
    try:
        import numpy as np
        from radar_grid.geometry import GridGeometry
        
        # Create a minimal GridGeometry
        nz, ny, nx = 10, 50, 50
        grid_shape = (nz, ny, nx)
        grid_limits = (
            (0.0, 10000.0),
            (-25000.0, 25000.0),
            (-25000.0, 25000.0)
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
        
        print(f"  ✓ Created GridGeometry: {geometry.grid_shape}")
        
        # Test that grid_limits are accessible
        assert len(geometry.grid_limits) == 3
        assert len(geometry.grid_limits[0]) == 2
        print("  ✓ GridGeometry structure is correct")
        
        return True
    except Exception as e:
        print(f"  ✗ GridGeometry integration test failed: {e}")
        traceback.print_exc()
        return False


def test_projection_support():
    """Test that projection specifications are valid."""
    print("\nTesting projection support...")
    try:
        import pyproj
        
        # Test common projections
        projections = [
            'EPSG:3857',  # Web Mercator
            'EPSG:4326',  # WGS84
            'EPSG:32633', # UTM Zone 33N
        ]
        
        for proj_str in projections:
            try:
                proj = pyproj.CRS(proj_str)
                print(f"  ✓ {proj_str}: {proj.name}")
            except Exception as e:
                print(f"  ✗ {proj_str} failed: {e}")
                return False
        
        return True
    except Exception as e:
        print(f"  ✗ Projection test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all validation tests."""
    print("=" * 60)
    print("GeoTIFF Generation Module Validation")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Colormap Function", test_colormap_function),
        ("GeoTIFF API", test_geotiff_api),
        ("GridGeometry Integration", test_geometry_integration),
        ("Projection Support", test_projection_support),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ Test '{name}' crashed: {e}")
            traceback.print_exc()
            results.append((name, False))
    
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8} {name}")
    
    total = len(results)
    passed = sum(1 for _, r in results if r)
    
    print("=" * 60)
    print(f"Total: {passed}/{total} tests passed")
    print("=" * 60)
    
    return 0 if passed == total else 1


if __name__ == '__main__':
    sys.exit(main())
