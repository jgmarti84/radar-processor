#!/usr/bin/env python
"""
Example: Using Configurable COG Overviews

Demonstrates how to use the new overview configuration parameters
in process_radar_to_cog() and convert_to_cog().
"""

from radar_cog_processor import process_radar_to_cog
from radar_cog_processor.processor import convert_to_cog
import time


def example_1_default_behavior():
    """Example 1: Default behavior (backward compatible)."""
    print("Example 1: Default Behavior")
    print("=" * 60)
    
    # Uses default settings: [2, 4, 8, 16] overviews with 'nearest' resampling
    result = process_radar_to_cog(
        'data/netcdf/radar_file.nc',
        product='PPI',
        field_requested='DBZH'
    )
    
    print(f"✓ Generated COG: {result['image_url']}")
    print(f"  Field: {result['field']}")
    print()


def example_2_fast_development():
    """Example 2: Fast processing for development."""
    print("Example 2: Fast Processing (Development)")
    print("=" * 60)
    
    start = time.time()
    result = process_radar_to_cog(
        'data/netcdf/radar_file.nc',
        product='CAPPI',
        cappi_height=5000,
        overview_factors=[2, 4],  # Only 2 levels instead of 4
        resampling_method='nearest'
    )
    elapsed = time.time() - start
    
    print(f"✓ Generated COG in {elapsed:.2f}s: {result['image_url']}")
    print(f"  Overviews: 2 levels (2x, 4x)")
    print(f"  Resampling: Nearest (fastest)")
    print()


def example_3_high_quality_radar():
    """Example 3: High quality radar data processing."""
    print("Example 3: High Quality Radar Data")
    print("=" * 60)
    
    result = process_radar_to_cog(
        'data/netcdf/radar_file.nc',
        product='COLMAX',
        resampling_method='average'  # Preserve intensity values
    )
    
    print(f"✓ Generated COG: {result['image_url']}")
    print(f"  Overviews: 4 levels (default)")
    print(f"  Resampling: Average (better for radar data)")
    print()


def example_4_web_mapping():
    """Example 4: Optimized for web mapping/tiling servers."""
    print("Example 4: Web Mapping Optimization")
    print("=" * 60)
    
    result = process_radar_to_cog(
        'data/netcdf/radar_file.nc',
        product='PPI',
        overview_factors=[2, 4, 8, 16, 32],  # 5 levels for all zoom levels
        resampling_method='nearest'
    )
    
    print(f"✓ Generated COG: {result['image_url']}")
    print(f"  Overviews: 5 levels (2x, 4x, 8x, 16x, 32x)")
    print(f"  Resampling: Nearest (fast tile generation)")
    print(f"  Use Case: Web mapping with comprehensive zoom levels")
    print()


def example_5_archival_no_overviews():
    """Example 5: Fast COG creation without overviews (archival mode)."""
    print("Example 5: Archival Mode (No Overviews)")
    print("=" * 60)
    
    start = time.time()
    result = process_radar_to_cog(
        'data/netcdf/radar_file.nc',
        overview_factors=[]  # Disable overviews
    )
    elapsed = time.time() - start
    
    print(f"✓ Generated COG in {elapsed:.2f}s: {result['image_url']}")
    print(f"  Overviews: None (disabled)")
    print(f"  Use Case: Fast export for archival, larger files")
    print()


def example_6_direct_cog_conversion():
    """Example 6: Direct GeoTIFF to COG conversion with custom settings."""
    print("Example 6: Direct COG Conversion")
    print("=" * 60)
    
    # Convert an existing GeoTIFF to COG
    cog_path = convert_to_cog(
        src_path='intermediate_output.tif',
        cog_path='output_custom.cog',
        overview_factors=[2, 4, 8],
        resampling_method='cubic'
    )
    
    print(f"✓ Converted to COG: {cog_path}")
    print(f"  Overviews: 3 levels (2x, 4x, 8x)")
    print(f"  Resampling: Cubic (high quality)")
    print()


def example_7_performance_comparison():
    """Example 7: Compare performance of different configurations."""
    print("Example 7: Performance Comparison")
    print("=" * 60)
    
    test_file = 'data/netcdf/radar_file.nc'
    
    configurations = [
        {
            'name': 'No Overviews',
            'overview_factors': [],
            'resampling_method': 'nearest'
        },
        {
            'name': 'Fast (2 levels)',
            'overview_factors': [2, 4],
            'resampling_method': 'nearest'
        },
        {
            'name': 'Standard (4 levels)',
            'overview_factors': [2, 4, 8, 16],
            'resampling_method': 'nearest'
        },
        {
            'name': 'Comprehensive (5 levels)',
            'overview_factors': [2, 4, 8, 16, 32],
            'resampling_method': 'nearest'
        },
        {
            'name': 'High Quality (4 levels, avg)',
            'overview_factors': [2, 4, 8, 16],
            'resampling_method': 'average'
        },
    ]
    
    print(f"{'Configuration':<30} {'Time (s)':<12} {'Relative':<12}")
    print("-" * 60)
    
    baseline_time = None
    for config in configurations:
        start = time.time()
        try:
            result = process_radar_to_cog(
                test_file,
                overview_factors=config['overview_factors'],
                resampling_method=config['resampling_method']
            )
            elapsed = time.time() - start
            
            if baseline_time is None:
                baseline_time = elapsed
            
            relative = elapsed / baseline_time
            relative_str = f"{relative:.2f}x" if baseline_time > 0 else "baseline"
            
            print(f"{config['name']:<30} {elapsed:<12.3f} {relative_str:<12}")
        except Exception as e:
            print(f"{config['name']:<30} Error: {str(e)[:30]}")
    
    print()


def example_8_resampling_methods():
    """Example 8: Demonstrate different resampling methods."""
    print("Example 8: Resampling Methods")
    print("=" * 60)
    
    methods = {
        'nearest': 'Fastest, preserves exact values (good for categories)',
        'bilinear': 'Fast smooth interpolation',
        'cubic': 'High quality interpolation',
        'average': 'Averages pixel values (good for radar intensity)',
        'mode': 'Most frequent value (good for categories)',
        'max': 'Maximum value (good for reflectivity)',
        'min': 'Minimum value',
    }
    
    print("Available Resampling Methods:")
    print()
    for method, description in methods.items():
        print(f"  '{method}'")
        print(f"    → {description}")
    
    print()
    print("Recommendation for Radar Data:")
    print("  Use 'average' or 'mode' for reflectivity fields")
    print("  Use 'nearest' for fastest processing")
    print()


def main():
    """Run all examples."""
    print()
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "  COG Overviews Configuration Examples".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "=" * 58 + "╝")
    print()
    
    # Note: These are example demonstrations. Actual execution requires
    # valid radar files in the paths specified.
    
    print("NOTE: These examples demonstrate the API usage.")
    print("      Actual execution requires valid radar NetCDF files.")
    print()
    
    # Uncomment to run actual examples:
    # example_1_default_behavior()
    # example_2_fast_development()
    # example_3_high_quality_radar()
    # example_4_web_mapping()
    # example_5_archival_no_overviews()
    # example_6_direct_cog_conversion()
    
    example_7_performance_comparison()  # Conceptual
    example_8_resampling_methods()      # Display methods
    
    print("=" * 60)
    print("For more information, see COG_CONFIGURATION_IMPLEMENTATION.md")
    print("=" * 60)


if __name__ == '__main__':
    main()
