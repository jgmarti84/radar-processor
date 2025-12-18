"""
Basic example: Process a single radar file to COG.

This example demonstrates the simplest use case - processing a single
radar NetCDF file into a Cloud-Optimized GeoTIFF.
"""
from radar_cog_processor import process_radar_to_cog
from pathlib import Path


def main():
    # Path to your radar NetCDF file
    radar_file = "data/netcdf/RMA1_0315_01_20251208T191648Z.nc"
    
    # Output directory for COG files
    output_dir = "output"
    
    # Process the radar file with default settings
    # This will create a PPI (Plan Position Indicator) of reflectivity
    result = process_radar_to_cog(
        filepath=radar_file,
        product="PPI",              # Product type: PPI, CAPPI, or COLMAX
        field_requested="DBZH",     # Reflectivity field
        elevation=0,                # Lowest elevation angle
        output_dir=output_dir
    )
    
    # Print results
    print(f"COG file created: {result['image_url']}")
    print(f"Processed field: {result['field']}")
    print(f"Source file: {result['source_file']}")
    
    # Verify file was created
    cog_path = Path(result['image_url'])
    if cog_path.exists():
        print(f"\nSuccess! File size: {cog_path.stat().st_size / 1024:.2f} KB")
    else:
        print("\nWarning: Output file not found!")


if __name__ == "__main__":
    main()
