"""
Advanced example: Process multiple fields with filters and custom colormaps.

This example demonstrates more advanced features:
- Processing multiple fields
- Applying quality control filters
- Using custom colormaps
- Processing different products (PPI, CAPPI, COLMAX)
"""
from radar_cog_processor import process_radar_to_cog
from pathlib import Path


class RangeFilter:
    """Simple filter class for demonstration."""
    def __init__(self, field, min_val=None, max_val=None):
        self.field = field
        self.min = min_val
        self.max = max_val


def process_with_filters():
    """Process radar data with quality control filters."""
    radar_file = "data/netcdf/RMA3_0315_01_20251215T231006Z.nc"
    output_dir = "output/filtered"
    
    # Define quality control filters
    # Filter out low correlation values
    filters = [
        RangeFilter(field="RHOHV", min_val=0.8, max_val=1.0),
    ]
    
    result = process_radar_to_cog(
        filepath=radar_file,
        product="PPI",
        field_requested="DBZH",
        elevation=0,
        filters=filters,
        output_dir=output_dir
    )
    
    print(f"Filtered COG created: {result['image_url']}")


def process_multiple_fields():
    """Process multiple radar fields."""
    radar_file = "data/netcdf/RMA3_0315_01_20251215T231006Z.nc"
    output_dir = "output/multi_field"
    
    # List of fields to process
    fields = ["DBZH", "ZDR", "RHOHV", "KDP", "VRAD"]
    
    results = []
    for field in fields:
        try:
            result = process_radar_to_cog(
                filepath=radar_file,
                product="PPI",
                field_requested=field,
                elevation=0,
                output_dir=output_dir
            )
            results.append(result)
            print(f"✓ Processed {field}: {result['image_url']}")
        except Exception as e:
            print(f"✗ Failed to process {field}: {e}")
    
    print(f"\nProcessed {len(results)} fields successfully")
    return results


def process_with_custom_colormap():
    """Process with custom colormap selection."""
    radar_file = "data/netcdf/RMA3_0315_01_20251215T231006Z.nc"
    output_dir = "output/custom_cmap"
    
    # Override default colormap
    colormap_overrides = {
        "DBZH": "grc_th2",  # Use alternative reflectivity colormap
    }
    
    result = process_radar_to_cog(
        filepath=radar_file,
        product="PPI",
        field_requested="DBZH",
        elevation=0,
        colormap_overrides=colormap_overrides,
        output_dir=output_dir
    )
    
    print(f"COG with custom colormap: {result['image_url']}")


def process_different_products():
    """Process different radar products."""
    radar_file = "data/netcdf/RMA3_0315_01_20251215T231006Z.nc"
    output_dir = "output/products"
    
    # PPI at different elevations
    for elev in [0, 1, 2]:
        result = process_radar_to_cog(
            filepath=radar_file,
            product="PPI",
            field_requested="DBZH",
            elevation=elev,
            output_dir=output_dir
        )
        print(f"PPI elevation {elev}: {result['image_url']}")
    
    # CAPPI at different heights
    for height in [2000, 4000, 6000]:
        result = process_radar_to_cog(
            filepath=radar_file,
            product="CAPPI",
            field_requested="DBZH",
            cappi_height=height,
            output_dir=output_dir
        )
        print(f"CAPPI at {height}m: {result['image_url']}")
    
    # COLMAX (composite maximum)
    result = process_radar_to_cog(
        filepath=radar_file,
        product="COLMAX",
        field_requested="DBZH",
        output_dir=output_dir
    )
    print(f"COLMAX: {result['image_url']}")


def main():
    print("=== Processing with Filters ===")
    process_with_filters()
    
    print("\n=== Processing Multiple Fields ===")
    process_multiple_fields()
    
    print("\n=== Processing with Custom Colormap ===")
    process_with_custom_colormap()
    
    print("\n=== Processing Different Products ===")
    process_different_products()


if __name__ == "__main__":
    main()
