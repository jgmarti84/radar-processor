"""
All Options Example - Demonstrate every combination of processing options.

This comprehensive example shows how to process radar data with:
- All products (PPI, CAPPI, COLMAX)
- All available fields (DBZH, ZDR, RHOHV, KDP, VRAD, WRAD, PHIDP)
- Multiple elevations for PPI
- Multiple heights for CAPPI
- Various filter combinations
- Custom colormaps

Run this example:
    python examples/all_options.py

This will create many output files demonstrating the full capability
of the radar_cog_processor library.
"""
from pathlib import Path
import sys

# Add parent to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from radar_cog_processor import process_radar_to_cog
from radar_cog_processor.cache import GRID2D_CACHE, GRID3D_CACHE
from radar_cog_processor.constants import FIELD_RENDER, FIELD_COLORMAP_OPTIONS


# ============================================================================
# Configuration
# ============================================================================

RADAR_FILE = Path("data/netcdf/RMA1_0315_01_20251208T191648Z.nc")
OUTPUT_BASE = Path("output/all_options")


class Filter:
    """Simple filter class."""
    def __init__(self, field: str, min: float = None, max: float = None):
        self.field = field
        self.min = min
        self.max = max
    
    def __repr__(self):
        parts = [f"field='{self.field}'"]
        if self.min is not None:
            parts.append(f"min={self.min}")
        if self.max is not None:
            parts.append(f"max={self.max}")
        return f"Filter({', '.join(parts)})"


# ============================================================================
# Available Options
# ============================================================================

# All available products
PRODUCTS = ["PPI", "CAPPI", "COLMAX"]

# Available fields (what the radar measures)
FIELDS = {
    "DBZH": {
        "name": "Horizontal Reflectivity",
        "unit": "dBZ",
        "description": "Precipitation intensity measurement"
    },
    "ZDR": {
        "name": "Differential Reflectivity", 
        "unit": "dB",
        "description": "Difference between horizontal and vertical reflectivity"
    },
    "RHOHV": {
        "name": "Cross-Correlation Coefficient",
        "unit": "",
        "description": "Correlation between H and V polarization (quality indicator)"
    },
    "KDP": {
        "name": "Specific Differential Phase",
        "unit": "deg/km",
        "description": "Phase shift difference (rain rate estimation)"
    },
    "VRAD": {
        "name": "Radial Velocity",
        "unit": "m/s",
        "description": "Motion toward/away from radar"
    },
    "WRAD": {
        "name": "Spectrum Width",
        "unit": "m/s",
        "description": "Velocity variance (turbulence indicator)"
    },
    "PHIDP": {
        "name": "Differential Phase",
        "unit": "degrees",
        "description": "Cumulative phase difference"
    },
}

# Elevation angles to test for PPI (indices, not degrees)
ELEVATIONS = [0, 1, 2]  # First three elevation angles

# Heights to test for CAPPI (in meters)
CAPPI_HEIGHTS = [2000, 3000, 4000, 5000]


# ============================================================================
# Example Functions
# ============================================================================

def demo_all_products():
    """
    Demonstrate all three product types with the same field.
    
    Products:
    - PPI (Plan Position Indicator): Horizontal slice at specific elevation angle
    - CAPPI (Constant Altitude PPI): Horizontal slice at constant height
    - COLMAX (Column Maximum): Maximum value in each vertical column
    """
    print("\n" + "="*70)
    print("DEMO 1: All Products (DBZH field)")
    print("="*70)
    
    output_dir = OUTPUT_BASE / "products"
    
    # Standard quality filter
    filters = [Filter("RHOHV", min=0.85)]
    
    # PPI at lowest elevation
    print("\n[1/3] Processing PPI (elevation 0)...")
    GRID2D_CACHE.clear()
    GRID3D_CACHE.clear()
    
    result_ppi = process_radar_to_cog(
        filepath=str(RADAR_FILE),
        product="PPI",
        field_requested="DBZH",
        elevation=0,
        filters=filters,
        output_dir=str(output_dir / "PPI")
    )
    print(f"    ✓ {Path(result_ppi['image_url']).name}")
    
    # CAPPI at 3km
    print("\n[2/3] Processing CAPPI (3000m)...")
    GRID2D_CACHE.clear()
    GRID3D_CACHE.clear()
    
    result_cappi = process_radar_to_cog(
        filepath=str(RADAR_FILE),
        product="CAPPI",
        field_requested="DBZH",
        cappi_height=3000,
        filters=filters,
        output_dir=str(output_dir / "CAPPI")
    )
    print(f"    ✓ {Path(result_cappi['image_url']).name}")
    
    # COLMAX
    print("\n[3/3] Processing COLMAX...")
    GRID2D_CACHE.clear()
    GRID3D_CACHE.clear()
    
    result_colmax = process_radar_to_cog(
        filepath=str(RADAR_FILE),
        product="COLMAX",
        field_requested="DBZH",
        filters=filters,
        output_dir=str(output_dir / "COLMAX")
    )
    print(f"    ✓ {Path(result_colmax['image_url']).name}")
    
    return [result_ppi, result_cappi, result_colmax]


def demo_all_fields():
    """
    Demonstrate processing different radar fields.
    
    Note: Not all fields may be available in every radar file.
    """
    print("\n" + "="*70)
    print("DEMO 2: All Available Fields (PPI product)")
    print("="*70)
    
    output_dir = OUTPUT_BASE / "fields"
    results = []
    
    for i, (field, info) in enumerate(FIELDS.items(), 1):
        print(f"\n[{i}/{len(FIELDS)}] Processing {field} - {info['name']}...")
        print(f"    Description: {info['description']}")
        
        GRID2D_CACHE.clear()
        GRID3D_CACHE.clear()
        
        try:
            result = process_radar_to_cog(
                filepath=str(RADAR_FILE),
                product="PPI",
                field_requested=field,
                elevation=0,
                filters=None,  # No filters for raw field visualization
                output_dir=str(output_dir / field)
            )
            print(f"    ✓ {Path(result['image_url']).name}")
            results.append(result)
        except (KeyError, ValueError) as e:
            print(f"    ✗ Field not available: {e}")
    
    return results


def demo_multiple_elevations():
    """
    Demonstrate PPI at multiple elevation angles.
    
    Lower elevations see further but lower in atmosphere.
    Higher elevations see higher up but with shorter range.
    """
    print("\n" + "="*70)
    print("DEMO 3: Multiple PPI Elevations")
    print("="*70)
    
    output_dir = OUTPUT_BASE / "elevations"
    results = []
    
    filters = [Filter("RHOHV", min=0.85)]
    
    for i, elev in enumerate(ELEVATIONS, 1):
        print(f"\n[{i}/{len(ELEVATIONS)}] Processing PPI at elevation {elev}...")
        
        GRID2D_CACHE.clear()
        GRID3D_CACHE.clear()
        
        try:
            result = process_radar_to_cog(
                filepath=str(RADAR_FILE),
                product="PPI",
                field_requested="DBZH",
                elevation=elev,
                filters=filters,
                output_dir=str(output_dir / f"elevation_{elev}")
            )
            print(f"    ✓ {Path(result['image_url']).name}")
            results.append(result)
        except ValueError as e:
            print(f"    ✗ Elevation not available: {e}")
    
    return results


def demo_multiple_heights():
    """
    Demonstrate CAPPI at multiple heights.
    
    Different heights show precipitation at different altitudes.
    Lower heights: Near surface precipitation
    Higher heights: Above ground level, may miss shallow precipitation
    """
    print("\n" + "="*70)
    print("DEMO 4: Multiple CAPPI Heights")
    print("="*70)
    
    output_dir = OUTPUT_BASE / "heights"
    results = []
    
    filters = [Filter("RHOHV", min=0.85)]
    
    for i, height in enumerate(CAPPI_HEIGHTS, 1):
        print(f"\n[{i}/{len(CAPPI_HEIGHTS)}] Processing CAPPI at {height}m...")
        
        GRID2D_CACHE.clear()
        GRID3D_CACHE.clear()
        
        result = process_radar_to_cog(
            filepath=str(RADAR_FILE),
            product="CAPPI",
            field_requested="DBZH",
            cappi_height=height,
            filters=filters,
            output_dir=str(output_dir / f"height_{height}m")
        )
        print(f"    ✓ {Path(result['image_url']).name}")
        results.append(result)
    
    return results


def demo_filter_variations():
    """
    Demonstrate various filter combinations.
    """
    print("\n" + "="*70)
    print("DEMO 5: Filter Variations")
    print("="*70)
    
    output_dir = OUTPUT_BASE / "filters"
    
    filter_configs = [
        {
            "name": "no_filter",
            "description": "No filters (raw data)",
            "filters": None
        },
        {
            "name": "rhohv_permissive",
            "description": "RHOHV >= 0.80 (permissive QC)",
            "filters": [Filter("RHOHV", min=0.80)]
        },
        {
            "name": "rhohv_standard",
            "description": "RHOHV >= 0.85 (standard QC)",
            "filters": [Filter("RHOHV", min=0.85)]
        },
        {
            "name": "rhohv_strict",
            "description": "RHOHV >= 0.92 (strict QC)",
            "filters": [Filter("RHOHV", min=0.92)]
        },
        {
            "name": "dbzh_range_low",
            "description": "DBZH 0-30 (light precipitation)",
            "filters": [Filter("DBZH", min=0, max=30)]
        },
        {
            "name": "dbzh_range_medium",
            "description": "DBZH 20-45 (moderate precipitation)",
            "filters": [Filter("DBZH", min=20, max=45)]
        },
        {
            "name": "dbzh_range_high",
            "description": "DBZH 40+ (heavy precipitation)",
            "filters": [Filter("DBZH", min=40)]
        },
        {
            "name": "combined_standard",
            "description": "RHOHV >= 0.85 AND 10 <= DBZH <= 60",
            "filters": [
                Filter("RHOHV", min=0.85),
                Filter("DBZH", min=10, max=60)
            ]
        },
        {
            "name": "combined_strict",
            "description": "RHOHV >= 0.90 AND 15 <= DBZH <= 55",
            "filters": [
                Filter("RHOHV", min=0.90),
                Filter("DBZH", min=15, max=55)
            ]
        },
    ]
    
    results = []
    
    for i, config in enumerate(filter_configs, 1):
        print(f"\n[{i}/{len(filter_configs)}] {config['name']}")
        print(f"    {config['description']}")
        
        GRID2D_CACHE.clear()
        GRID3D_CACHE.clear()
        
        result = process_radar_to_cog(
            filepath=str(RADAR_FILE),
            product="PPI",
            field_requested="DBZH",
            elevation=0,
            filters=config["filters"],
            output_dir=str(output_dir / config["name"])
        )
        print(f"    ✓ {Path(result['image_url']).name}")
        results.append(result)
    
    return results


def demo_colormap_options():
    """
    Demonstrate different colormap options for DBZH.
    """
    print("\n" + "="*70)
    print("DEMO 6: Colormap Options (DBZH)")
    print("="*70)
    
    output_dir = OUTPUT_BASE / "colormaps"
    
    # Get available colormaps for DBZH
    dbzh_colormaps = FIELD_COLORMAP_OPTIONS.get("DBZH", ["grc_th"])
    
    results = []
    filters = [Filter("RHOHV", min=0.85)]
    
    for i, cmap in enumerate(dbzh_colormaps, 1):
        print(f"\n[{i}/{len(dbzh_colormaps)}] Using colormap: {cmap}")
        
        GRID2D_CACHE.clear()
        GRID3D_CACHE.clear()
        
        result = process_radar_to_cog(
            filepath=str(RADAR_FILE),
            product="PPI",
            field_requested="DBZH",
            elevation=0,
            filters=filters,
            colormap_overrides={"DBZH": cmap},
            output_dir=str(output_dir / cmap.replace("/", "_"))
        )
        print(f"    ✓ {Path(result['image_url']).name}")
        results.append(result)
    
    return results


def demo_comprehensive():
    """
    Create a comprehensive set of outputs: one for each combination.
    
    This creates files for quick reference of all options.
    """
    print("\n" + "="*70)
    print("DEMO 7: Comprehensive Reference Set")
    print("="*70)
    print("Creating one file for each major option combination...")
    
    output_dir = OUTPUT_BASE / "comprehensive"
    
    # Standard filter
    filters = [Filter("RHOHV", min=0.85), Filter("DBZH", min=10, max=60)]
    
    combinations = [
        # Product variations
        {"product": "PPI", "field": "DBZH", "elevation": 0, "cappi_height": None},
        {"product": "PPI", "field": "DBZH", "elevation": 1, "cappi_height": None},
        {"product": "PPI", "field": "DBZH", "elevation": 2, "cappi_height": None},
        {"product": "CAPPI", "field": "DBZH", "elevation": 0, "cappi_height": 2000},
        {"product": "CAPPI", "field": "DBZH", "elevation": 0, "cappi_height": 3000},
        {"product": "CAPPI", "field": "DBZH", "elevation": 0, "cappi_height": 4000},
        {"product": "COLMAX", "field": "DBZH", "elevation": 0, "cappi_height": None},
        
        # Field variations (PPI only)
        {"product": "PPI", "field": "ZDR", "elevation": 0, "cappi_height": None},
        {"product": "PPI", "field": "RHOHV", "elevation": 0, "cappi_height": None},
        {"product": "PPI", "field": "VRAD", "elevation": 0, "cappi_height": None},
    ]
    
    results = []
    
    for i, combo in enumerate(combinations, 1):
        suffix = f"{combo['product']}_{combo['field']}"
        if combo['product'] == 'PPI':
            suffix += f"_elev{combo['elevation']}"
        elif combo['product'] == 'CAPPI':
            suffix += f"_{combo['cappi_height']}m"
        
        print(f"\n[{i}/{len(combinations)}] {suffix}")
        
        GRID2D_CACHE.clear()
        GRID3D_CACHE.clear()
        
        try:
            result = process_radar_to_cog(
                filepath=str(RADAR_FILE),
                product=combo["product"],
                field_requested=combo["field"],
                elevation=combo["elevation"],
                cappi_height=combo["cappi_height"] or 4000,
                filters=filters if combo["field"] == "DBZH" else None,
                output_dir=str(output_dir)
            )
            print(f"    ✓ {Path(result['image_url']).name}")
            results.append(result)
        except (KeyError, ValueError) as e:
            print(f"    ✗ Skipped: {e}")
    
    return results


# ============================================================================
# Main
# ============================================================================

def main():
    """Run all demonstrations."""
    print("\n" + "#"*70)
    print("#" + " "*68 + "#")
    print("#" + "    RADAR COG PROCESSOR - ALL OPTIONS DEMONSTRATION".center(68) + "#")
    print("#" + " "*68 + "#")
    print("#"*70)
    
    # Check if data file exists
    if not RADAR_FILE.exists():
        print(f"\n❌ Error: Radar file not found: {RADAR_FILE}")
        print("Please ensure you have sample data in data/netcdf/")
        return
    
    # Create output directory
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    
    print(f"\nUsing radar file: {RADAR_FILE}")
    print(f"Output base directory: {OUTPUT_BASE}")
    
    # Run demos
    demo_all_products()
    demo_all_fields()
    demo_multiple_elevations()
    demo_multiple_heights()
    demo_filter_variations()
    demo_colormap_options()
    demo_comprehensive()
    
    print("\n" + "="*70)
    print("ALL DEMONSTRATIONS COMPLETE")
    print("="*70)
    print(f"\nOutput files organized in: {OUTPUT_BASE}")
    print("\nDirectory structure:")
    print("  └── all_options/")
    print("      ├── products/     - PPI, CAPPI, COLMAX comparisons")
    print("      ├── fields/       - Different radar measurements")
    print("      ├── elevations/   - Multiple PPI angles")
    print("      ├── heights/      - Multiple CAPPI altitudes")
    print("      ├── filters/      - Filter combinations")
    print("      ├── colormaps/    - Color scheme options")
    print("      └── comprehensive/- Quick reference set")
    print("="*70)


if __name__ == "__main__":
    main()
