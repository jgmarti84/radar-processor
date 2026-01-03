"""
Filter Examples - Comprehensive demonstration of filter functionality.

This example demonstrates all aspects of filtering:
1. What filters are and how they work
2. QC (Quality Control) filters - applied during gridding
3. Visual filters - applied after gridding
4. Combining multiple filters
5. Common filtering scenarios

Understanding Filters
=====================

Filters mask (hide/remove) radar data based on conditions. They help:
- Remove non-meteorological echoes (ground clutter, birds, insects)
- Focus on specific precipitation intensity ranges
- Improve data quality for analysis

Filter Types
------------
1. QC Filters: Based on quality control fields (RHOHV)
   - Applied DURING the gridding process
   - Affect which data contributes to the final grid
   
2. Visual Filters: Based on the visualized field itself (e.g., DBZH range)
   - Applied AFTER gridding
   - Only affect what is displayed, not interpolation

Run this example:
    python examples/filter_examples.py
"""
from pathlib import Path
import sys

# Add parent to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from radar_processor import process_radar_to_cog
from radar_processor.cache import GRID2D_CACHE, GRID3D_CACHE


# ============================================================================
# Filter Class Definition
# ============================================================================

class Filter:
    """
    A filter object that specifies masking conditions.
    
    Attributes
    ----------
    field : str
        The field name to filter on (e.g., 'RHOHV', 'DBZH')
    min : float or None
        Minimum value - data BELOW this will be masked
    max : float or None
        Maximum value - data ABOVE this will be masked
        
    Examples
    --------
    >>> f = Filter("DBZH", min=10, max=50)  # Only show 10-50 dBZ
    >>> f = Filter("RHOHV", min=0.85)       # Only where RHOHV >= 0.85
    >>> f = Filter("DBZH", max=60)          # Only where DBZH <= 60
    """
    
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
# Example Data Path
# ============================================================================

RADAR_FILE = Path("data/netcdf/RMA1_0315_01_20251208T191648Z.nc")
OUTPUT_DIR = Path("output/filter_examples")


def ensure_output_dir():
    """Create output directory if it doesn't exist."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def clear_cache():
    """Clear the grid caches between examples for clean comparison."""
    GRID2D_CACHE.clear()
    GRID3D_CACHE.clear()


# ============================================================================
# Example 1: No Filters (Baseline)
# ============================================================================

def example_no_filters():
    """
    Process without any filters - the baseline for comparison.
    
    This shows ALL data from the radar, including:
    - Ground clutter (near the radar)
    - Non-meteorological echoes (birds, insects)
    - Very weak and very strong signals
    """
    print("\n" + "="*70)
    print("Example 1: NO FILTERS (Baseline)")
    print("="*70)
    print("Description: Raw radar data without any masking")
    print("Use case: See everything the radar detected")
    print("-"*70)
    
    clear_cache()
    
    result = process_radar_to_cog(
        filepath=str(RADAR_FILE),
        product="PPI",
        field_requested="DBZH",
        elevation=0,
        filters=None,  # No filters
        output_dir=str(OUTPUT_DIR / "01_no_filters")
    )
    
    print(f"✓ Output: {result['image_url']}")
    return result


# ============================================================================
# Example 2: QC Filter - RHOHV Quality Control
# ============================================================================

def example_rhohv_filter():
    """
    Apply RHOHV quality control filter.
    
    RHOHV (Cross-Correlation Coefficient) explained:
    ------------------------------------------------
    - Measures how similar horizontal and vertical polarization returns are
    - Range: 0 to 1
    - RHOHV ≈ 0.95-1.0: Pure rain or snow (meteorological)
    - RHOHV ≈ 0.85-0.95: Mixed precipitation, melting layer
    - RHOHV < 0.85: Likely non-meteorological (clutter, birds, insects)
    - RHOHV < 0.70: Almost certainly not precipitation
    
    Common thresholds:
    - min=0.80: Permissive (keep most echoes)
    - min=0.85: Standard (good balance)
    - min=0.90: Strict (only clear meteorological)
    - min=0.95: Very strict (may remove melting layer)
    """
    print("\n" + "="*70)
    print("Example 2: RHOHV Quality Control Filter")
    print("="*70)
    print("Description: Remove non-meteorological echoes using RHOHV")
    print("Filter: RHOHV >= 0.85 (standard threshold)")
    print("-"*70)
    
    clear_cache()
    
    # Create RHOHV filter
    rhohv_filter = Filter(
        field="RHOHV",
        min=0.85,   # Minimum correlation (removes clutter/birds)
        max=1.0     # Maximum (1.0 is theoretical max for pure targets)
    )
    
    print(f"Filter applied: {rhohv_filter}")
    print("\nWhat this does:")
    print("  - RHOHV < 0.85 → MASKED (likely ground clutter, birds, insects)")
    print("  - RHOHV >= 0.85 → VISIBLE (likely precipitation)")
    
    result = process_radar_to_cog(
        filepath=str(RADAR_FILE),
        product="PPI",
        field_requested="DBZH",
        elevation=0,
        filters=[rhohv_filter],
        output_dir=str(OUTPUT_DIR / "02_rhohv_filter")
    )
    
    print(f"\n✓ Output: {result['image_url']}")
    return result


# ============================================================================
# Example 3: Visual Filter - DBZH Range
# ============================================================================

def example_dbzh_range_filter():
    """
    Apply visual range filter to show specific reflectivity intensities.
    
    DBZH (Reflectivity) ranges:
    ---------------------------
    - -30 to 0 dBZ: Noise, insects, very light precipitation
    -  0 to 15 dBZ: Very light rain or mist
    - 15 to 30 dBZ: Light rain (~0.1-1 mm/hr)
    - 30 to 40 dBZ: Moderate rain (~1-10 mm/hr)
    - 40 to 50 dBZ: Heavy rain (~10-50 mm/hr)
    - 50 to 60 dBZ: Very heavy rain, possible hail
    - 60+ dBZ: Extreme - likely large hail
    
    This filter shows only moderate to heavy precipitation.
    """
    print("\n" + "="*70)
    print("Example 3: Visual Range Filter (DBZH)")
    print("="*70)
    print("Description: Show only moderate to heavy precipitation")
    print("Filter: 20 dBZ <= DBZH <= 55 dBZ")
    print("-"*70)
    
    clear_cache()
    
    # Create DBZH range filter
    dbzh_filter = Filter(
        field="DBZH",
        min=20,    # Hide weak echoes (< 20 dBZ)
        max=55     # Hide extreme echoes (> 55 dBZ, likely hail/errors)
    )
    
    print(f"Filter applied: {dbzh_filter}")
    print("\nWhat this does:")
    print("  - DBZH < 20 dBZ → MASKED (light drizzle, noise)")
    print("  - DBZH 20-55 dBZ → VISIBLE (rain)")
    print("  - DBZH > 55 dBZ → MASKED (possible hail)")
    
    result = process_radar_to_cog(
        filepath=str(RADAR_FILE),
        product="PPI",
        field_requested="DBZH",
        elevation=0,
        filters=[dbzh_filter],
        output_dir=str(OUTPUT_DIR / "03_dbzh_range")
    )
    
    print(f"\n✓ Output: {result['image_url']}")
    return result


# ============================================================================
# Example 4: Combined Filters
# ============================================================================

def example_combined_filters():
    """
    Combine QC and visual filters for best results.
    
    This is the recommended approach for clean visualizations:
    1. RHOHV filter removes non-meteorological echoes
    2. DBZH filter focuses on relevant intensity range
    
    The filters are applied in this order:
    1. QC filters (RHOHV) during gridding
    2. Visual filters (DBZH) after gridding
    """
    print("\n" + "="*70)
    print("Example 4: Combined QC + Visual Filters")
    print("="*70)
    print("Description: Best practice - clean meteorological display")
    print("Filters: RHOHV >= 0.85 AND 15 <= DBZH <= 60")
    print("-"*70)
    
    clear_cache()
    
    filters = [
        # QC filter - remove non-meteorological
        Filter(field="RHOHV", min=0.85, max=1.0),
        
        # Visual filter - focus on significant precipitation
        Filter(field="DBZH", min=15, max=60),
    ]
    
    print("Filters applied:")
    for f in filters:
        print(f"  - {f}")
    
    print("\nProcessing order:")
    print("  1. RHOHV filter applied during gridding (affects interpolation)")
    print("  2. DBZH filter applied after gridding (affects display only)")
    
    result = process_radar_to_cog(
        filepath=str(RADAR_FILE),
        product="PPI",
        field_requested="DBZH",
        elevation=0,
        filters=filters,
        output_dir=str(OUTPUT_DIR / "04_combined")
    )
    
    print(f"\n✓ Output: {result['image_url']}")
    return result


# ============================================================================
# Example 5: Strict QC Filter
# ============================================================================

def example_strict_qc():
    """
    Very strict RHOHV filter for maximum data quality.
    
    Use this when:
    - You need absolutely clean data
    - Doing quantitative precipitation estimation
    - There's a lot of non-meteorological contamination
    
    Note: May remove some valid data (melting layer, mixed phase)
    """
    print("\n" + "="*70)
    print("Example 5: Strict QC Filter")
    print("="*70)
    print("Description: Very strict filtering for clean data")
    print("Filter: RHOHV >= 0.92")
    print("-"*70)
    
    clear_cache()
    
    strict_filter = Filter(field="RHOHV", min=0.92, max=1.0)
    
    print(f"Filter applied: {strict_filter}")
    print("\nWhat this does:")
    print("  - Keeps only high-quality precipitation echoes")
    print("  - May remove melting layer (where RHOHV drops)")
    print("  - Use for quantitative analysis")
    
    result = process_radar_to_cog(
        filepath=str(RADAR_FILE),
        product="PPI",
        field_requested="DBZH",
        elevation=0,
        filters=[strict_filter],
        output_dir=str(OUTPUT_DIR / "05_strict_qc")
    )
    
    print(f"\n✓ Output: {result['image_url']}")
    return result


# ============================================================================
# Example 6: Minimum Value Only
# ============================================================================

def example_min_only():
    """
    Filter with only a minimum threshold.
    
    When max is None, there's no upper limit - useful for:
    - "Show everything above X dBZ"
    - Highlighting significant features
    """
    print("\n" + "="*70)
    print("Example 6: Minimum Threshold Only")
    print("="*70)
    print("Description: Show all echoes above 25 dBZ")
    print("Filter: DBZH >= 25 (no maximum)")
    print("-"*70)
    
    clear_cache()
    
    min_filter = Filter(field="DBZH", min=25, max=None)
    
    print(f"Filter applied: {min_filter}")
    print("\nWhat this does:")
    print("  - Shows all values >= 25 dBZ")
    print("  - No upper limit (even 70+ dBZ visible)")
    print("  - Good for 'significant weather' displays")
    
    result = process_radar_to_cog(
        filepath=str(RADAR_FILE),
        product="PPI",
        field_requested="DBZH",
        elevation=0,
        filters=[min_filter],
        output_dir=str(OUTPUT_DIR / "06_min_only")
    )
    
    print(f"\n✓ Output: {result['image_url']}")
    return result


# ============================================================================
# Example 7: Maximum Value Only
# ============================================================================

def example_max_only():
    """
    Filter with only a maximum threshold.
    
    When min is None, there's no lower limit - useful for:
    - "Show everything below X dBZ"
    - Removing extreme outliers
    """
    print("\n" + "="*70)
    print("Example 7: Maximum Threshold Only")
    print("="*70)
    print("Description: Exclude extreme values (possible hail/errors)")
    print("Filter: DBZH <= 55 (no minimum)")
    print("-"*70)
    
    clear_cache()
    
    max_filter = Filter(field="DBZH", min=None, max=55)
    
    print(f"Filter applied: {max_filter}")
    print("\nWhat this does:")
    print("  - Shows all values <= 55 dBZ")
    print("  - Removes extreme values (hail, calibration issues)")
    print("  - Light precipitation still visible")
    
    result = process_radar_to_cog(
        filepath=str(RADAR_FILE),
        product="PPI",
        field_requested="DBZH",
        elevation=0,
        filters=[max_filter],
        output_dir=str(OUTPUT_DIR / "07_max_only")
    )
    
    print(f"\n✓ Output: {result['image_url']}")
    return result


# ============================================================================
# Example 8: CAPPI with Filters
# ============================================================================

def example_cappi_with_filters():
    """
    Apply filters to CAPPI product.
    
    Filters work the same way for all products (PPI, CAPPI, COLMAX).
    This example shows a CAPPI at 3000m with quality filtering.
    """
    print("\n" + "="*70)
    print("Example 8: CAPPI Product with Filters")
    print("="*70)
    print("Description: Constant altitude slice with quality control")
    print("Product: CAPPI at 3000m altitude")
    print("Filters: RHOHV >= 0.85 AND DBZH >= 10")
    print("-"*70)
    
    clear_cache()
    
    filters = [
        Filter(field="RHOHV", min=0.85),
        Filter(field="DBZH", min=10),
    ]
    
    print("Filters applied:")
    for f in filters:
        print(f"  - {f}")
    
    result = process_radar_to_cog(
        filepath=str(RADAR_FILE),
        product="CAPPI",
        field_requested="DBZH",
        cappi_height=3000,  # 3000 meters above radar
        filters=filters,
        output_dir=str(OUTPUT_DIR / "08_cappi_filtered")
    )
    
    print(f"\n✓ Output: {result['image_url']}")
    return result


# ============================================================================
# Example 9: COLMAX with Filters
# ============================================================================

def example_colmax_with_filters():
    """
    Apply filters to COLMAX product.
    
    COLMAX shows the maximum value in each vertical column.
    Filtering helps remove noise and focus on significant echoes.
    """
    print("\n" + "="*70)
    print("Example 9: COLMAX Product with Filters")
    print("="*70)
    print("Description: Column maximum with quality control")
    print("Product: COLMAX (max value in vertical column)")
    print("Filters: RHOHV >= 0.85 AND 20 <= DBZH <= 65")
    print("-"*70)
    
    clear_cache()
    
    filters = [
        Filter(field="RHOHV", min=0.85),
        Filter(field="DBZH", min=20, max=65),
    ]
    
    print("Filters applied:")
    for f in filters:
        print(f"  - {f}")
    
    result = process_radar_to_cog(
        filepath=str(RADAR_FILE),
        product="COLMAX",
        field_requested="DBZH",
        filters=filters,
        output_dir=str(OUTPUT_DIR / "09_colmax_filtered")
    )
    
    print(f"\n✓ Output: {result['image_url']}")
    return result


# ============================================================================
# Example 10: Common Filtering Scenarios
# ============================================================================

def example_scenarios():
    """
    Common real-world filtering scenarios.
    """
    print("\n" + "="*70)
    print("Example 10: Common Filtering Scenarios")
    print("="*70)
    
    scenarios = [
        {
            "name": "General Purpose (Balanced)",
            "description": "Good for most visualizations",
            "filters": [
                Filter("RHOHV", min=0.85),
                Filter("DBZH", min=5, max=65),
            ],
            "output_subdir": "10a_general_purpose"
        },
        {
            "name": "Heavy Precipitation Focus",
            "description": "Highlight storms and heavy rain",
            "filters": [
                Filter("RHOHV", min=0.85),
                Filter("DBZH", min=35),  # Only show 35+ dBZ
            ],
            "output_subdir": "10b_heavy_precip"
        },
        {
            "name": "Light Precipitation Detection",
            "description": "Sensitive to light drizzle/snow",
            "filters": [
                Filter("RHOHV", min=0.90),  # Stricter QC
                # No DBZH filter - show even weak echoes
            ],
            "output_subdir": "10c_light_precip"
        },
        {
            "name": "Clean Display (Maximum Filtering)",
            "description": "Very clean output for presentations",
            "filters": [
                Filter("RHOHV", min=0.92),  # Strict QC
                Filter("DBZH", min=15, max=55),
            ],
            "output_subdir": "10d_clean_display"
        },
    ]
    
    results = []
    for scenario in scenarios:
        print(f"\n--- {scenario['name']} ---")
        print(f"Description: {scenario['description']}")
        print(f"Filters: {scenario['filters']}")
        
        clear_cache()
        
        result = process_radar_to_cog(
            filepath=str(RADAR_FILE),
            product="PPI",
            field_requested="DBZH",
            elevation=0,
            filters=scenario["filters"],
            output_dir=str(OUTPUT_DIR / scenario["output_subdir"])
        )
        
        print(f"✓ Output: {result['image_url']}")
        results.append(result)
    
    return results


# ============================================================================
# Main Function
# ============================================================================

def main():
    """Run all filter examples."""
    print("\n" + "#"*70)
    print("#" + " "*68 + "#")
    print("#" + "    RADAR COG PROCESSOR - FILTER EXAMPLES".center(68) + "#")
    print("#" + " "*68 + "#")
    print("#"*70)
    
    # Check if data file exists
    if not RADAR_FILE.exists():
        print(f"\n❌ Error: Radar file not found: {RADAR_FILE}")
        print("Please ensure you have sample data in data/netcdf/")
        return
    
    ensure_output_dir()
    
    print(f"\nUsing radar file: {RADAR_FILE}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Run all examples
    example_no_filters()
    example_rhohv_filter()
    example_dbzh_range_filter()
    example_combined_filters()
    example_strict_qc()
    example_min_only()
    example_max_only()
    example_cappi_with_filters()
    example_colmax_with_filters()
    example_scenarios()
    
    print("\n" + "="*70)
    print("ALL EXAMPLES COMPLETE")
    print("="*70)
    print(f"\nOutput files saved to: {OUTPUT_DIR}")
    print("\nKey takeaways:")
    print("  1. Use RHOHV filter (min=0.85) to remove non-meteorological echoes")
    print("  2. Use DBZH filters to focus on specific intensity ranges")
    print("  3. Combine filters for best results")
    print("  4. QC filters affect gridding; visual filters affect display only")
    print("="*70)


if __name__ == "__main__":
    main()
