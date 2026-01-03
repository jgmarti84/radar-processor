"""
Products Explained - Detailed explanation of each radar product type.

This example explains and demonstrates the three radar product types:
1. PPI (Plan Position Indicator) - Horizontal slice at an elevation angle
2. CAPPI (Constant Altitude PPI) - Horizontal slice at constant height
3. COLMAX (Column Maximum) - Maximum value in vertical column

Each product type is suited for different meteorological applications.

Run this example:
    python examples/products_explained.py
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from radar_processor import process_radar_to_cog
from radar_processor.cache import GRID2D_CACHE, GRID3D_CACHE


# ============================================================================
# Configuration
# ============================================================================

RADAR_FILE = Path("data/netcdf/RMA1_0315_01_20251208T191648Z.nc")
OUTPUT_DIR = Path("output/products_explained")


class Filter:
    """Simple filter class."""
    def __init__(self, field: str, min: float = None, max: float = None):
        self.field = field
        self.min = min
        self.max = max


# ============================================================================
# PPI - Plan Position Indicator
# ============================================================================

def explain_ppi():
    """
    PPI (Plan Position Indicator)
    =============================
    
    A PPI is a horizontal "slice" of radar data taken at a specific elevation 
    angle. It shows what the radar sees looking in a cone at that angle.
    
    How it works:
    - The radar rotates 360° at a fixed elevation angle (e.g., 0.5°, 1.5°, etc.)
    - At each angle, it sends out a pulse and listens for returns
    - The result is a circular "sweep" of data
    
    Key characteristics:
    - Near the radar: sees low in the atmosphere
    - Far from radar: sees higher due to Earth's curvature and beam elevation
    - The "cone" of the beam widens with distance
    
    Use cases:
    - General precipitation monitoring
    - Real-time weather tracking
    - Quick overview of storm positions
    
    Parameters:
    - elevation: The sweep index (0 = lowest angle, 1 = second lowest, etc.)
    """
    print("\n" + "="*70)
    print("PPI (Plan Position Indicator)")
    print("="*70)
    
    output_dir = OUTPUT_DIR / "PPI"
    
    # Standard quality filter
    filters = [Filter("RHOHV", min=0.85)]
    
    print("\nPPI Concept:")
    print("  • Horizontal slice at a specific elevation angle")
    print("  • Radar scans 360° at that angle")
    print("  • Height varies with distance from radar")
    print()
    
    # Process multiple elevations to show the difference
    elevations = [0, 1, 2]
    
    for elev in elevations:
        print(f"Processing PPI elevation {elev}...")
        
        GRID2D_CACHE.clear()
        GRID3D_CACHE.clear()
        
        try:
            result = process_radar_to_cog(
                filepath=str(RADAR_FILE),
                product="PPI",
                field_requested="DBZH",
                elevation=elev,
                filters=filters,
                output_dir=str(output_dir)
            )
            print(f"  ✓ Created: {Path(result['image_url']).name}")
        except ValueError as e:
            print(f"  ✗ Not available: {e}")
    
    print("\nPPI Tips:")
    print("  • Use elevation 0 for the widest ground-level coverage")
    print("  • Higher elevations see storms aloft better")
    print("  • Compare elevations to understand storm vertical structure")


# ============================================================================
# CAPPI - Constant Altitude Plan Position Indicator
# ============================================================================

def explain_cappi():
    """
    CAPPI (Constant Altitude Plan Position Indicator)
    ==================================================
    
    A CAPPI is a horizontal "slice" of radar data at a constant height above 
    the ground (or sea level). It interpolates from multiple elevation scans 
    to create data at that exact height everywhere.
    
    How it works:
    - Uses data from multiple elevation angles
    - Interpolates between angles to create a "flat" slice
    - The slice is at the same height everywhere on the map
    
    Key characteristics:
    - Consistent height across the entire display
    - Requires 3D grid interpolation
    - May have gaps at edges where no elevation angle covers that height
    
    Use cases:
    - Comparing precipitation across large areas
    - Aviation weather (specific flight levels)
    - Research (consistent height measurements)
    - Automated algorithms that need consistent heights
    
    Parameters:
    - cappi_height: Height in meters above ground/sea level
    
    Common heights:
    - 1500-2000m: Low-level precipitation, rain near ground
    - 3000-4000m: Mid-level precipitation, main rain layer
    - 5000-6000m: Upper precipitation, hail zone in thunderstorms
    """
    print("\n" + "="*70)
    print("CAPPI (Constant Altitude Plan Position Indicator)")
    print("="*70)
    
    output_dir = OUTPUT_DIR / "CAPPI"
    
    filters = [Filter("RHOHV", min=0.85)]
    
    print("\nCAPPI Concept:")
    print("  • Horizontal slice at constant height everywhere")
    print("  • Interpolated from multiple elevation angles")
    print("  • Uses 3D gridding internally")
    print()
    
    # Process multiple heights
    heights = [2000, 3000, 4000, 5000]
    
    for height in heights:
        print(f"Processing CAPPI at {height}m...")
        
        GRID2D_CACHE.clear()
        GRID3D_CACHE.clear()
        
        result = process_radar_to_cog(
            filepath=str(RADAR_FILE),
            product="CAPPI",
            field_requested="DBZH",
            cappi_height=height,
            filters=filters,
            output_dir=str(output_dir)
        )
        print(f"  ✓ Created: {Path(result['image_url']).name}")
    
    print("\nCAPPI Tips:")
    print("  • Use 2000-3000m for general precipitation")
    print("  • Use 4000-5000m to see storms above the melting layer")
    print("  • Compare heights to see vertical distribution")
    print("  • CAPPI processing is slower than PPI (3D interpolation)")


# ============================================================================
# COLMAX - Column Maximum
# ============================================================================

def explain_colmax():
    """
    COLMAX (Column Maximum)
    =======================
    
    COLMAX shows the maximum value found in each vertical column above each 
    point on the ground. It "compresses" all vertical data into a single 2D 
    image showing the strongest returns.
    
    How it works:
    - Creates a 3D grid of the atmosphere
    - For each (x, y) position, finds the maximum value in the vertical
    - Creates a 2D image of these maximum values
    
    Key characteristics:
    - Shows the most intense returns regardless of height
    - Good for detecting severe weather
    - Cannot tell you at what height the maximum occurs
    
    Use cases:
    - Severe weather detection (hail, strong updrafts)
    - Maximum precipitation intensity
    - Storm severity ranking
    - Quick overview of the strongest storms
    
    Parameters:
    - No height/elevation needed - uses entire column
    """
    print("\n" + "="*70)
    print("COLMAX (Column Maximum)")
    print("="*70)
    
    output_dir = OUTPUT_DIR / "COLMAX"
    
    filters = [Filter("RHOHV", min=0.85)]
    
    print("\nCOLMAX Concept:")
    print("  • Maximum value in each vertical column")
    print("  • Shows strongest returns regardless of height")
    print("  • Good for severe weather detection")
    print()
    
    print("Processing COLMAX...")
    
    GRID2D_CACHE.clear()
    GRID3D_CACHE.clear()
    
    result = process_radar_to_cog(
        filepath=str(RADAR_FILE),
        product="COLMAX",
        field_requested="DBZH",
        filters=filters,
        output_dir=str(output_dir)
    )
    print(f"  ✓ Created: {Path(result['image_url']).name}")
    
    print("\nCOLMAX Tips:")
    print("  • High DBZH values (>50) may indicate hail")
    print("  • Use for quick severe storm identification")
    print("  • Compare with CAPPI at different heights to locate the max")


# ============================================================================
# Product Comparison
# ============================================================================

def compare_products():
    """
    Create side-by-side comparison of all products.
    """
    print("\n" + "="*70)
    print("Product Comparison")
    print("="*70)
    
    output_dir = OUTPUT_DIR / "comparison"
    
    filters = [Filter("RHOHV", min=0.85), Filter("DBZH", min=10, max=60)]
    
    comparisons = [
        ("PPI at lowest elevation", {
            "product": "PPI",
            "elevation": 0,
            "cappi_height": 4000
        }),
        ("CAPPI at 3km", {
            "product": "CAPPI",
            "elevation": 0,
            "cappi_height": 3000
        }),
        ("COLMAX (column maximum)", {
            "product": "COLMAX",
            "elevation": 0,
            "cappi_height": 4000
        }),
    ]
    
    print("\nGenerating comparison images...")
    print("All use the same filters for fair comparison.\n")
    
    for name, params in comparisons:
        print(f"  {name}...")
        
        GRID2D_CACHE.clear()
        GRID3D_CACHE.clear()
        
        result = process_radar_to_cog(
            filepath=str(RADAR_FILE),
            product=params["product"],
            field_requested="DBZH",
            elevation=params.get("elevation", 0),
            cappi_height=params.get("cappi_height", 4000),
            filters=filters,
            output_dir=str(output_dir)
        )
        print(f"    ✓ {Path(result['image_url']).name}")
    
    print("\n" + "-"*70)
    print("COMPARISON SUMMARY")
    print("-"*70)
    print("""
┌─────────────────────────────────────────────────────────────────────┐
│ Product   │ Height         │ Use Case                              │
├─────────────────────────────────────────────────────────────────────┤
│ PPI       │ Varies w/range │ General monitoring, real-time ops     │
│ CAPPI     │ Constant       │ Research, aviation, consistent height │
│ COLMAX    │ Maximum column │ Severe weather, storm intensity       │
└─────────────────────────────────────────────────────────────────────┘
    """)


# ============================================================================
# Main
# ============================================================================

def main():
    """Run all product explanations."""
    print("\n" + "#"*70)
    print("#" + " RADAR PRODUCTS EXPLAINED ".center(68, "#") + "#")
    print("#"*70)
    
    if not RADAR_FILE.exists():
        print(f"\n❌ Error: Radar file not found: {RADAR_FILE}")
        return
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Explain each product
    explain_ppi()
    explain_cappi()
    explain_colmax()
    
    # Create comparison
    compare_products()
    
    print("\n" + "="*70)
    print("COMPLETE")
    print("="*70)
    print(f"Output files in: {OUTPUT_DIR}")
    print("\nCheck each product subdirectory to compare the outputs.")


if __name__ == "__main__":
    main()
