"""
Fields Explained - Comprehensive guide to radar measurement fields.

This example explains all the radar measurement types (fields) available
and their meteorological significance.

Radar Fields:
- DBZH: Horizontal Reflectivity (precipitation intensity)
- DBZV: Vertical Reflectivity (less common)
- ZDR: Differential Reflectivity (drop shape/size)
- RHOHV: Cross-Correlation Coefficient (data quality/hydrometeor type)
- KDP: Specific Differential Phase (rain rate, hail detection)
- VRAD: Radial Velocity (motion toward/away from radar)
- WRAD: Spectrum Width (turbulence, wind shear)
- PHIDP: Differential Phase (cumulative phase shift)

Run this example:
    python examples/fields_explained.py
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
OUTPUT_DIR = Path("output/fields_explained")


# ============================================================================
# Field Definitions
# ============================================================================

FIELD_EXPLANATIONS = {
    "DBZH": {
        "full_name": "Horizontal Reflectivity",
        "unit": "dBZ (decibels of Z)",
        "description": """
DBZH (Horizontal Reflectivity) is the PRIMARY radar measurement for 
precipitation. It measures how much of the radar pulse is reflected back
by precipitation particles.

How it works:
- Radar sends horizontal polarized pulses
- Particles (rain, snow, hail) scatter energy back
- Larger particles reflect more energy

Typical Values:
  • < 10 dBZ: Light drizzle, mist
  • 10-25 dBZ: Light rain, light snow
  • 25-40 dBZ: Moderate rain, wet snow
  • 40-50 dBZ: Heavy rain
  • 50-60 dBZ: Very heavy rain, small hail
  • > 60 dBZ: Large hail, severe storms

Use Cases:
  • Precipitation detection and intensity
  • Storm tracking
  • Rainfall estimation (with rain rate equations)
  • General weather monitoring
""",
    },
    
    "DBZV": {
        "full_name": "Vertical Reflectivity",
        "unit": "dBZ (decibels of Z)",
        "description": """
DBZV (Vertical Reflectivity) is similar to DBZH but uses vertically 
polarized radar pulses. It's less commonly used alone but is essential 
for calculating dual-pol products.

How it differs from DBZH:
- Measures vertical dimension of particles
- Combined with DBZH gives information about particle shape
- Usually similar to DBZH for spherical particles

Use Cases:
  • Calculating ZDR (differential reflectivity)
  • Quality control
  • Research applications
""",
    },
    
    "ZDR": {
        "full_name": "Differential Reflectivity",
        "unit": "dB (decibels)",
        "description": """
ZDR (Differential Reflectivity) is the DIFFERENCE between horizontal and 
vertical reflectivity. It tells you about particle SHAPE.

Formula: ZDR = DBZH - DBZV

What it reveals:
- Spherical particles (like small drops): ZDR ≈ 0
- Oblate particles (flattened drops): ZDR > 0
- Prolate particles (elongated): ZDR < 0

Typical Values:
  • 0 dB: Spherical particles (drizzle, tumbling hail)
  • 0-1 dB: Small raindrops
  • 1-3 dB: Moderate rain (drops flatten as they fall)
  • 3-5 dB: Large raindrops
  • > 5 dB: Very large drops, melting hail
  • < 0 dB: Vertically oriented ice crystals

Use Cases:
  • Distinguishing rain from hail
  • Drop size estimation
  • Hydrometeor classification
  • Melting layer detection
""",
    },
    
    "RHOHV": {
        "full_name": "Cross-Correlation Coefficient",
        "unit": "dimensionless (0-1)",
        "description": """
RHOHV (Cross-Correlation Coefficient) measures the CORRELATION between 
horizontal and vertical radar returns. It's primarily used for DATA QUALITY 
and hydrometeor classification.

What it means:
- High RHOHV (>0.95): Uniform particles (pure rain, pure snow)
- Medium RHOHV (0.85-0.95): Mixed precipitation
- Low RHOHV (<0.85): Non-meteorological targets or mixed particles

Typical Values:
  • > 0.97: Pure rain (all similar drops)
  • 0.92-0.97: Light rain, dry snow
  • 0.85-0.92: Mixed phase, wet snow, small hail
  • 0.70-0.85: Large hail, melting layer
  • < 0.70: Biological targets (birds, insects), ground clutter

USE AS A QUALITY FILTER:
This is the MOST IMPORTANT field for quality control. By filtering out 
low RHOHV values, you remove:
  • Ground clutter (buildings, mountains)
  • Biological targets (bird migrations)
  • Chaff and debris
  • Anomalous propagation

Recommended Filters:
  • Permissive: RHOHV >= 0.80
  • Standard: RHOHV >= 0.85
  • Strict: RHOHV >= 0.92
""",
    },
    
    "KDP": {
        "full_name": "Specific Differential Phase",
        "unit": "deg/km (degrees per kilometer)",
        "description": """
KDP (Specific Differential Phase) measures the RATE OF CHANGE of the 
differential phase along the radar beam. It's excellent for rain rate 
estimation because it's unaffected by attenuation.

What it measures:
- Phase difference between H and V polarizations
- Increases in rain (oblate drops slow H more than V)
- Proportional to water content

Why it's useful:
- Not affected by attenuation (works in heavy rain)
- Not affected by radar calibration errors
- More linear relationship with rain rate than DBZH
- Can detect heavy rain behind hail

Typical Values:
  • 0 deg/km: No rain or dry snow
  • 0-1 deg/km: Light to moderate rain
  • 1-3 deg/km: Moderate to heavy rain
  • 3-6 deg/km: Heavy rain
  • > 6 deg/km: Very heavy rain, possible hail

Use Cases:
  • Rainfall rate estimation (R = f(KDP))
  • Heavy rain detection
  • Hail detection (high DBZH + low KDP = hail)
  • Distinguishing rain from hail
""",
    },
    
    "VRAD": {
        "full_name": "Radial Velocity",
        "unit": "m/s (meters per second)",
        "description": """
VRAD (Radial Velocity) measures the MOTION of precipitation particles 
toward or away from the radar using the Doppler effect.

How it works:
- Doppler shift of returning pulse
- Motion toward radar: positive (or negative, depends on convention)
- Motion away from radar: opposite sign

What it shows:
- Wind patterns within storms
- Storm rotation (tornado signatures)
- Wind shear
- Storm motion

Typical Patterns:
  • Uniform colors: Steady wind field
  • Radial patterns: Divergence/convergence
  • Couplets (adjacent +/-): Rotation, possible tornado
  • Sharp boundaries: Wind shear, fronts

Use Cases:
  • Tornado detection (velocity couplets)
  • Wind field analysis
  • Storm motion estimation
  • Wind shear detection for aviation
  • Atmospheric research
""",
    },
    
    "WRAD": {
        "full_name": "Spectrum Width",
        "unit": "m/s (meters per second)",
        "description": """
WRAD (Spectrum Width) measures the VARIATION in velocities within a 
single radar sample volume. High values indicate turbulence or wind shear.

What it measures:
- Standard deviation of velocities in the sample volume
- Variability of particle motion

High values indicate:
- Turbulence
- Wind shear
- Tornado circulation
- Multiple particle types with different fall speeds

Typical Values:
  • 0-2 m/s: Uniform, laminar flow
  • 2-4 m/s: Moderate turbulence, light shear
  • 4-8 m/s: Significant turbulence, strong shear
  • > 8 m/s: Severe turbulence, tornado circulation

Use Cases:
  • Turbulence detection for aviation
  • Tornado signature enhancement
  • Identifying storm boundaries
  • Wind shear detection
""",
    },
    
    "PHIDP": {
        "full_name": "Differential Phase",
        "unit": "degrees",
        "description": """
PHIDP (Differential Phase) measures the CUMULATIVE phase shift between 
horizontal and vertical polarizations along the radar beam.

How it differs from KDP:
- PHIDP: Cumulative (increases through rain)
- KDP: Rate of change (derivative of PHIDP)

Typical characteristics:
- Starts at system offset (varies by radar)
- Increases through rain
- Can be noisy, especially in light rain
- Used to calculate KDP

Typical Values:
  • Varies widely based on path through precipitation
  • Can range from 0° to over 180°
  • Usually smoothed before use

Use Cases:
  • Calculating KDP
  • Attenuation correction
  • Research applications
  • Rarely displayed directly
""",
    },
}


# ============================================================================
# Processing Functions
# ============================================================================

def explain_field(field_name: str, field_info: dict):
    """Display explanation and process a field."""
    print("\n" + "="*70)
    print(f"{field_name}: {field_info['full_name']}")
    print("="*70)
    print(f"Unit: {field_info['unit']}")
    print(field_info['description'])
    
    # Try to process this field
    output_dir = OUTPUT_DIR / field_name
    
    GRID2D_CACHE.clear()
    GRID3D_CACHE.clear()
    
    try:
        print(f"Processing {field_name}...")
        result = process_radar_to_cog(
            filepath=str(RADAR_FILE),
            product="PPI",
            field_requested=field_name,
            elevation=0,
            filters=None,  # No filters to show raw data
            output_dir=str(output_dir)
        )
        print(f"✓ Created: {Path(result['image_url']).name}")
    except (KeyError, ValueError) as e:
        print(f"✗ Field not available in this file: {e}")


def create_field_comparison():
    """Create comparison images of key fields with and without filters."""
    print("\n" + "="*70)
    print("FIELD COMPARISON: DBZH with Different Quality Filters")
    print("="*70)
    
    output_dir = OUTPUT_DIR / "comparison"
    
    class Filter:
        def __init__(self, field, min=None, max=None):
            self.field = field
            self.min = min
            self.max = max
    
    comparisons = [
        ("DBZH_no_filter", None),
        ("DBZH_rhohv_80", [Filter("RHOHV", min=0.80)]),
        ("DBZH_rhohv_85", [Filter("RHOHV", min=0.85)]),
        ("DBZH_rhohv_92", [Filter("RHOHV", min=0.92)]),
    ]
    
    for name, filters in comparisons:
        GRID2D_CACHE.clear()
        GRID3D_CACHE.clear()
        
        print(f"\nProcessing {name}...")
        result = process_radar_to_cog(
            filepath=str(RADAR_FILE),
            product="PPI",
            field_requested="DBZH",
            elevation=0,
            filters=filters,
            output_dir=str(output_dir)
        )
        print(f"  ✓ {Path(result['image_url']).name}")
    
    print("\n" + "-"*70)
    print("Notice how RHOHV filtering removes non-meteorological echoes:")
    print("  • No filter: May show ground clutter, birds, etc.")
    print("  • RHOHV >= 0.80: Removes worst non-met echoes")
    print("  • RHOHV >= 0.85: Good balance for general use")
    print("  • RHOHV >= 0.92: Strictest, only pure precipitation")
    print("-"*70)


# ============================================================================
# Main
# ============================================================================

def main():
    """Run field explanations."""
    print("\n" + "#"*70)
    print("#" + " RADAR FIELDS EXPLAINED ".center(68, "#") + "#")
    print("#"*70)
    
    if not RADAR_FILE.exists():
        print(f"\n❌ Error: Radar file not found: {RADAR_FILE}")
        return
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("\nThis example explains each radar field and its meteorological use.")
    print("Each field provides different information about precipitation.\n")
    
    # Explain each field
    for field_name, field_info in FIELD_EXPLANATIONS.items():
        explain_field(field_name, field_info)
    
    # Create comparison
    create_field_comparison()
    
    print("\n" + "="*70)
    print("COMPLETE")
    print("="*70)
    print(f"\nOutput files organized in: {OUTPUT_DIR}")
    print("\nKey Takeaways:")
    print("  • DBZH: Primary precipitation intensity")
    print("  • RHOHV: Use as quality control filter (min 0.85)")
    print("  • ZDR: Drop shape (distinguishes rain/hail)")
    print("  • KDP: Reliable rain rate (unaffected by attenuation)")
    print("  • VRAD: Wind/storm motion")
    print("="*70)


if __name__ == "__main__":
    main()
