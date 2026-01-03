"""
Constants for radar field definitions, colormaps, and rendering parameters.

This file has been synchronized with radarlib's configuration to ensure
PNG and COG outputs use identical rendering parameters.

Source: radarlib/src/radarlib/config.py
"""

# Field name aliases for different radar data formats
FIELD_ALIASES = {
    "DBZH": ["DBZH", "corrected_reflectivity_horizontal"],  # Horizontal reflectivity
    "DBZV": ["DBZV", "corrected_reflectivity_vertical"],    # Vertical reflectivity
    "DBZHF": ["DBZHF", "DBZHf"],
    "ZDR": ["ZDR", "zdr"],                                  # Differential reflectivity
    "RHOHV": ["RHOHV", "rhohv"],                            # Cross-correlation coefficient
    "KDP": ["KDP", "kdp"],                                  # Specific differential phase
    "VRAD": ["VRAD", "velocity", "corrected_velocity"],     # Radial velocity
    "WRAD": ["WRAD", "spectrum_width"],                     # Spectrum width
    "PHIDP": ["PHIDP", "differential_phase"],               # Differential phase
}

# Default rendering parameters for each field
# Updated to match radarlib's NOFILTERS configuration
FIELD_RENDER = {
    "DBZH": {"vmin": -20.0, "vmax": 70.0, "cmap": "grc_th"},
    "DBZHF": {"vmin": -20.0, "vmax": 70.0, "cmap": "grc_th"},
    "DBZV": {"vmin": -20.0, "vmax": 70.0, "cmap": "grc_th"},
    "ZDR": {"vmin": -7.5, "vmax": 7.5, "cmap": "grc_zdr"},
    "RHOHV": {"vmin": 0.0, "vmax": 1.0, "cmap": "grc_rho"},
    "KDP": {"vmin": -4.0, "vmax": 8.0, "cmap": "jet"},
    "VRAD": {"vmin": -30.0, "vmax": 30.0, "cmap": "grc_vrad"},
    "WRAD": {"vmin": -2.0, "vmax": 6.0, "cmap": "grc_th"},
    "PHIDP": {"vmin": -5.0, "vmax": 360.0, "cmap": "grc_th"},
}

# Variable units
VARIABLE_UNITS = {
    "WRAD": "m/s",
    "KDP": "deg/km",
    "DBZV": "dBZ",
    "DBZH": "dBZ",
    "ZDR": "dBZ",
    "VRAD": "m/s",
    "RHOHV": "",
    "PHIDP": "deg",
}

# Available colormap options by field
FIELD_COLORMAP_OPTIONS = {
    "DBZH": ["grc_th", "grc_th2", "grc_rain", "pyart_NWSRef", "pyart_HomeyerRainbow"],
    "DBZHF": ["grc_th", "grc_th2", "grc_rain", "pyart_NWSRef", "pyart_HomeyerRainbow"],
    "DBZV": ["grc_th", "grc_th2", "grc_rain", "pyart_NWSRef", "pyart_HomeyerRainbow"],
    "ZDR": ["grc_zdr", "grc_zdr2", "pyart_RefDiff", "pyart_Theodore16"],
    "RHOHV": ["grc_rho", "pyart_RefDiff", "Greys", "viridis"],
    "KDP": ["jet", "grc_rain", "grc_th", "pyart_Theodore16", "plasma"],
    "VRAD": ["grc_vrad", "NWSVel", "pyart_BuDRd18", "seismic", "RdBu_r"],
    "WRAD": ["grc_th", "Oranges", "YlOrRd", "hot", "plasma"],
    "PHIDP": ["grc_th", "Theodore16", "hsv", "twilight", "twilight_shifted"],
}

# Fields that affect interpolation (QC fields)
AFFECTS_INTERP_FIELDS = {"RHOHV"}
