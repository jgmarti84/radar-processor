"""
Constants for radar field definitions, colormaps, and rendering parameters.
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
FIELD_RENDER = {
    "DBZH": {"vmin": -30.0, "vmax": 70.0, "cmap": "grc_th"},
    "DBZHF": {"vmin": -30.0, "vmax": 70.0, "cmap": "grc_th"},
    "DBZV": {"vmin": -30.0, "vmax": 70.0, "cmap": "grc_th"},
    "ZDR": {"vmin": -5.0, "vmax": 10.5, "cmap": "grc_zdr2"},
    "RHOHV": {"vmin": 0.0, "vmax": 1.0, "cmap": "grc_rho"},
    "KDP": {"vmin": 0.0, "vmax": 8.0, "cmap": "grc_rain"},
    "VRAD": {"vmin": -35.0, "vmax": 35.0, "cmap": "NWSVel"},
    "WRAD": {"vmin": 0.0, "vmax": 10.0, "cmap": "Oranges"},
    "PHIDP": {"vmin": 0.0, "vmax": 360.0, "cmap": "Theodore16"},
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
    "ZDR": ["grc_zdr2", "grc_zdr", "pyart_RefDiff", "pyart_Theodore16"],
    "RHOHV": ["grc_rho", "pyart_RefDiff", "Greys", "viridis"],
    "KDP": ["grc_rain", "grc_th", "pyart_Theodore16", "plasma"],
    "VRAD": ["NWSVel", "pyart_BuDRd18", "seismic", "RdBu_r"],
    "WRAD": ["Oranges", "YlOrRd", "hot", "plasma"],
    "PHIDP": ["Theodore16", "hsv", "twilight", "twilight_shifted"],
}

# Fields that affect interpolation (QC fields)
AFFECTS_INTERP_FIELDS = {"RHOHV"}
