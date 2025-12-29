"""
Utility functions for radar data processing.
"""
from __future__ import annotations
from typing import Tuple, Optional, Any
import hashlib
import json
import numpy as np
import pyart

from .constants import FIELD_ALIASES, FIELD_RENDER, AFFECTS_INTERP_FIELDS
from . import colormaps


def md5_file(path: str, chunk: int = 1024 * 1024) -> str:
    """
    Calculate MD5 hash of a file.
    
    Parameters
    ----------
    path : str
        Path to the file
    chunk : int, optional
        Chunk size for reading file, by default 1MB
    
    Returns
    -------
    str
        Hexadecimal MD5 hash
    """
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def stable_hash(obj: Any) -> str:
    """Generate stable hash of a JSON-serializable object."""
    s = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def _roundf(x: float, nd: int = 6) -> float:
    """Round float to specified decimal places."""
    try:
        return float(round(float(x), nd))
    except Exception:
        return float(x)


def _stable(obj: Any):
    """Convert to stable JSON structure (tuples/lists → lists, rounded floats)."""
    if isinstance(obj, dict):
        return {k: _stable(v) for k, v in sorted(obj.items())}
    if isinstance(obj, (list, tuple)):
        return [_stable(v) for v in obj]
    if isinstance(obj, float):
        return _roundf(obj, 6)
    return obj


def _hash_of(payload: Any) -> str:
    """Generate hash of payload for cache keys."""
    s = json.dumps(_stable(payload), separators=(",", ":"), ensure_ascii=False)
    return hashlib.blake2b(s.encode("utf-8"), digest_size=16).hexdigest()


def resolve_field(radar: pyart.core.Radar, requested: str) -> Tuple[str, str]:
    """
    Resolve field name from requested field using aliases.
    
    Parameters
    ----------
    radar : pyart.core.Radar
        PyART Radar object
    requested : str
        Requested field name
    
    Returns
    -------
    Tuple[str, str]
        (field_name_in_file, canonical_field_key)
    
    Raises
    ------
    KeyError
        If field not found or not supported
    """
    key = requested.upper()
    if key not in FIELD_ALIASES:
        raise KeyError(f"Campo no soportado: {requested}")
    for cand in FIELD_ALIASES[key]:
        if cand in radar.fields:
            return cand, key
    raise KeyError(f"No se encontró alias disponible para '{requested}' en el archivo.")


def colormap_for(field_key: str, override_cmap: Optional[str] = None):
    """
    Get colormap and render defaults for a field.
    
    Parameters
    ----------
    field_key : str
        Canonical field key (e.g., 'DBZH')
    override_cmap : str, optional
        Override colormap key
    
    Returns
    -------
    Tuple[object, float, float, str]
        (colormap_object, vmin, vmax, colormap_key)
    """
    spec = FIELD_RENDER.get(field_key.upper(), {"vmin": -30.0, "vmax": 70.0, "cmap": "grc_th"})
    vmin, vmax = spec["vmin"], spec["vmax"]
    cmap_key = override_cmap if override_cmap else spec["cmap"]
    
    # Determine if custom (grc_*) or pyart/matplotlib colormap
    if cmap_key.startswith("grc_"):
        # Custom colormap from colormaps module
        cmap = getattr(colormaps, f"get_cmap_{cmap_key}")()
    elif cmap_key.startswith("pyart_"):
        # PyART colormap (remove prefix)
        cmap = cmap_key.replace("pyart_", "")
    else:
        # Standard matplotlib/pyart colormap
        cmap = cmap_key
    
    return cmap, vmin, vmax, cmap_key


def get_radar_site(radar: pyart.core.Radar) -> Tuple[float, float, float]:
    """
    Get radar site coordinates.
    
    Parameters
    ----------
    radar : pyart.core.Radar
        PyART Radar object
    
    Returns
    -------
    Tuple[float, float, float]
        (longitude, latitude, altitude_m)
    """
    lat = float(np.asarray(radar.latitude["data"]).ravel()[0])
    lon = float(np.asarray(radar.longitude["data"]).ravel()[0])
    alt = 0.0
    try:
        alt = float(np.asarray(radar.altitude["data"]).ravel()[0])
    except Exception:
        pass
    return lon, lat, alt


def safe_range_max_m(radar: pyart.core.Radar, default: float = 240e3) -> float:
    """
    Get maximum range (last gate) in meters with fallback.
    
    Parameters
    ----------
    radar : pyart.core.Radar
        PyART Radar object
    default : float, optional
        Default value if range cannot be determined
    
    Returns
    -------
    float
        Maximum range in meters
    """
    r = radar.range["data"]
    arr = np.asarray(getattr(r, "filled", lambda v: r)(np.nan), dtype=float)
    if arr.size == 0:
        return float(default)
    last = float(arr[-1])
    if np.isfinite(last):
        return last
    # fallback to maximum finite value
    finite = arr[np.isfinite(arr)]
    return float(finite.max()) if finite.size else float(default)


def build_gatefilter(
    radar: pyart.core.Radar,
    field: Optional[str],
    filters: Optional[list] = None,
    is_rhi: Optional[bool] = False
) -> pyart.filters.GateFilter:
    """
    Build a consistent GateFilter for radar data.
    
    Parameters
    ----------
    radar : pyart.core.Radar
        PyART Radar object
    field : str, optional
        Primary field name
    filters : list, optional
        List of filter objects with 'field', 'min', 'max' attributes
    is_rhi : bool, optional
        Whether processing RHI data
    
    Returns
    -------
    pyart.filters.GateFilter
        Configured gate filter
    """
    gf = pyart.filters.GateFilter(radar)
    try:
        gf.exclude_transition()
    except Exception:
        pass

    if field in radar.fields:
        try:
            gf.exclude_invalid(field)
            gf.exclude_masked(field)
        except Exception:
            pass

    for f in (filters or []):
        fld = getattr(f, "field", None)
        if not fld:
            continue
        # Apply filters for QC fields or all fields if RHI
        if fld in radar.fields and (fld in AFFECTS_INTERP_FIELDS or (is_rhi and fld == field)):
            fmin = getattr(f, "min", None)
            fmax = getattr(f, "max", None)
            if fmin is not None:
                if fmin <= 0.3 and fld in AFFECTS_INTERP_FIELDS:
                    continue
                else:
                    gf.exclude_below(fld, float(fmin))
            if fmax is not None:
                gf.exclude_above(fld, float(fmax))
    return gf


def qc_signature(filters):
    """
    Generate QC signature from filters for cache key.
    
    Only filters that affect interpolation are included.
    """
    sig = []
    for f in (filters or []):
        ffield = getattr(f, "field", None)
        if not ffield:
            continue
        up = str(ffield).upper()
        if up in AFFECTS_INTERP_FIELDS:
            sig.append((up, getattr(f, "min", None), getattr(f, "max", None)))
    return tuple(sig)


def grid2d_cache_key(*, file_hash, product_upper, field_to_use,
                     elevation, cappi_height, volume,
                     interp, qc_sig) -> str:
    """Generate cache key for 2D grid."""
    payload = {
        "v": 1,  # version
        "file": file_hash,
        "prod": product_upper,
        "field": str(field_to_use).upper(),
        "elev": float(elevation) if elevation is not None else None,
        "h": int(cappi_height) if cappi_height is not None else None,
        "vol": str(volume) if volume is not None else None,
        "interp": str(interp),
        "qc": list(qc_sig) if isinstance(qc_sig, (list, tuple)) else qc_sig,
    }
    return "g2d_" + _hash_of(payload)


def grid3d_cache_key(*, file_hash: str, field_to_use: str,
                     volume: str | None, qc_sig, grid_res_xy: float,
                     grid_res_z: float, z_top_m: float) -> str:
    """Generate cache key for 3D grid."""
    payload = {
        "v": 1,
        "file": file_hash,
        "field": str(field_to_use).upper(),
        "vol": str(volume) if volume is not None else None,
        "qc": list(qc_sig) if isinstance(qc_sig, (list, tuple)) else qc_sig,
        "gxy": float(grid_res_xy),
        "gz": float(grid_res_z),
        "ztop": float(z_top_m),
    }
    return "g3d_" + _hash_of(payload)


def normalize_proj_dict(grid, grid_origin):
    """
    Convert PyART projection dict to pyproj-compatible format.
    
    Parameters
    ----------
    grid : pyart.core.Grid
        PyART Grid object
    grid_origin : tuple
        (latitude, longitude) of grid origin
    
    Returns
    -------
    dict
        Normalized projection dictionary
    """
    proj = dict(getattr(grid, "projection", {}) or {})
    if not proj:
        proj = dict(grid.get_projparams() or {})

    # Fallback to radar origin if lat/lon missing
    lat0 = float(proj.get("lat_0", grid_origin[0]))
    lon0 = float(proj.get("lon_0", grid_origin[1]))

    # PyART uses "pyart_aeqd" as internal alias; PROJ wants "aeqd"
    if proj.get("proj") in ("pyart_aeqd", None):
        proj = {
            "proj": "aeqd",
            "lat_0": lat0,
            "lon_0": lon0,
            "datum": "WGS84",
            "units": "m",
        }
    else:
        # Ensure reasonable units/datum
        proj.setdefault("datum", "WGS84")
        proj.setdefault("units", "m")

    # Remove 'type' key that can cause issues
    proj.pop("type", None)
    return proj


def collapse_field_3d_to_2d(data3d, product, *,
                            x_coords=None, y_coords=None, z_levels=None,
                            elevation_deg=None, target_height_m=None):
    """
    Collapse 3D field data to 2D based on product type.
    
    Non-destructive version that doesn't modify the Grid object.
    
    Parameters
    ----------
    data3d : np.ndarray
        3D data array (nz, ny, nx)
    product : str
        Product type ('ppi', 'cappi', 'colmax')
    x_coords : np.ndarray, optional
        X coordinates
    y_coords : np.ndarray, optional
        Y coordinates
    z_levels : np.ndarray, optional
        Z levels
    elevation_deg : float, optional
        Elevation angle for PPI
    target_height_m : float, optional
        Target height for CAPPI
    
    Returns
    -------
    np.ma.MaskedArray
        2D masked array
    """
    if data3d.ndim == 2:
        arr2d = data3d
    else:
        if product == "ppi":
            assert elevation_deg is not None and x_coords is not None and y_coords is not None and z_levels is not None
            X, Y = np.meshgrid(x_coords, y_coords, indexing='xy')
            r = np.sqrt(X**2 + Y**2)
            Re = 8.49e6  # 4/3 Earth radius
            z_target = r * np.sin(np.deg2rad(elevation_deg)) + (r**2) / (2.0 * Re)
            iz = np.abs(z_target[..., None] - z_levels[None, None, :]).argmin(axis=2)
            yy = np.arange(len(y_coords))[:, None]
            xx = np.arange(len(x_coords))[None, :]
            arr2d = data3d[iz, yy, xx]
        elif product == "cappi":
            assert target_height_m is not None and z_levels is not None
            iz = np.abs(z_levels - float(target_height_m)).argmin()
            arr2d = data3d[iz, :, :]
        elif product == "colmax":
            arr2d = data3d.max(axis=0)
        else:
            raise ValueError("Producto inválido")
    return np.ma.array(arr2d.astype(np.float32), mask=np.ma.getmaskarray(arr2d))
