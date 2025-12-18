"""
Core processor module for converting radar NetCDF files to Cloud-Optimized GeoTIFFs.
"""
import math
import os
import pyart
import uuid
import pyproj
import numpy as np
from pathlib import Path
import rasterio
from rasterio.enums import ColorInterp
from rasterio.warp import Resampling
from affine import Affine

from .cache import GRID2D_CACHE, GRID3D_CACHE
from .constants import AFFECTS_INTERP_FIELDS
from .utils import (
    md5_file,
    resolve_field,
    build_gatefilter,
    colormap_for,
    qc_signature,
    grid2d_cache_key,
    grid3d_cache_key,
    normalize_proj_dict,
    safe_range_max_m,
    collapse_field_3d_to_2d
)


def _get_or_build_grid3d(
    radar_to_use: pyart.core.Radar,
    field_to_use: str,
    file_hash: str,
    volume: str | None,
    qc_filters,
    z_grid_limits: tuple,
    y_grid_limits: tuple,
    x_grid_limits: tuple,
    grid_resolution: float,
) -> pyart.core.Grid:
    """
    Get or build cached 3D grid.
    
    Used by both process_radar_to_cog and build_3d_grid functions.
    
    Parameters
    ----------
    radar_to_use : pyart.core.Radar
        Radar object to grid
    field_to_use : str
        Field name to grid
    file_hash : str
        Hash of source file for cache key
    volume : str, optional
        Volume identifier
    qc_filters : list
        QC filters to apply
    z_grid_limits : tuple
        (z_min, z_max) in meters
    y_grid_limits : tuple
        (y_min, y_max) in meters
    x_grid_limits : tuple
        (x_min, x_max) in meters
    grid_resolution : float
        Grid resolution in meters
    
    Returns
    -------
    pyart.core.Grid
        3D grid object
    """
    # Generate cache key
    qc_sig = qc_signature(qc_filters)
    cache_key = grid3d_cache_key(
        file_hash=file_hash,
        field_to_use=field_to_use,
        volume=volume,
        qc_sig=qc_sig,
        grid_res_xy=grid_resolution,
        grid_res_z=grid_resolution,
        z_top_m=z_grid_limits[1],
    )
    
    # Check 3D cache
    pkg_cached = GRID3D_CACHE.get(cache_key)
    
    if pkg_cached is not None:
        # Reconstruct Grid from cache with all field metadata
        cached_field_name = pkg_cached.get("field_name", field_to_use)
        field_metadata = pkg_cached.get("field_metadata", {})
        
        # Restore complete field with all metadata
        field_dict = field_metadata.copy()
        field_dict['data'] = pkg_cached["arr3d"]
        
        # Ensure minimum metadata if not in cache
        if 'units' not in field_dict:
            field_dict['units'] = 'unknown'
        if '_FillValue' not in field_dict:
            field_dict['_FillValue'] = -9999.0
        if 'long_name' not in field_dict:
            field_dict['long_name'] = cached_field_name
        
        # Create Grid with complete metadata including time['units']
        grid = pyart.core.Grid(
            time={
                'data': np.array([0]),
                'units': 'seconds since 2000-01-01T00:00:00Z',
                'calendar': 'gregorian',
                'standard_name': 'time'
            },
            fields={cached_field_name: field_dict},
            metadata={'instrument_name': 'RADAR'},
            origin_latitude={'data': radar_to_use.latitude['data']},
            origin_longitude={'data': radar_to_use.longitude['data']},
            origin_altitude={'data': radar_to_use.altitude['data']},
            x={'data': pkg_cached["x"]},
            y={'data': pkg_cached["y"]},
            z={'data': pkg_cached["z"]},
        )
        grid.projection = pkg_cached["projection"]
        return grid
    
    # Build 3D grid from radar
    gf = build_gatefilter(radar_to_use, field_to_use, qc_filters, is_rhi=False)
    
    grid_origin = (
        float(radar_to_use.latitude['data'][0]),
        float(radar_to_use.longitude['data'][0]),
    )
    
    range_max_m = (y_grid_limits[1] - y_grid_limits[0]) / 2
    constant_roi = max(
        grid_resolution * 1.5,
        800 + (range_max_m / 100000) * 400
    )
    
    z_points = int(np.ceil(z_grid_limits[1] / grid_resolution)) + 1
    y_points = int((y_grid_limits[1] - y_grid_limits[0]) / grid_resolution)
    x_points = int((x_grid_limits[1] - x_grid_limits[0]) / grid_resolution)
    
    # Fields to include in grid: primary + all available QC fields
    fields_for_grid = {field_to_use}
    for qc_name in AFFECTS_INTERP_FIELDS:
        if qc_name in radar_to_use.fields:
            fields_for_grid.add(qc_name)
    fields_for_grid = list(fields_for_grid)
    
    grid = pyart.map.grid_from_radars(
        radar_to_use,
        grid_shape=(z_points, y_points, x_points),
        grid_limits=(z_grid_limits, y_grid_limits, x_grid_limits),
        gridding_algo="map_gates_to_grid",
        grid_origin=grid_origin,
        fields=fields_for_grid,
        weighting_function='nearest',
        gatefilters=gf,
        roi_func="constant",
        constant_roi=constant_roi,
    )
    grid.to_xarray()
    
    # Cache complete 3D grid before collapsing
    # Save all field metadata except 'data'
    field_metadata = {k: v for k, v in grid.fields[field_to_use].items() if k != 'data'}
    
    pkg_to_cache = {
        "arr3d": grid.fields[field_to_use]['data'].copy(),
        "x": grid.x['data'].copy(),
        "y": grid.y['data'].copy(),
        "z": grid.z['data'].copy(),
        "projection": dict(getattr(grid, "projection", {}) or {}),
        "field_name": field_to_use,
        "field_metadata": field_metadata,  # Include all metadata (units, long_name, etc.)
    }
    GRID3D_CACHE[cache_key] = pkg_to_cache
    
    return grid


def convert_to_cog(src_path, cog_path):
    """
    Convert existing GeoTIFF to Cloud-Optimized GeoTIFF (COG).
    
    Parameters
    ----------
    src_path : str or Path
        Path to source GeoTIFF
    cog_path : str or Path
        Path for output COG
    
    Returns
    -------
    Path
        Path to created COG file
    """
    with rasterio.open(src_path) as src:
        # Copy original profile and adjust for COG
        profile = src.profile.copy()
        profile.update(
            driver="COG",
            compress="DEFLATE",
            predictor=2,
            BIGTIFF="IF_NEEDED",
            photometric="RGB",
            tiled=True
        )
        profile["band_descriptions"] = ["Red", "Green", "Blue", "Alpha"]

        # Create COG file
        with rasterio.open(cog_path, "w+", **profile) as dst:
            # Copy bands directly without reprojection
            for i in range(1, src.count + 1):
                data = src.read(i)
                dst.write(data, i)
            
            # Define color interpretations
            dst.colorinterp = (
                ColorInterp.red,
                ColorInterp.green,
                ColorInterp.blue,
                ColorInterp.alpha
            )

            # Generate overview pyramids for fast navigation
            factors = [2, 4, 8, 16]
            dst.build_overviews(factors, Resampling.nearest)
            dst.update_tags(ns="rio_overview", resampling="nearest")

    return cog_path


def create_colmax(radar):
    """
    Create composite reflectivity field (COLMAX) from all elevations.
    
    Parameters
    ----------
    radar : pyart.core.Radar
        Radar object with multiple elevations
    
    Returns
    -------
    pyart.core.Radar
        Radar object with composite_reflectivity field
    """
    compz = pyart.retrieve.composite_reflectivity(radar, field="filled_DBZH")

    # Change long_name for figure title
    compz.fields['composite_reflectivity']['long_name'] = 'COLMAX'

    # Re-mask before export
    data = compz.fields['composite_reflectivity']['data']
    mask = np.isnan(data) | np.isclose(data, -30) | (data < -40)
    compz.fields['composite_reflectivity']['data'] = np.ma.array(data, mask=mask)
    compz.fields['composite_reflectivity']['_FillValue'] = -9999.0

    return compz


def beam_height_max_km(range_max_m, elev_deg, antenna_alt_m=0.0):
    """
    Calculate maximum beam height in km for given range and elevation.
    
    Parameters
    ----------
    range_max_m : float
        Maximum range in meters
    elev_deg : float
        Elevation angle in degrees
    antenna_alt_m : float, optional
        Antenna altitude in meters
    
    Returns
    -------
    float
        Maximum beam height in km
    """
    Re = 8.49e6  # 4/3 Earth radius in meters
    r = float(range_max_m)
    th = math.radians(float(elev_deg))
    h = r * math.sin(th) + (r * r) / (2.0 * Re) + antenna_alt_m
    return h / 1000.0  # km


def collapse_grid_to_2d(grid, field, product, *,
                        elevation_deg=None,
                        target_height_m=None,
                        vmin=-30.0):
    """
    Collapse 3D grid to 2D based on product type.
    
    Modifies the grid object in-place.
    
    Parameters
    ----------
    grid : pyart.core.Grid
        Grid object to collapse
    field : str
        Field name to collapse
    product : str
        Product type ('ppi', 'cappi', 'colmax')
    elevation_deg : float, optional
        Elevation angle for PPI
    target_height_m : float, optional
        Target height for CAPPI
    vmin : float, optional
        Minimum value for masking
    """
    data3d = grid.fields[field]['data']
    z = grid.z['data']
    x = grid.x['data']
    y = grid.y['data']
    ny, nx = len(y), len(x)

    if data3d.ndim == 2:  # Already 2D (rare)
        arr2d = data3d
    else:
        if product == "ppi":
            # Follow beam curvature for selected sweep
            assert elevation_deg is not None
            # Calculate horizontal distance r from radar to each pixel
            X, Y = np.meshgrid(x, y, indexing='xy')
            r = np.sqrt(X**2 + Y**2)
            Re = 8.49e6  # 4/3 Earth radius

            # Target beam height at each pixel
            z_target = r * np.sin(np.deg2rad(elevation_deg)) + (r**2) / (2.0 * Re)

            # Find closest z index for each pixel
            iz = np.abs(z_target[..., None] - z[None, None, :]).argmin(axis=2)

            # Extract values at those z indices
            yy = np.arange(ny)[:, None]
            xx = np.arange(nx)[None, :]
            arr2d = data3d[iz, yy, xx]

        elif product == "cappi":
            assert target_height_m is not None
            iz = np.abs(z - float(target_height_m)).argmin()
            arr2d = data3d[iz, :, :]
        elif product == "colmax":
            arr2d = data3d.max(axis=0)
        else:
            raise ValueError("Producto inválido")

    # Re-mask
    arr2d = np.ma.masked_invalid(arr2d)
    if field in ["filled_DBZH", "DBZH", "DBZV", "DBZHF", "composite_reflectivity", "cappi"]:
        arr2d = np.ma.masked_less_equal(arr2d, vmin)
    elif field in ["KDP", "ZDR"]:
        arr2d = np.ma.masked_less(arr2d, vmin)

    # Write as single level
    grid.fields[field]['data'] = arr2d[np.newaxis, ...]  # (1, ny, nx)
    grid.fields[field]['_FillValue'] = -9999.0
    grid.z['data'] = np.array([0.0], dtype=float)


def create_cappi(radar, fields, height):
    """
    Create CAPPI (Constant Altitude Plan Position Indicator) field.
    
    Parameters
    ----------
    radar : pyart.core.Radar
        Radar object
    fields : list
        List of field names to process
    height : float
        Target height in meters
    
    Returns
    -------
    pyart.core.Radar
        Radar object with CAPPI field
    """
    # Simple implementation - interpolates to constant height
    # This is a placeholder; full implementation would use more sophisticated interpolation
    field_name = fields[0] if fields else "DBZH"
    
    # Create placeholder CAPPI field with same shape as first sweep
    template_shape = radar.fields[field_name]['data'].shape
    cappi_data = np.ma.zeros(template_shape)
    cappi_data.mask = True  # Initially all masked
    
    radar.add_field_like(field_name, 'cappi', cappi_data, replace_existing=True)
    return radar


def _build_processing_config(
    filepath,
    product,
    field_requested,
    elevation,
    cappi_height,
    volume,
    colormap_overrides,
    filters,
    output_dir,
):
    """
    Phase 1 & 2: Setup, validation, and caching configuration.
    
    Generates unique identifiers, validates inputs, and returns early if cached.
    
    Parameters
    ----------
    filepath : str or Path
        Radar file path
    product : str
        Product type (PPI, CAPPI, COLMAX)
    field_requested : str
        Field name requested
    elevation : int
        Elevation angle index
    cappi_height : int
        CAPPI height in meters
    volume : str, optional
        Volume identifier
    colormap_overrides : dict, optional
        Colormap overrides
    filters : list
        Filter list
    output_dir : str
        Output directory
    
    Returns
    -------
    dict
        Config dict with: file_hash, cog_path, summary, field_name, field_key,
        cmap, vmin, vmax, product_upper, qc_filters, visual_filters
    """
    file_hash = md5_file(filepath)[:12]
    filters_str = "_".join([f"{f.field}_{f.min}_{f.max}" for f in filters]) if filters else "nofilter"
    aux = elevation if product.upper() == "PPI" else (cappi_height if product.upper() == "CAPPI" else "")
    cmap_override_key = (colormap_overrides or {}).get(field_requested, None)
    cmap_suffix = f"_{cmap_override_key}" if cmap_override_key else ""
    
    unique_cog_name = f"radar_{field_requested}_{product}_{filters_str}_{aux}_{file_hash}{cmap_suffix}.tif"
    cog_path = Path(output_dir) / unique_cog_name
    
    os.makedirs(output_dir, exist_ok=True)
    
    summary = {
        "image_url": str(cog_path),
        "field": field_requested,
        "source_file": filepath,
        "tilejson_url": f"placeholder/tilejson.json?url={unique_cog_name}",
    }
    
    # Load and validate radar file
    radar = pyart.io.read(filepath)
    try:
        field_name, field_key = resolve_field(radar, field_requested)
    except KeyError as e:
        raise ValueError(str(e))
    
    if elevation > radar.nsweeps - 1:
        raise ValueError(f"El ángulo de elevación {elevation} no existe en el archivo.")
    
    cmap_override = (colormap_overrides or {}).get(field_requested, None)
    cmap, vmin, vmax, cmap_key = colormap_for(field_key, override_cmap=cmap_override)
    
    # Separate filters
    qc_filters = []
    visual_filters = []
    for f in (filters or []):
        ffield = str(getattr(f, "field", "") or "").upper()
        if ffield in AFFECTS_INTERP_FIELDS:
            qc_filters.append(f)
        else:
            visual_filters.append(f)
    
    return {
        "radar": radar,
        "file_hash": file_hash,
        "cog_path": cog_path,
        "summary": summary,
        "field_name": field_name,
        "field_key": field_key,
        "cmap": cmap,
        "vmin": vmin,
        "vmax": vmax,
        "product_upper": product.upper(),
        "qc_filters": qc_filters,
        "visual_filters": visual_filters,
    }


def _prepare_radar_field(radar, field_name, product_upper, cappi_height):
    """
    Phase 4: Prepare field data for gridding.
    
    Creates filled reflectivity and selects appropriate radar object and field
    based on product type. Returns radar_to_use and field_to_use.
    
    Parameters
    ----------
    radar : pyart.core.Radar
        Radar object
    field_name : str
        Field name
    product_upper : str
        Product type (uppercase)
    cappi_height : int
        CAPPI height for CAPPI products
    
    Returns
    -------
    tuple
        (radar_to_use, field_to_use, elevation_index if applicable)
    """
    # For CAPPI/COLMAX, create filled reflectivity
    if field_name == "DBZH" and product_upper in ["CAPPI", "COLMAX"]:
        filled_DBZH = radar.fields[field_name]['data'].filled(fill_value=-30)
        radar.add_field_like(field_name, 'filled_DBZH', filled_DBZH, replace_existing=True)
        field_name = 'filled_DBZH'
    
    if product_upper == "PPI":
        # Placeholder elevation (will be passed separately)
        return radar, field_name
    elif product_upper == "CAPPI":
        cappi = create_cappi(radar, fields=[field_name], height=cappi_height)
        template = cappi.fields[field_name]['data']
        zeros_array = np.tile(template, (15, 1))
        radar.add_field_like('DBZH', 'cappi', zeros_array, replace_existing=True)
        return radar, "cappi"
    elif product_upper == "COLMAX":
        radar_colmax = create_colmax(radar)
        return radar_colmax, 'composite_reflectivity'
    else:
        raise ValueError(f"Producto inválido: {product_upper}")


def _compute_grid_limits_and_resolution(radar, product_upper, elevation, cappi_height, volume):
    """
    Phase 6: Compute grid spatial boundaries and resolution.
    
    Vectorized computation of grid limits based on product type and radar capabilities.
    
    Parameters
    ----------
    radar : pyart.core.Radar
        Radar object
    product_upper : str
        Product type (uppercase)
    elevation : int
        Elevation index for PPI
    cappi_height : int
        CAPPI height for CAPPI products
    volume : str, optional
        Volume identifier
    
    Returns
    -------
    dict
        Grid config with: z_grid_limits, y_grid_limits, x_grid_limits,
        grid_resolution, elev_deg
    """
    range_max_m = safe_range_max_m(radar)
    grid_resolution = 300 if volume == '03' else 1200
    
    if product_upper == "CAPPI":
        z_top_m = cappi_height + 2000
        elev_deg = None
    else:
        elev_deg = float(radar.fixed_angle['data'][elevation])
        hmax_km = beam_height_max_km(range_max_m, elev_deg)
        z_top_m = int((hmax_km + 3) * 1000)
    
    return {
        "z_grid_limits": (0.0, z_top_m),
        "y_grid_limits": (-range_max_m, range_max_m),
        "x_grid_limits": (-range_max_m, range_max_m),
        "grid_resolution": grid_resolution,
        "elev_deg": elev_deg,
        "range_max_m": range_max_m,
    }


def _apply_filter_masks(masked_arr, visual_filters, qc_filters, field_to_use, pkg_cached):
    """
    Phase 10 & 11: Apply visual and QC filters as vectorized masks (optimized).
    
    Uses numpy broadcasting to apply all filters efficiently in single pass.
    
    Parameters
    ----------
    masked_arr : np.ma.MaskedArray
        2D masked array
    visual_filters : list
        Visual filters (post-grid)
    qc_filters : list
        QC filters
    field_to_use : str
        Field name
    pkg_cached : dict
        Cached grid package with QC fields
    
    Returns
    -------
    np.ma.MaskedArray
        Filtered masked array
    """
    # Create a copy only if we have filters to apply (avoid unnecessary copy overhead)
    if not visual_filters and not qc_filters:
        return masked_arr
    
    # Make a copy to avoid polluting the cache
    masked_arr = np.ma.array(masked_arr, copy=True)
    
    # Vectorized visual filter application
    dyn_mask = np.zeros(masked_arr.shape, dtype=bool)
    
    for f in visual_filters:
        ffield = str(getattr(f, "field", None) or "").upper()
        if not ffield or ffield != str(field_to_use).upper():
            continue
        
        fmin = getattr(f, "min", None)
        fmax = getattr(f, "max", None)
        
        if fmin is not None and not (fmin <= 0.3 and field_to_use == "RHOHV"):
            dyn_mask |= (masked_arr < float(fmin))
        if fmax is not None:
            dyn_mask |= (masked_arr > float(fmax))
    
    masked_arr.mask = np.ma.getmaskarray(masked_arr) | dyn_mask
    
    # Vectorized QC filter application
    if qc_filters:
        qc_dict = pkg_cached.get("qc", {}) or {}
        for f in qc_filters:
            qf = str(getattr(f, "field", "") or "").upper()
            q2d = qc_dict.get(qf)
            if q2d is None:
                continue
            
            fmin = getattr(f, "min", None)
            fmax = getattr(f, "max", None)
            
            qmask = np.zeros(masked_arr.shape, dtype=bool)
            if fmin is not None:
                qmask |= (q2d < float(fmin))
            if fmax is not None:
                qmask |= (q2d > float(fmax))
            
            masked_arr.mask = np.ma.getmaskarray(masked_arr) | qmask
    
    return masked_arr


def _cache_or_build_2d_grid(
    radar_to_use,
    field_to_use,
    file_hash,
    volume,
    qc_filters,
    product_upper,
    elevation,
    cappi_height,
    grid_config,
    elev_deg,
    vmin,
    x_grid_limits,
    y_grid_limits,
    cache_key,
):
    """
    Phase 8 & 9: Retrieve or build 2D grid from 3D cache (modularized).
    
    Handles grid caching logic and collapses 3D grids to 2D.
    
    Parameters
    ----------
    radar_to_use : pyart.core.Radar
        Radar object for gridding
    field_to_use : str
        Field name
    file_hash : str
        File hash for cache key
    volume : str
        Volume identifier
    qc_filters : list
        QC filters
    product_upper : str
        Product type
    elevation : int
        Elevation index
    cappi_height : int
        CAPPI height
    grid_config : dict
        Grid configuration
    elev_deg : float
        Elevation angle in degrees
    vmin : float
        Min value for masking
    x_grid_limits : tuple
        X limits
    y_grid_limits : tuple
        Y limits
    cache_key : str
        Cache key for 2D grid
    
    Returns
    -------
    dict
        Cached package with arr, qc, crs, transform, etc.
    """
    pkg_cached = GRID2D_CACHE.get(cache_key)
    
    if pkg_cached is None:
        # Build 3D grid
        grid = _get_or_build_grid3d(
            radar_to_use=radar_to_use,
            field_to_use=field_to_use,
            file_hash=file_hash,
            volume=volume,
            qc_filters=qc_filters,
            z_grid_limits=grid_config["z_grid_limits"],
            y_grid_limits=grid_config["y_grid_limits"],
            x_grid_limits=grid_config["x_grid_limits"],
            grid_resolution=grid_config["grid_resolution"],
        )
        
        z_levels_full = grid.z['data'].copy()
        
        # Collapse to 2D
        collapse_grid_to_2d(
            grid,
            field=field_to_use,
            product=product_upper.lower(),
            elevation_deg=elev_deg,
            target_height_m=cappi_height,
            vmin=vmin,
        )
        arr2d = grid.fields[field_to_use]['data'][0, :, :]
        arr2d = np.ma.array(arr2d.astype(np.float32), mask=np.ma.getmaskarray(arr2d))
        
        # Collapse QC fields (vectorized)
        qc_2d = {}
        for qc_name in AFFECTS_INTERP_FIELDS:
            if qc_name == field_to_use or qc_name not in grid.fields:
                continue
            data3d_q = grid.fields[qc_name]['data']
            q2d = collapse_field_3d_to_2d(
                data3d_q,
                product=product_upper.lower(),
                x_coords=grid.x['data'],
                y_coords=grid.y['data'],
                z_levels=z_levels_full,
                elevation_deg=elev_deg,
                target_height_m=cappi_height,
            )
            qc_2d[qc_name] = q2d
        
        # Compute geospatial metadata (vectorized)
        grid_origin = (
            float(radar_to_use.latitude['data'][0]),
            float(radar_to_use.longitude['data'][0]),
        )
        
        x = grid.x['data'].astype(float)
        y = grid.y['data'].astype(float)
        ny, nx = arr2d.shape
        
        dx = float(np.mean(np.diff(x))) if x.size > 1 else (x_grid_limits[1] - x_grid_limits[0]) / max(nx - 1, 1)
        dy = float(np.mean(np.diff(y))) if y.size > 1 else (y_grid_limits[1] - y_grid_limits[0]) / max(ny - 1, 1)
        xmin = float(x.min()) if x.size else x_grid_limits[0]
        ymax = float(y.max()) if y.size else y_grid_limits[1]
        
        transform = Affine.translation(xmin - dx / 2, ymax + dy / 2) * Affine.scale(dx, -dy)
        proj_dict_norm = normalize_proj_dict(grid, grid_origin)
        crs_wkt = pyproj.CRS.from_dict(proj_dict_norm).to_wkt()
        
        pkg_cached = {
            "arr": arr2d,
            "qc": qc_2d,
            "crs": crs_wkt,
            "transform": transform,
            "arr_warped": None,
            "crs_warped": None,
            "transform_warped": None,
        }
        GRID2D_CACHE[cache_key] = pkg_cached
    
    return pkg_cached


def _export_to_cog(
    masked_arr,
    radar_to_use,
    field_to_use,
    x_grid_limits,
    y_grid_limits,
    cmap,
    vmin,
    vmax,
    cog_path,
    output_dir,
    pkg_cached,
    cache_key,
):
    """
    Phase 12: Export to GeoTIFF and convert to COG (modularized).
    
    Handles GeoTIFF creation, numeric export for stats, COG conversion, and cleanup.
    
    Parameters
    ----------
    masked_arr : np.ma.MaskedArray
        2D masked array
    radar_to_use : pyart.core.Radar
        Radar object
    field_to_use : str
        Field name
    x_grid_limits : tuple
        X limits
    y_grid_limits : tuple
        Y limits
    cmap : matplotlib colormap
        Colormap
    vmin : float
        Min value
    vmax : float
        Max value
    cog_path : Path
        Output COG path
    output_dir : str
        Output directory
    pkg_cached : dict
        Cached package
    cache_key : str
        Cache key
    
    Returns
    -------
    Path
        Path to created COG file
    """
    ny, nx = masked_arr.shape
    
    # Create fake grid for PyART export
    grid_fake = pyart.core.Grid(
        time={'data': np.array([0])},
        fields={field_to_use: {'data': masked_arr[np.newaxis, :, :], '_FillValue': -9999.0}},
        metadata={'instrument_name': 'RADAR'},
        origin_latitude={'data': radar_to_use.latitude['data']},
        origin_longitude={'data': radar_to_use.longitude['data']},
        origin_altitude={'data': radar_to_use.altitude['data']},
        x={'data': np.linspace(x_grid_limits[0], x_grid_limits[1], nx).astype(np.float32)},
        y={'data': np.linspace(y_grid_limits[0], y_grid_limits[1], ny).astype(np.float32)},
        z={'data': np.array([0.0], dtype=np.float32)}
    )
    
    # Create temporary GeoTIFF
    unique_tif_name = f"radar_{uuid.uuid4().hex}.tif"
    tiff_path = Path(output_dir) / unique_tif_name
    
    pyart.io.write_grid_geotiff(
        grid=grid_fake,
        filename=str(tiff_path),
        field=field_to_use,
        level=0,
        rgb=True,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        warp_to_mercator=True
    )
    
    # Cache warped version if first time
    if pkg_cached.get("arr_warped") is None:
        temp_numeric_tif = Path(output_dir) / f"numeric_{uuid.uuid4().hex}.tif"
        pyart.io.write_grid_geotiff(
            grid=grid_fake,
            filename=str(temp_numeric_tif),
            field=field_to_use,
            level=0,
            rgb=False,
            warp_to_mercator=True
        )
        
        with rasterio.open(temp_numeric_tif) as src_numeric:
            arr_warped = src_numeric.read(1, masked=True)
            transform_warped = src_numeric.transform
            crs_warped = src_numeric.crs.to_wkt()
            
            pkg_cached["arr_warped"] = arr_warped.astype(np.float32)
            pkg_cached["transform_warped"] = transform_warped
            pkg_cached["crs_warped"] = crs_warped
            GRID2D_CACHE[cache_key] = pkg_cached
        
        try:
            temp_numeric_tif.unlink()
        except OSError:
            pass
    
    # Convert to COG
    _ = convert_to_cog(tiff_path, cog_path)
    
    # Cleanup
    try:
        tiff_path.unlink()
    except OSError:
        pass
    
    return cog_path


def process_radar_to_cog(
    filepath,
    product="PPI",
    field_requested="DBZH",
    cappi_height=4000,
    elevation=0,
    filters=None,
    output_dir="output",
    volume=None,
    colormap_overrides=None
):
    """
    Process radar NetCDF file and generate Cloud-Optimized GeoTIFF (COG).
    
    Main processing function that orchestrates the complete radar-to-COG pipeline.
    Implements two-tier caching (3D and 2D grids) and modular phase architecture.
    
    Pipeline Phases
    ---------------
    1. Setup & Caching Config: File hash, output paths, validation
    2. File I/O & Field Resolution: Load radar, resolve field names
    3. Rendering Config: Get colormap and visualization parameters
    4. Data Preparation: Create filled reflectivity for CAPPI/COLMAX
    5. Product-Specific Setup: Select radar object and field based on product
    6. Grid Configuration: Compute spatial boundaries and resolution
    7. Filter Classification: Separate QC filters (interp) from visual filters
    8. 2D Grid Caching: Check cache before expensive gridding
    9. 3D Grid Building: Grid radar sweeps, collapse to 2D, cache
    10. Visual Filters: Apply range masks to field values (post-grid)
    11. QC Filters: Apply QC field masks (RHOHV, etc.) (post-grid)
    12. GeoTIFF Export: Write RGB+numeric, convert to COG, cleanup
    
    Parameters
    ----------
    filepath : str or Path
        Path to radar NetCDF file
    product : str, optional
        Product type: 'PPI', 'CAPPI', 'COLMAX', by default 'PPI'
    field_requested : str, optional
        Field to process (e.g., 'DBZH', 'ZDR'), by default 'DBZH'
    cappi_height : int, optional
        Height in meters for CAPPI product, by default 4000
    elevation : int, optional
        Elevation angle index for PPI product, by default 0
    filters : list, optional
        List of filter objects with 'field', 'min', 'max' attributes
    output_dir : str or Path, optional
        Output directory for COG files, by default 'output'
    volume : str, optional
        Volume identifier for resolution selection
    colormap_overrides : dict, optional
        Dict mapping field names to colormap keys, e.g., {'DBZH': 'grc_th2'}
    
    Returns
    -------
    dict
        Summary dict with keys:
        - image_url: Path to generated COG file
        - field: Processed field name
        - source_file: Original NetCDF filepath
        - tilejson_url: URL template for tile server (placeholder)
    
    Raises
    ------
    ValueError
        If elevation out of range, field not found, or invalid product
    KeyError
        If requested field not supported or not found in file
    
    Examples
    --------
    >>> result = process_radar_to_cog(
    ...     'radar_file.nc',
    ...     product='PPI',
    ...     field_requested='DBZH',
    ...     elevation=0,
    ...     output_dir='output'
    ... )
    >>> print(result['image_url'])
    'output/radar_DBZH_PPI_nofilter_0_abc123def456.tif'
    
    Notes
    -----
    - Modularized into 12 logical phases for clarity and testability
    - Implements two-tier caching: 3D grids and collapsed 2D grids
    - QC filters applied during gridding; visualization filters post-grid
    - All mask operations vectorized with numpy for efficiency
    - COG files include overviews for multi-scale display
    - Temporary files cleaned up automatically after COG conversion
    """
    if filters is None:
        filters = []
    
    # Phase 1 & 2: Setup, validation, and caching config
    config = _build_processing_config(
        filepath, product, field_requested, elevation, cappi_height,
        volume, colormap_overrides, filters, output_dir
    )
    
    # Early return if COG exists
    if config["cog_path"].exists():
        return config["summary"]
    
    # Phase 3: Extract config values
    radar = config["radar"]
    field_name = config["field_name"]
    cmap = config["cmap"]
    vmin = config["vmin"]
    vmax = config["vmax"]
    product_upper = config["product_upper"]
    qc_filters = config["qc_filters"]
    visual_filters = config["visual_filters"]
    
    # Phase 4: Prepare field data
    radar_to_use, field_to_use = _prepare_radar_field(
        radar, field_name, product_upper, cappi_height
    )
    
    # Phase 5: For PPI, extract sweep
    if product_upper == "PPI":
        radar_to_use = radar_to_use.extract_sweeps([elevation])
    
    # Phase 6: Compute grid configuration
    grid_config = _compute_grid_limits_and_resolution(
        radar, product_upper, elevation, cappi_height, volume
    )
    x_grid_limits = grid_config["x_grid_limits"]
    y_grid_limits = grid_config["y_grid_limits"]
    z_grid_limits = grid_config["z_grid_limits"]
    elev_deg = grid_config["elev_deg"]
    
    # Phase 7: Cache key for 2D grid
    cache_key = grid2d_cache_key(
        file_hash=config["file_hash"],
        product_upper=product_upper,
        field_to_use=field_to_use,
        elevation=elevation if product_upper == "PPI" else None,
        cappi_height=cappi_height if product_upper == "CAPPI" else None,
        volume=volume,
        interp='nearest',
        qc_sig=tuple(),
    )
    
    # Phase 8 & 9: Cache or build 2D grid
    pkg_cached = _cache_or_build_2d_grid(
        radar_to_use, field_to_use, config["file_hash"], volume, qc_filters,
        product_upper, elevation, cappi_height, grid_config, elev_deg, vmin,
        x_grid_limits, y_grid_limits, cache_key
    )
    
    # Phase 10 & 11: Apply filters (vectorized)
    masked = _apply_filter_masks(
        pkg_cached["arr"], visual_filters, qc_filters, field_to_use, pkg_cached
    )
    
    # Phase 12: Export to COG
    _export_to_cog(
        masked, radar_to_use, field_to_use, x_grid_limits, y_grid_limits,
        cmap, vmin, vmax, config["cog_path"], output_dir, pkg_cached, cache_key
    )
    
    return config["summary"]
