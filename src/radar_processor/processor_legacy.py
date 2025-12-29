"""
Legacy processor module - Original implementation before refactoring.

This module contains the original process_radar_to_cog implementation for
performance comparison and testing against the refactored version.

WARNING: This is legacy code. Use processor.py for production.
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

# Import shared helper functions from processor module
from .processor import (
    _get_or_build_grid3d,
    convert_to_cog,
    create_colmax,
    beam_height_max_km,
    collapse_grid_to_2d,
    create_cappi,
    warp_to_web_mercator,
)


def process_radar_to_cog_legacy(
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
    LEGACY IMPLEMENTATION: Process radar NetCDF file and generate COG.
    
    This is the original implementation before refactoring. Use for
    performance comparison testing only. For production, use
    process_radar_to_cog() from processor.py instead.
    
    Notes
    -----
    - Original monolithic function (280 lines)
    - Sequential non-vectorized operations
    - Less modular structure
    - All processing logic in single function
    """
    if filters is None:
        filters = []
    
    # Create stable filename based on processing parameters
    file_hash = md5_file(filepath)[:12]
    filters_str = "_".join([f"{f.field}_{f.min}_{f.max}" for f in filters]) if filters else "nofilter"
    aux = elevation if product.upper() == "PPI" else (cappi_height if product.upper() == "CAPPI" else "")
    
    # Include colormap in filename if overridden
    cmap_override_key = (colormap_overrides or {}).get(field_requested, None)
    cmap_suffix = f"_{cmap_override_key}" if cmap_override_key else ""
    
    unique_cog_name = f"radar_{field_requested}_{product}_{filters_str}_{aux}_{file_hash}{cmap_suffix}.tif"
    cog_path = Path(output_dir) / unique_cog_name
    
    # Generate output summary
    summary = {
        "image_url": str(cog_path),
        "field": field_requested,
        "source_file": filepath,
        "tilejson_url": f"placeholder/tilejson.json?url={unique_cog_name}",
    }

    # Early return if COG already exists
    if cog_path.exists():
        return summary

    # Process radar file
    os.makedirs(output_dir, exist_ok=True)
    radar = pyart.io.read(filepath)

    try:
        field_name, field_key = resolve_field(radar, field_requested)
    except KeyError as e:
        raise ValueError(str(e))
    
    if elevation > radar.nsweeps - 1:
        raise ValueError(f"El ángulo de elevación {elevation} no existe en el archivo.")
    
    # Get render defaults with optional colormap override
    cmap_override = (colormap_overrides or {}).get(field_requested, None)
    cmap, vmin, vmax, cmap_key = colormap_for(field_key, override_cmap=cmap_override)

    # Prepare field for products requiring filled reflectivity
    if field_name == "DBZH" and product.upper() in ["CAPPI", "COLMAX"]:
        filled_DBZH = radar.fields[field_name]['data'].filled(fill_value=-30)
        radar.add_field_like(field_name, 'filled_DBZH', filled_DBZH, replace_existing=True)
        field_name = 'filled_DBZH'

    # Define radar and field to use based on product
    product_upper = product.upper()
    if product_upper == "PPI":
        radar_to_use = radar.extract_sweeps([elevation])
        field_to_use = field_name
    elif product_upper == "CAPPI":
        cappi = create_cappi(radar, fields=[field_name], height=cappi_height)
        # Create field filled with CAPPI data
        template = cappi.fields[field_name]['data']  # (360, 523)
        zeros_array = np.tile(template, (15, 1))  # (5400, 523)
        radar.add_field_like('DBZH', 'cappi', zeros_array, replace_existing=True)

        radar_to_use = radar
        field_to_use = "cappi"
    elif product_upper == "COLMAX" and field_key.upper() == "DBZH":
        radar_to_use = create_colmax(radar)
        field_to_use = 'composite_reflectivity'
    else:
        raise ValueError(f"Producto inválido: {product_upper}")

    # Define grid limits
    range_max_m = safe_range_max_m(radar)

    if product_upper == "CAPPI":
        z_top_m = cappi_height + 2000  # +2 km margin
        elev_deg = None
    else:
        elev_deg = float(radar.fixed_angle['data'][elevation])
        hmax_km = beam_height_max_km(range_max_m, elev_deg)
        z_top_m = int((hmax_km + 3) * 1000)  # +3 km margin
    
    z_grid_limits = (0.0, z_top_m)
    y_grid_limits = (-range_max_m, range_max_m)
    x_grid_limits = (-range_max_m, range_max_m)

    # Calculate grid points
    grid_resolution = 300 if volume == '03' else 1200
    z_points = int(np.ceil(z_grid_limits[1] / grid_resolution)) + 1
    y_points = int((y_grid_limits[1] - y_grid_limits[0]) / grid_resolution)
    x_points = int((x_grid_limits[1] - x_grid_limits[0]) / grid_resolution)

    interp = 'nearest'

    # Separate QC vs visual filters (all applied post-grid as 2D masks)
    qc_filters = []
    visual_filters = []
    for f in (filters or []):
        ffield = str(getattr(f, "field", "") or "").upper()
        if ffield in AFFECTS_INTERP_FIELDS:
            qc_filters.append(f)
        else:
            visual_filters.append(f)

    # Try to use cached collapsed 2D grid
    cache_key = grid2d_cache_key(
        file_hash=file_hash,
        product_upper=product_upper,
        field_to_use=field_to_use,
        elevation=elevation if product_upper == "PPI" else None,
        cappi_height=cappi_height if product_upper == "CAPPI" else None,
        volume=volume,
        interp=interp,
        qc_sig=tuple(),  # Filters don't affect cache
    )

    pkg_cached = GRID2D_CACHE.get(cache_key)

    if pkg_cached is None:
        # Build or retrieve cached 3D grid
        grid = _get_or_build_grid3d(
            radar_to_use=radar_to_use,
            field_to_use=field_to_use,
            file_hash=file_hash,
            volume=volume,
            qc_filters=qc_filters,
            z_grid_limits=z_grid_limits,
            y_grid_limits=y_grid_limits,
            x_grid_limits=x_grid_limits,
            grid_resolution=grid_resolution,
        )

        # Save full z levels before collapsing main field
        z_levels_full = grid.z['data'].copy()

        collapse_grid_to_2d(
            grid,
            field=field_to_use,
            product=product.lower(),
            elevation_deg=elev_deg,
            target_height_m=cappi_height,
            vmin=vmin,
        )
        arr2d = grid.fields[field_to_use]['data'][0, :, :]
        arr2d = np.ma.array(arr2d.astype(np.float32), mask=np.ma.getmaskarray(arr2d))

        # Collapse QC fields to 2D using non-destructive method
        qc_2d = {}
        for qc_name in AFFECTS_INTERP_FIELDS:
            if qc_name == field_to_use:
                continue
            if qc_name not in grid.fields:
                continue
            data3d_q = grid.fields[qc_name]['data']
            q2d = collapse_field_3d_to_2d(
                data3d_q,
                product=product.lower(),
                x_coords=grid.x['data'],
                y_coords=grid.y['data'],
                z_levels=z_levels_full,
                elevation_deg=elev_deg,
                target_height_m=cappi_height,
            )
            qc_2d[qc_name] = q2d

        # Get grid_origin for normalize_proj_dict
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
        
        # Save to cache in local CRS (warped version added after first warp by PyART)
        pkg_cached = {
            "arr": arr2d,
            "qc": qc_2d,
            "crs": crs_wkt,
            "transform": transform,
            "arr_warped": None,  # Filled after first warp
            "crs_warped": None,
            "transform_warped": None,
        }
        GRID2D_CACHE[cache_key] = pkg_cached

    # Apply visual filters on same field as masks post-grid
    masked = np.ma.array(pkg_cached["arr"], copy=True)
    dyn_mask = np.zeros(masked.shape, dtype=bool)

    for f in (visual_filters or []):
        ffield = getattr(f, "field", None)
        if not ffield:
            continue
        if str(ffield).upper() == str(field_to_use).upper():
            fmin = getattr(f, "min", None)
            fmax = getattr(f, "max", None)
            if fmin is not None:
                if fmin <= 0.3 and field_to_use == "RHOHV":
                    continue
                else:
                    dyn_mask |= (masked < float(fmin))
            if fmax is not None:
                dyn_mask |= (masked > float(fmax))

    masked.mask = np.ma.getmaskarray(masked) | dyn_mask

    # Apply QC filters post-grid (without regridding)
    if qc_filters:
        qc_dict = pkg_cached.get("qc", {}) or {}
        for f in qc_filters:
            qf = str(getattr(f, "field", "") or "").upper()
            q2d = qc_dict.get(qf)
            if q2d is None:
                continue
            qmask = np.zeros(masked.shape, dtype=bool)
            fmin = getattr(f, "min", None)
            fmax = getattr(f, "max", None)
            if fmin is not None:
                qmask |= (q2d < float(fmin))
            if fmax is not None:
                qmask |= (q2d > float(fmax))
            masked.mask = np.ma.getmaskarray(masked) | qmask

    # Create temporary GeoTIFF path
    unique_tif_name = f"radar_{uuid.uuid4().hex}.tif"
    tiff_path = Path(output_dir) / unique_tif_name

    # Create minimal grid for write_grid_geotiff (2D)
    ny, nx = masked.shape
    grid_fake = pyart.core.Grid(
        time={'data': np.array([0])},
        fields={field_to_use: {'data': masked[np.newaxis, :, :], '_FillValue': -9999.0}},
        metadata={'instrument_name': 'RADAR'},
        origin_latitude={'data': radar_to_use.latitude['data']},
        origin_longitude={'data': radar_to_use.longitude['data']},
        origin_altitude={'data': radar_to_use.altitude['data']},
        x={'data': np.linspace(x_grid_limits[0], x_grid_limits[1], nx).astype(np.float32)},
        y={'data': np.linspace(y_grid_limits[0], y_grid_limits[1], ny).astype(np.float32)},
        z={'data': np.array([0.0], dtype=np.float32)}
    )

    # Export to GeoTIFF with colormap
    pyart.io.write_grid_geotiff(
        grid=grid_fake,
        filename=str(tiff_path),
        field=field_to_use,
        level=0,
        rgb=True,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        warp=False
    )
    
    # Warp to Web Mercator projection (standalone process, independent of pyart)
    warp_to_web_mercator(tiff_path)
    
    # Update cache with warped version if first time (for stats)
    if pkg_cached.get("arr_warped") is None:
        temp_numeric_tif = Path(output_dir) / f"numeric_{uuid.uuid4().hex}.tif"
        pyart.io.write_grid_geotiff(
            grid=grid_fake,
            filename=str(temp_numeric_tif),
            field=field_to_use,
            level=0,
            rgb=False,  # Numeric values without colormap
            warp=False
        )
        
        # Warp to Web Mercator projection (standalone process, independent of pyart)
        warp_to_web_mercator(temp_numeric_tif)
        
        # Read warped numeric GeoTIFF for stats
        with rasterio.open(temp_numeric_tif) as src_numeric:
            arr_warped = src_numeric.read(1, masked=True)
            transform_warped = src_numeric.transform
            crs_warped = src_numeric.crs.to_wkt()
            
            # Add warped version to cache
            pkg_cached["arr_warped"] = arr_warped.astype(np.float32)
            pkg_cached["transform_warped"] = transform_warped
            pkg_cached["crs_warped"] = crs_warped
            GRID2D_CACHE[cache_key] = pkg_cached
        
        # Clean up temporary numeric GeoTIFF
        try:
            temp_numeric_tif.unlink()
        except OSError:
            pass

    # Convert to COG
    _ = convert_to_cog(tiff_path, cog_path)

    # Clean up temporary GeoTIFF (only COG remains)
    try:
        tiff_path.unlink()
    except OSError:
        pass

    return summary
