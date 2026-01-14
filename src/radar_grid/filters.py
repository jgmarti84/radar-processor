"""
Gate filters for excluding radar gates from interpolation.

Similar to PyART's GateFilter but designed to work with our interpolation pipeline.
"""
import logging
import numpy as np
from typing import Optional, Callable, List, Tuple

logger = logging.getLogger(__name__)

class GateFilter:
    """
    A class for filtering radar gates based on various criteria.
    
    The filter maintains a boolean mask where True = excluded gate.
    Multiple filter conditions can be combined (OR logic - if any 
    condition excludes a gate, it's excluded).
    
    Parameters
    ----------
    radar : pyart.core.Radar
        PyART radar object
    
    Examples
    --------
    >>> gf = GateFilter(radar)
    >>> gf.exclude_below('DBZH', -10)      # Exclude low reflectivity
    >>> gf.exclude_below('RHOHV', 0.7)     # Exclude low correlation
    >>> gf.exclude_above('ZDR', 8)         # Exclude high ZDR
    >>> gf.exclude_invalid('DBZH')         # Exclude NaN/Inf in DBZH
    >>> 
    >>> # Use the filter
    >>> filter_mask = gf.gate_excluded
    >>> data, field_mask = get_field_data(radar, 'DBZH')
    >>> combined_mask = field_mask | filter_mask
    >>> grid = apply_geometry(geometry, data, combined_mask)
    """
    
    def __init__(self, radar):
        """
        Initialize with all gates included (none excluded).
        
        Parameters
        ----------
        radar : pyart.core.Radar
            PyART radar object
        """
        self.radar = radar
        self.n_gates = radar.nrays * radar.ngates
        self._gate_excluded = np.zeros(self.n_gates, dtype=bool)
        self._filter_history: List[str] = []
    
    @property
    def gate_excluded(self) -> np.ndarray:
        """Boolean mask where True = gate is excluded."""
        return self._gate_excluded
    
    @property
    def gate_included(self) -> np.ndarray:
        """Boolean mask where True = gate is included."""
        return ~self._gate_excluded
    
    def n_excluded(self) -> int:
        """Return number of excluded gates."""
        return self._gate_excluded.sum()
    
    def n_included(self) -> int:
        """Return number of included gates."""
        return (~self._gate_excluded).sum()
    
    def summary(self) -> str:
        """Return a summary of the filter."""
        lines = [
            f"GateFilter Summary:",
            f"  Total gates: {self.n_gates:,}",
            f"  Excluded: {self.n_excluded():,} ({100*self.n_excluded()/self.n_gates:.1f}%)",
            f"  Included: {self.n_included():,} ({100*self.n_included()/self.n_gates:.1f}%)",
            f"  Filters applied ({len(self._filter_history)}):"
        ]
        for f in self._filter_history:
            lines.append(f"    - {f}")
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        return f"GateFilter(excluded={self.n_excluded():,}/{self.n_gates:,}, filters={len(self._filter_history)})"
    
    # def _get_field_data(self, field_name: str) -> np.ndarray:
    #     """Get flattened field data."""
    #     return np.ma.getdata(self.radar.fields[field_name]['data']).ravel().astype('float32')
    def _get_field_data(self, field_name: str) -> np.ndarray:
        """
        Get flattened field data.
        
        Note: This returns raw data values. Invalid values (NaN, Inf) are 
        still present. Use _get_valid_field_data() when you need to know
        which values are valid.
        """
        field = self.radar.fields[field_name]['data']
        # Apply masked_invalid to properly handle NaN/Inf even if mask is scalar False
        field_masked = np.ma.masked_invalid(field)
        return np.ma.getdata(field_masked).ravel().astype('float32')
    
    def _add_filter(self, mask: np.ndarray, description: str) -> 'GateFilter':
        """Add a filter mask and record it."""
        self._gate_excluded = self._gate_excluded | mask
        self._filter_history.append(description)
        return self
    
    # -------------------------------------------------------------------------
    # Basic threshold filters
    # -------------------------------------------------------------------------
    
    def exclude_below(self, field_name: str, threshold: float) -> 'GateFilter':
        """
        Exclude gates where field value is below threshold.
        
        Parameters
        ----------
        field_name : str
            Name of the field to filter on
        threshold : float
            Gates with values < threshold are excluded
        
        Returns
        -------
        self : GateFilter
            Returns self for method chaining
        """
        if field_name not in self.radar.fields:
            logger.warning(f"Field '{field_name}' not found in radar. No gates excluded.")
            return self
        data = self._get_field_data(field_name)
        mask = data < threshold
        return self._add_filter(mask, f"{field_name} < {threshold}")
    
    def exclude_above(self, field_name: str, threshold: float) -> 'GateFilter':
        """
        Exclude gates where field value is above threshold.
        
        Parameters
        ----------
        field_name : str
            Name of the field to filter on
        threshold : float
            Gates with values > threshold are excluded
        
        Returns
        -------
        self : GateFilter
            Returns self for method chaining
        """
        if field_name not in self.radar.fields:
            logger.warning(f"Field '{field_name}' not found in radar. No gates excluded.")
            return self
        data = self._get_field_data(field_name)
        mask = data > threshold
        return self._add_filter(mask, f"{field_name} > {threshold}")
    
    def exclude_between(self, field_name: str, low: float, high: float) -> 'GateFilter':
        """
        Exclude gates where field value is between low and high.
        
        Parameters
        ----------
        field_name : str
            Name of the field to filter on
        low : float
            Lower bound (exclusive)
        high : float
            Upper bound (exclusive)
        
        Returns
        -------
        self : GateFilter
            Returns self for method chaining
        """
        if field_name not in self.radar.fields:
            logger.warning(f"Field '{field_name}' not found in radar. No gates excluded.")
            return self
        data = self._get_field_data(field_name)
        mask = (data > low) & (data < high)
        return self._add_filter(mask, f"{low} < {field_name} < {high}")
    
    def exclude_outside(self, field_name: str, low: float, high: float) -> 'GateFilter':
        """
        Exclude gates where field value is outside [low, high] range.
        
        Parameters
        ----------
        field_name : str
            Name of the field to filter on
        low : float
            Lower bound
        high : float
            Upper bound
        
        Returns
        -------
        self : GateFilter
            Returns self for method chaining
        """
        if field_name not in self.radar.fields:
            logger.warning(f"Field '{field_name}' not found in radar. No gates excluded.")
            return self
        data = self._get_field_data(field_name)
        mask = (data < low) | (data > high)
        return self._add_filter(mask, f"{field_name} outside [{low}, {high}]")
    
    def exclude_equal(self, field_name: str, value: float, atol: float = 1e-5) -> 'GateFilter':
        """
        Exclude gates where field value equals a specific value.
        
        Parameters
        ----------
        field_name : str
            Name of the field to filter on
        value : float
            Value to exclude
        atol : float
            Absolute tolerance for comparison
        
        Returns
        -------
        self : GateFilter
            Returns self for method chaining
        """
        if field_name not in self.radar.fields:
            logger.warning(f"Field '{field_name}' not found in radar. No gates excluded.")
            return self
        data = self._get_field_data(field_name)
        mask = np.abs(data - value) < atol
        return self._add_filter(mask, f"{field_name} == {value}")
    
    # -------------------------------------------------------------------------
    # Invalid data filters
    # -------------------------------------------------------------------------
    
    def exclude_invalid(self, field_name: str) -> 'GateFilter':
        """
        Exclude gates with invalid data (NaN, Inf) in the specified field.
        
        Parameters
        ----------
        field_name : str
            Name of the field to check
        
        Returns
        -------
        self : GateFilter
            Returns self for method chaining
        """
        if field_name not in self.radar.fields:
            logger.warning(f"Field '{field_name}' not found in radar. No gates excluded.")
            return self
        data = self._get_field_data(field_name)
        mask = np.isnan(data) | np.isinf(data)
        return self._add_filter(mask, f"{field_name} invalid (NaN/Inf)")
    
    def exclude_masked(self, field_name: str) -> 'GateFilter':
        """
        Exclude gates that are masked in the original radar field.
        
        Parameters
        ----------
        field_name : str
            Name of the field to check
        
        Returns
        -------
        self : GateFilter
            Returns self for method chaining
        """
        if field_name not in self.radar.fields:
            logger.warning(f"Field '{field_name}' not found in radar. No gates excluded.")
            return self
        field = self.radar.fields[field_name]['data']
        if isinstance(field, np.ma.MaskedArray):
            mask = np.ma.getmaskarray(field).ravel()
        else:
            mask = np.zeros(self.n_gates, dtype=bool)
        return self._add_filter(mask, f"{field_name} masked")
    
    def exclude_all_invalid(self, field_name: str) -> 'GateFilter':
        """
        Exclude gates with any invalid data: NaN, Inf, or masked.
        
        This is equivalent to calling exclude_invalid() and exclude_masked().
        
        Parameters
        ----------
        field_name : str
            Name of the field to check
        
        Returns
        -------
        self : GateFilter
            Returns self for method chaining
        """
        if field_name not in self.radar.fields:
            logger.warning(f"Field '{field_name}' not found in radar. No gates excluded.")
            return self
        field = self.radar.fields[field_name]['data']
        field_masked = np.ma.masked_invalid(field)
        mask = np.ma.getmaskarray(field_masked).ravel()
        return self._add_filter(mask, f"{field_name} all invalid (NaN/Inf/masked)")
    
    # -------------------------------------------------------------------------
    # Geometry-based filters
    # -------------------------------------------------------------------------
    
    def exclude_below_altitude(self, altitude: float) -> 'GateFilter':
        """
        Exclude gates below a certain altitude.
        
        Parameters
        ----------
        altitude : float
            Altitude threshold in meters
        
        Returns
        -------
        self : GateFilter
            Returns self for method chaining
        """
        gate_alt = self.radar.gate_altitude['data'].ravel()
        mask = gate_alt < altitude
        return self._add_filter(mask, f"altitude < {altitude}m")
    
    def exclude_above_altitude(self, altitude: float) -> 'GateFilter':
        """
        Exclude gates above a certain altitude.
        
        Parameters
        ----------
        altitude : float
            Altitude threshold in meters
        
        Returns
        -------
        self : GateFilter
            Returns self for method chaining
        """
        gate_alt = self.radar.gate_altitude['data'].ravel()
        mask = gate_alt > altitude
        return self._add_filter(mask, f"altitude > {altitude}m")
    
    def exclude_below_range(self, range_min: float) -> 'GateFilter':
        """
        Exclude gates closer than a certain range from radar.
        
        Parameters
        ----------
        range_min : float
            Minimum range in meters
        
        Returns
        -------
        self : GateFilter
            Returns self for method chaining
        """
        # Range is 1D, need to broadcast to all rays
        ranges = self.radar.range['data']
        range_2d = np.broadcast_to(ranges, (self.radar.nrays, self.radar.ngates))
        mask = range_2d.ravel() < range_min
        return self._add_filter(mask, f"range < {range_min}m")
    
    def exclude_above_range(self, range_max: float) -> 'GateFilter':
        """
        Exclude gates farther than a certain range from radar.
        
        Parameters
        ----------
        range_max : float
            Maximum range in meters
        
        Returns
        -------
        self : GateFilter
            Returns self for method chaining
        """
        ranges = self.radar.range['data']
        range_2d = np.broadcast_to(ranges, (self.radar.nrays, self.radar.ngates))
        mask = range_2d.ravel() > range_max
        return self._add_filter(mask, f"range > {range_max}m")
    
    def exclude_below_elevation_angle(self, min_elev: float) -> 'GateFilter':
        """
        Exclude gates with elevation angle below a threshold.
        
        Parameters
        ----------
        min_elev : float
            Minimum elevation angle in degrees
        
        Returns
        -------
        self : GateFilter
            Returns self for method chaining
        
        Notes
        -----
        Elevation angles are per-ray, so all gates in a ray with elevation
        below the threshold will be excluded. Useful for filtering out 
        low-elevation scans that may have ground clutter or be affected by terrain.
        
        Example
        -------
        >>> gf.exclude_below_elevation_angle(2.0)  # Exclude angles < 2 degrees
        """
        elev_angles = self.radar.elevation['data']  # shape (nrays,)
        elev_repeated = np.repeat(elev_angles, self.radar.ngates)  # shape (nrays * ngates,)
        mask = elev_repeated < min_elev
        return self._add_filter(mask, f"elevation angle < {min_elev}°")
    
    def exclude_above_elevation_angle(self, max_elev: float) -> 'GateFilter':
        """
        Exclude gates with elevation angle above a threshold.
        
        Parameters
        ----------
        max_elev : float
            Maximum elevation angle in degrees
        
        Returns
        -------
        self : GateFilter
            Returns self for method chaining
        
        Notes
        -----
        Elevation angles are per-ray, so all gates in a ray with elevation
        above the threshold will be excluded. Useful for filtering out very 
        high elevation angles where the beam is nearly vertical and gate 
        volumes become very large or sampling becomes sparse.
        
        Example
        -------
        >>> gf.exclude_above_elevation_angle(30.0)  # Exclude angles > 30 degrees
        """
        elev_angles = self.radar.elevation['data']  # shape (nrays,)
        elev_repeated = np.repeat(elev_angles, self.radar.ngates)  # shape (nrays * ngates,)
        mask = elev_repeated > max_elev
        return self._add_filter(mask, f"elevation angle > {max_elev}°")
    
    def exclude_outside_elevation_range(self, min_elev: float, max_elev: float) -> 'GateFilter':
        """
        Exclude gates with elevation angle outside [min_elev, max_elev] range.
        
        Parameters
        ----------
        min_elev : float
            Minimum elevation angle in degrees (inclusive)
        max_elev : float
            Maximum elevation angle in degrees (inclusive)
        
        Returns
        -------
        self : GateFilter
            Returns self for method chaining
        
        Example
        -------
        >>> gf.exclude_outside_elevation_range(1.0, 25.0)
        """
        elev_angles = self.radar.elevation['data']  # shape (nrays,)
        elev_repeated = np.repeat(elev_angles, self.radar.ngates)  # shape (nrays * ngates,)
        mask = (elev_repeated < min_elev) | (elev_repeated > max_elev)
        return self._add_filter(mask, f"elevation angle outside [{min_elev}°, {max_elev}°]")
    
    # -------------------------------------------------------------------------
    # Custom filters
    # -------------------------------------------------------------------------
    
    def exclude_where(self, mask: np.ndarray, description: str = "custom") -> 'GateFilter':
        """
        Exclude gates based on a custom boolean mask.
        
        Parameters
        ----------
        mask : np.ndarray
            Boolean array of shape (nrays, ngates) or (nrays * ngates,)
            where True = exclude
        description : str
            Description of this filter for the summary
        
        Returns
        -------
        self : GateFilter
            Returns self for method chaining
        """
        mask_flat = mask.ravel().astype(bool)
        if len(mask_flat) != self.n_gates:
            raise ValueError(f"Mask size {len(mask_flat)} doesn't match n_gates {self.n_gates}")
        return self._add_filter(mask_flat, description)
    
    def exclude_by_function(
        self, 
        field_name: str, 
        func: Callable[[np.ndarray], np.ndarray],
        description: str = "custom function"
    ) -> 'GateFilter':
        """
        Exclude gates based on a custom function applied to field data.
        
        Parameters
        ----------
        field_name : str
            Name of the field to process
        func : callable
            Function that takes field data array and returns boolean mask
            where True = exclude
        description : str
            Description of this filter
        
        Returns
        -------
        self : GateFilter
            Returns self for method chaining
        
        Example
        -------
        >>> # Exclude gates where DBZH is in the bottom 10%
        >>> gf.exclude_by_function('DBZH', 
        ...     lambda x: x < np.nanpercentile(x, 10),
        ...     description="DBZH bottom 10%")
        """
        data = self._get_field_data(field_name)
        mask = func(data)
        return self._add_filter(mask, f"{field_name}: {description}")
    
    # -------------------------------------------------------------------------
    # Utility methods
    # -------------------------------------------------------------------------
    
    def copy(self) -> 'GateFilter':
        """Create a copy of this filter."""
        new_filter = GateFilter(self.radar)
        new_filter._gate_excluded = self._gate_excluded.copy()
        new_filter._filter_history = self._filter_history.copy()
        return new_filter
    
    def reset(self) -> 'GateFilter':
        """Reset the filter to include all gates."""
        self._gate_excluded = np.zeros(self.n_gates, dtype=bool)
        self._filter_history = []
        return self
    
    def include_all(self) -> 'GateFilter':
        """Alias for reset()."""
        return self.reset()
    
    def exclude_all(self) -> 'GateFilter':
        """Exclude all gates."""
        self._gate_excluded = np.ones(self.n_gates, dtype=bool)
        self._filter_history.append("exclude all")
        return self


def create_mask_from_filter(
    radar, 
    field_name: str, 
    gatefilter: Optional[GateFilter] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create field data and combined mask from radar field and optional filter.
    
    This is a convenience function that combines get_field_data behavior
    with gate filtering.
    
    Parameters
    ----------
    radar : pyart.core.Radar
        PyART radar object
    field_name : str
        Name of the field to extract
    gatefilter : GateFilter, optional
        Gate filter to apply. If None, only invalid data is masked.
    
    Returns
    -------
    field_data : np.ndarray
        Flattened field values
    combined_mask : np.ndarray
        Combined mask (True = excluded) from invalid data and filter
    """
    field = radar.fields[field_name]['data']
    field_masked = np.ma.masked_invalid(field)
    
    field_data = np.ma.getdata(field_masked).ravel().astype('float32')
    field_mask = np.ma.getmaskarray(field_masked).ravel()
    
    if gatefilter is not None:
        combined_mask = field_mask | gatefilter.gate_excluded
    else:
        combined_mask = field_mask
    
    return field_data, combined_mask