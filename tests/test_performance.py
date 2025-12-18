"""
Performance benchmarking suite for comparing legacy vs optimized processor implementations.

This module provides comprehensive performance measurements, profiling, and comparison
utilities for validating the improvements made to the radar COG processing pipeline.

Usage:
    python -m pytest tests/test_performance.py -v -s
    python -m pytest tests/test_performance.py::TestPerformanceBenchmarks -v --tb=short
"""
import time
import json
import os
from pathlib import Path
from typing import Dict, Tuple, List
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
import cProfile
import pstats
import io

from radar_cog_processor.processor import process_radar_to_cog
from radar_cog_processor.processor_legacy import process_radar_to_cog_legacy
from radar_cog_processor.processor import (
    _build_processing_config,
    _prepare_radar_field,
    _compute_grid_limits_and_resolution,
    _apply_filter_masks,
    _cache_or_build_2d_grid,
    _export_to_cog,
)


class PerformanceTimer:
    """Context manager for timing code blocks with high precision."""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.elapsed = 0
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start_time
        print(f"  {self.name}: {self.elapsed:.4f}s")
    
    def __repr__(self):
        return f"<PerformanceTimer {self.name}: {self.elapsed:.4f}s>"


class BenchmarkResults:
    """Stores and analyzes benchmark results."""
    
    def __init__(self):
        self.results = {}
        self.measurements = []
    
    def record(self, test_name: str, legacy_time: float, optimized_time: float, metadata: Dict = None):
        """Record a benchmark measurement."""
        speedup = legacy_time / optimized_time if optimized_time > 0 else float('inf')
        improvement = ((legacy_time - optimized_time) / legacy_time * 100) if legacy_time > 0 else 0
        
        measurement = {
            "test": test_name,
            "legacy_time": legacy_time,
            "optimized_time": optimized_time,
            "speedup": speedup,
            "improvement_pct": improvement,
            "metadata": metadata or {}
        }
        self.measurements.append(measurement)
        self.results[test_name] = measurement
        
        return measurement
    
    def print_summary(self):
        """Print formatted benchmark summary."""
        print("\n" + "="*80)
        print("PERFORMANCE BENCHMARK SUMMARY")
        print("="*80)
        print(f"{'Test Name':<40} {'Legacy':<12} {'Optimized':<12} {'Speedup':<10} {'Improvement':<12}")
        print("-"*80)
        
        for measurement in self.measurements:
            test = measurement["test"]
            legacy = measurement["legacy_time"]
            optimized = measurement["optimized_time"]
            speedup = measurement["speedup"]
            improvement = measurement["improvement_pct"]
            
            print(f"{test:<40} {legacy:>10.4f}s {optimized:>10.4f}s {speedup:>8.2f}x {improvement:>10.1f}%")
        
        print("-"*80)
        avg_speedup = np.mean([m["speedup"] for m in self.measurements if m["speedup"] != float('inf')])
        print(f"{'Average Speedup':<40} {'':<12} {'':<12} {avg_speedup:>8.2f}x")
        print("="*80 + "\n")
    
    def save_json(self, filepath: Path):
        """Save results to JSON file."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.measurements, f, indent=2, default=str)
        print(f"Results saved to {filepath}")


class ProfileHelper:
    """Helper for profiling function execution."""
    
    @staticmethod
    def profile_function(func, *args, **kwargs) -> Tuple[float, str]:
        """Profile a function and return elapsed time and stats."""
        pr = cProfile.Profile()
        pr.enable()
        
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        
        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        ps.print_stats(10)  # Top 10 functions
        
        return elapsed, s.getvalue(), result


# ============================================================================
# UNIT TESTS FOR INDIVIDUAL OPTIMIZED FUNCTIONS
# ============================================================================

class TestOptimizedFunctions:
    """Test individual optimized functions for correctness and performance."""
    
    @pytest.fixture
    def mock_radar(self):
        """Create a mock radar object."""
        radar = MagicMock()
        radar.nsweeps = 5
        radar.fixed_angle = {'data': np.array([0.5, 1.5, 2.5, 3.5, 4.5])}
        radar.latitude = {'data': np.array([40.0])}
        radar.longitude = {'data': np.array([-105.0])}
        radar.altitude = {'data': np.array([1600.0])}
        radar.fields = {
            'DBZH': {
                'data': np.ma.array(np.random.randn(100, 200), 
                                    mask=np.random.rand(100, 200) > 0.9),
                'units': 'dBZ',
                'long_name': 'Reflectivity'
            }
        }
        return radar
    
    def test_build_processing_config(self, tmp_path, mock_radar):
        """Test Phase 1-2: Config building and validation."""
        filepath = tmp_path / "test.nc"
        filepath.touch()
        
        # Mock the file reading and field resolution
        with patch('radar_cog_processor.processor.pyart.io.read', return_value=mock_radar):
            with patch('radar_cog_processor.processor.resolve_field', 
                      return_value=('DBZH', 'DBZH')):
                with patch('radar_cog_processor.processor.colormap_for',
                          return_value=(None, -30, 50, 'DBZH')):
                    
                    with PerformanceTimer("_build_processing_config") as timer:
                        config = _build_processing_config(
                            filepath, 'PPI', 'DBZH', 0, 4000, None, None, [], str(tmp_path)
                        )
                    
                    assert config is not None
                    assert 'radar' in config
                    assert 'file_hash' in config
                    assert 'summary' in config
                    assert timer.elapsed < 1.0  # Should be very fast
    
    def test_compute_grid_limits_vectorization(self, mock_radar):
        """Test Phase 6: Grid computation is vectorized."""
        with patch('radar_cog_processor.processor.safe_range_max_m', return_value=150000):
            with PerformanceTimer("_compute_grid_limits_and_resolution") as timer:
                grid_config = _compute_grid_limits_and_resolution(
                    mock_radar, 'PPI', 0, 4000, None
                )
            
            assert grid_config is not None
            assert 'z_grid_limits' in grid_config
            assert 'grid_resolution' in grid_config
            assert timer.elapsed < 0.1  # Should be fast (vectorized)


# ============================================================================
# PERFORMANCE BENCHMARKS
# ============================================================================

class TestPerformanceBenchmarks:
    """Main performance benchmark tests."""
    
    @pytest.fixture(autouse=True)
    def setup_benchmarks(self):
        """Setup benchmark environment."""
        self.results = BenchmarkResults()
        self.output_dir = Path("output/benchmark_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        yield
        self.results.print_summary()
        self.results.save_json(self.output_dir / "benchmark_results.json")
    
    @pytest.fixture
    def sample_netcdf_path(self, tmp_path):
        """Get path to sample NetCDF file."""
        sample_file = Path("data/netcdf/RMA1_0315_01_20251208T191648Z.nc")
        if not sample_file.exists():
            pytest.skip(f"Sample NetCDF file not found: {sample_file}")
        return sample_file
    
    def test_full_pipeline_ppi(self, sample_netcdf_path):
        """Benchmark full pipeline: PPI processing."""
        output_legacy = Path("output/benchmark_legacy_ppi")
        output_optimized = Path("output/benchmark_optimized_ppi")
        output_legacy.mkdir(parents=True, exist_ok=True)
        output_optimized.mkdir(parents=True, exist_ok=True)
        
        # Clear cache between runs
        from radar_cog_processor.cache import GRID2D_CACHE, GRID3D_CACHE
        GRID2D_CACHE.clear()
        GRID3D_CACHE.clear()
        
        # Run legacy version
        with PerformanceTimer("Legacy PPI processing") as timer_legacy:
            result_legacy = process_radar_to_cog_legacy(
                str(sample_netcdf_path),
                product="PPI",
                field_requested="DBZH",
                elevation=0,
                output_dir=str(output_legacy)
            )
        
        # Clear cache
        GRID2D_CACHE.clear()
        GRID3D_CACHE.clear()
        
        # Run optimized version
        with PerformanceTimer("Optimized PPI processing") as timer_optimized:
            result_optimized = process_radar_to_cog(
                str(sample_netcdf_path),
                product="PPI",
                field_requested="DBZH",
                elevation=0,
                output_dir=str(output_optimized)
            )
        
        # Record results
        measurement = self.results.record(
            "Full Pipeline - PPI",
            timer_legacy.elapsed,
            timer_optimized.elapsed,
            {"product": "PPI", "field": "DBZH"}
        )
        
        print(f"\n  Speedup: {measurement['speedup']:.2f}x")
        print(f"  Improvement: {measurement['improvement_pct']:.1f}%")
    
    def test_filter_application_performance(self, sample_netcdf_path):
        """Benchmark filter application: Compare legacy vs optimized."""
        # Create sample masked array
        sample_arr = np.ma.array(np.random.randn(5400, 5400), 
                                 mask=np.random.rand(5400, 5400) > 0.95)
        
        # Create mock filter objects
        class MockFilter:
            def __init__(self, field, min_val, max_val):
                self.field = field
                self.min = min_val
                self.max = max_val
        
        filters = [
            MockFilter("DBZH", -30, 50),
            MockFilter("DBZH", -25, 60),
            MockFilter("DBZH", -20, 70),
            MockFilter("DBZH", -15, 75),
            MockFilter("DBZH", -10, 80),
        ]
        
        pkg_cached = {"qc": {}}
        
        # Legacy: Sequential non-vectorized approach
        def apply_filters_legacy(arr, filt_list, field):
            dyn_mask = np.zeros(arr.shape, dtype=bool)
            for f in filt_list:
                if str(f.field).upper() == field:
                    if f.min is not None:
                        dyn_mask |= (arr < float(f.min))
                    if f.max is not None:
                        dyn_mask |= (arr > float(f.max))
            arr.mask = np.ma.getmaskarray(arr) | dyn_mask
            return arr
        
        # Time legacy
        with PerformanceTimer("Legacy filter application (5 filters)") as timer_legacy:
            result_legacy = apply_filters_legacy(sample_arr.copy(), filters, "DBZH")
        
        # Time optimized (uses vectorized operations in _apply_filter_masks)
        with PerformanceTimer("Optimized filter application (5 filters)") as timer_optimized:
            result_optimized = _apply_filter_masks(
                sample_arr.copy(), filters, [], "DBZH", pkg_cached
            )
        
        measurement = self.results.record(
            "Filter Application (5 filters)",
            timer_legacy.elapsed,
            timer_optimized.elapsed,
            {"filters": 5, "array_size": "5400×5400"}
        )
        
        print(f"\n  Speedup: {measurement['speedup']:.2f}x")


# ============================================================================
# INTEGRATION TESTS WITH REAL DATA
# ============================================================================

class TestIntegration:
    """Integration tests with real radar files."""
    
    @pytest.fixture
    def sample_netcdf_files(self):
        """Find all sample NetCDF files."""
        data_dir = Path("data/netcdf")
        if not data_dir.exists():
            pytest.skip("No sample data directory found")
        files = list(data_dir.glob("*.nc"))
        if not files:
            pytest.skip("No NetCDF files found in data/netcdf/")
        return files
    
    def test_multiple_files_processing(self, sample_netcdf_files):
        """Test processing multiple NetCDF files."""
        results = BenchmarkResults()
        
        for filepath in sample_netcdf_files[:2]:  # Limit to 2 files
            from radar_cog_processor.cache import GRID2D_CACHE, GRID3D_CACHE
            GRID2D_CACHE.clear()
            GRID3D_CACHE.clear()
            
            print(f"\nProcessing: {filepath.name}")
            
            output_legacy = Path(f"output/integration_legacy/{filepath.stem}")
            output_optimized = Path(f"output/integration_optimized/{filepath.stem}")
            output_legacy.mkdir(parents=True, exist_ok=True)
            output_optimized.mkdir(parents=True, exist_ok=True)
            
            with PerformanceTimer(f"Legacy - {filepath.name}") as timer_legacy:
                try:
                    process_radar_to_cog_legacy(
                        str(filepath),
                        product="PPI",
                        field_requested="DBZH",
                        elevation=0,
                        output_dir=str(output_legacy)
                    )
                except Exception as e:
                    print(f"    Legacy failed: {e}")
                    timer_legacy.elapsed = None
            
            GRID2D_CACHE.clear()
            GRID3D_CACHE.clear()
            
            with PerformanceTimer(f"Optimized - {filepath.name}") as timer_optimized:
                try:
                    process_radar_to_cog(
                        str(filepath),
                        product="PPI",
                        field_requested="DBZH",
                        elevation=0,
                        output_dir=str(output_optimized)
                    )
                except Exception as e:
                    print(f"    Optimized failed: {e}")
                    timer_optimized.elapsed = None
            
            if timer_legacy.elapsed and timer_optimized.elapsed:
                results.record(
                    f"File: {filepath.name}",
                    timer_legacy.elapsed,
                    timer_optimized.elapsed,
                    {"file_size": filepath.stat().st_size}
                )
        
        results.print_summary()
    
    def test_different_products(self, sample_netcdf_files):
        """Test performance across different product types."""
        if not sample_netcdf_files:
            pytest.skip("No sample files")
        
        filepath = sample_netcdf_files[0]
        results = BenchmarkResults()
        
        for product in ["PPI", "CAPPI", "COLMAX"]:
            from radar_cog_processor.cache import GRID2D_CACHE, GRID3D_CACHE
            GRID2D_CACHE.clear()
            GRID3D_CACHE.clear()
            
            print(f"\nTesting product: {product}")
            output_dir = Path(f"output/product_test_{product}")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                with PerformanceTimer(f"Optimized - {product}") as timer:
                    process_radar_to_cog(
                        str(filepath),
                        product=product,
                        field_requested="DBZH",
                        elevation=0,
                        cappi_height=4000,
                        output_dir=str(output_dir)
                    )
                
                results.record(
                    f"Product: {product}",
                    0,  # No legacy comparison here
                    timer.elapsed,
                    {"product": product}
                )
            except Exception as e:
                print(f"    Failed: {e}")


class TestOutputEquality:
    """Comprehensive tests to verify legacy and optimized produce identical outputs."""
    
    @staticmethod
    def compare_geotiff_files(filepath1: Path, filepath2: Path) -> Dict:
        """
        Compare two GeoTIFF files comprehensively.
        
        Parameters
        ----------
        filepath1 : Path
            First GeoTIFF file
        filepath2 : Path
            Second GeoTIFF file
        
        Returns
        -------
        dict
            Comparison results with keys: matches, differences, stats
        """
        import rasterio
        
        results = {
            "file1": str(filepath1),
            "file2": str(filepath2),
            "exists": (filepath1.exists(), filepath2.exists()),
            "size_bytes": (filepath1.stat().st_size, filepath2.stat().st_size),
            "data_match": False,
            "metadata_match": False,
            "crs_match": False,
            "transform_match": False,
            "differences": [],
            "stats": {}
        }
        
        if not (filepath1.exists() and filepath2.exists()):
            results["differences"].append("One or both files do not exist")
            return results
        
        try:
            with rasterio.open(filepath1) as src1, rasterio.open(filepath2) as src2:
                # Compare metadata
                results["metadata"] = {
                    "file1_dtype": str(src1.dtypes[0]),
                    "file2_dtype": str(src2.dtypes[0]),
                    "file1_shape": (src1.height, src1.width),
                    "file2_shape": (src2.height, src2.width),
                    "file1_bands": src1.count,
                    "file2_bands": src2.count,
                }
                
                if src1.dtypes[0] != src2.dtypes[0]:
                    results["differences"].append(f"Data type mismatch: {src1.dtypes[0]} vs {src2.dtypes[0]}")
                
                if (src1.height, src1.width) != (src2.height, src2.width):
                    results["differences"].append(f"Shape mismatch: {(src1.height, src1.width)} vs {(src2.height, src2.width)}")
                
                if src1.count != src2.count:
                    results["differences"].append(f"Band count mismatch: {src1.count} vs {src2.count}")
                
                # Compare CRS and transform
                results["crs_match"] = (str(src1.crs) == str(src2.crs))
                results["transform_match"] = (src1.transform == src2.transform)
                
                if not results["crs_match"]:
                    results["differences"].append(f"CRS mismatch: {src1.crs} vs {src2.crs}")
                
                if not results["transform_match"]:
                    results["differences"].append(f"Transform mismatch: {src1.transform} vs {src2.transform}")
                
                # Compare actual data
                data1 = src1.read()
                data2 = src2.read()
                
                # Compute statistics
                results["stats"] = {
                    "file1_min": float(np.nanmin(data1)),
                    "file1_max": float(np.nanmax(data1)),
                    "file1_mean": float(np.nanmean(data1)),
                    "file2_min": float(np.nanmin(data2)),
                    "file2_max": float(np.nanmax(data2)),
                    "file2_mean": float(np.nanmean(data2)),
                }
                
                # Check data equality (with tolerance for floating point)
                if data1.shape == data2.shape:
                    # Compare with numpy's allclose (allows small floating point differences)
                    data_equal = np.allclose(data1, data2, rtol=1e-10, atol=1e-10, equal_nan=True)
                    results["data_match"] = bool(data_equal)
                    
                    if not data_equal:
                        # Compute difference statistics
                        valid_mask = ~(np.isnan(data1) | np.isnan(data2))
                        if np.any(valid_mask):
                            diff = np.abs(data1[valid_mask] - data2[valid_mask])
                            results["stats"]["max_difference"] = float(np.max(diff))
                            results["stats"]["mean_difference"] = float(np.mean(diff))
                            results["stats"]["percent_different"] = float(100 * np.sum(diff > 1e-10) / np.sum(valid_mask))
                            results["differences"].append(f"Data values differ (max diff: {results['stats']['max_difference']:.2e})")
                else:
                    results["differences"].append(f"Data shape mismatch prevents comparison")
                
                results["metadata_match"] = (
                    src1.dtypes[0] == src2.dtypes[0]
                    and (src1.height, src1.width) == (src2.height, src2.width)
                    and src1.count == src2.count
                )
        
        except Exception as e:
            results["differences"].append(f"Error comparing files: {str(e)}")
        
        return results
    
    def test_legacy_vs_optimized_output_identity(self):
        sample_radar_file = Path("data/netcdf/RMA1_0315_01_20251208T191648Z.nc")
        """
        Test that legacy and optimized implementations produce byte-for-byte identical outputs.
        
        This is a comprehensive equality test that verifies:
        1. Output file exists for both implementations
        2. File metadata matches (CRS, transform, shape, dtype)
        3. Data values are identical (within floating point tolerance)
        4. Statistics match (min, max, mean)
        """
        from radar_cog_processor.cache import GRID2D_CACHE, GRID3D_CACHE
        import shutil
        
        # Clean directories
        legacy_dir = Path("output/test_equality_legacy")
        optimized_dir = Path("output/test_equality_optimized")
        
        for d in [legacy_dir, optimized_dir]:
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)
        
        # Clear caches
        GRID2D_CACHE.clear()
        GRID3D_CACHE.clear()
        
        # Run legacy
        legacy_result = process_radar_to_cog_legacy(
            str(sample_radar_file),
            product="PPI",
            field_requested="DBZH",
            elevation=0,
            output_dir=str(legacy_dir)
        )
        legacy_cog = Path(legacy_result["image_url"])
        
        # Clear caches between runs
        GRID2D_CACHE.clear()
        GRID3D_CACHE.clear()
        
        # Run optimized
        optimized_result = process_radar_to_cog(
            str(sample_radar_file),
            product="PPI",
            field_requested="DBZH",
            elevation=0,
            output_dir=str(optimized_dir)
        )
        optimized_cog = Path(optimized_result["image_url"])
        
        # Verify files exist
        assert legacy_cog.exists(), f"Legacy COG not found: {legacy_cog}"
        assert optimized_cog.exists(), f"Optimized COG not found: {optimized_cog}"
        
        # Compare files
        comparison = self.compare_geotiff_files(legacy_cog, optimized_cog)
        
        # Assert equality
        assert comparison["crs_match"], f"CRS mismatch: {comparison['differences']}"
        assert comparison["transform_match"], f"Transform mismatch: {comparison['differences']}"
        assert comparison["metadata_match"], f"Metadata mismatch: {comparison['differences']}"
        assert comparison["data_match"], f"Data mismatch: {comparison['differences']}"
        
        # Verify statistics are identical
        for stat in ["min", "max", "mean"]:
            legacy_stat = comparison["stats"][f"file1_{stat}"]
            optimized_stat = comparison["stats"][f"file2_{stat}"]
            assert np.isclose(legacy_stat, optimized_stat, rtol=1e-10), \
                f"Statistic mismatch: {stat} differs ({legacy_stat} vs {optimized_stat})"
        
        print(f"\n✓ Output files are identical:")
        print(f"  Legacy:    {legacy_cog}")
        print(f"  Optimized: {optimized_cog}")
        print(f"  Shape:     {comparison['metadata']['file1_shape']}")
        print(f"  Data type: {comparison['metadata']['file1_dtype']}")
        print(f"  Min/Max:   {comparison['stats']['file1_min']:.2f} / {comparison['stats']['file1_max']:.2f}")
        print(f"  Mean:      {comparison['stats']['file1_mean']:.2f}")
    
    def test_output_files_with_filters(self):
        sample_radar_file = Path("data/netcdf/RMA1_0315_01_20251208T191648Z.nc")
        """Test that filtered outputs are also identical between implementations."""
        from radar_cog_processor.cache import GRID2D_CACHE, GRID3D_CACHE
        import shutil
        
        legacy_dir = Path("output/test_equality_filters_legacy")
        optimized_dir = Path("output/test_equality_filters_optimized")
        
        for d in [legacy_dir, optimized_dir]:
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)
        
        # Create simple filter object with required attributes
        class Filter:
            def __init__(self, field, min=None, max=None):
                self.field = field
                self.min = min
                self.max = max
        
        filters = [Filter(field="DBZH", min=10, max=50)]
        
        # Legacy with filters
        GRID2D_CACHE.clear()
        GRID3D_CACHE.clear()
        
        legacy_result = process_radar_to_cog_legacy(
            str(sample_radar_file),
            product="PPI",
            field_requested="DBZH",
            elevation=0,
            filters=filters,
            output_dir=str(legacy_dir)
        )
        
        # Optimized with filters
        GRID2D_CACHE.clear()
        GRID3D_CACHE.clear()
        
        optimized_result = process_radar_to_cog(
            str(sample_radar_file),
            product="PPI",
            field_requested="DBZH",
            elevation=0,
            filters=filters,
            output_dir=str(optimized_dir)
        )
        
        legacy_cog = Path(legacy_result["image_url"])
        optimized_cog = Path(optimized_result["image_url"])
        
        assert legacy_cog.exists(), f"Legacy COG with filters not found"
        assert optimized_cog.exists(), f"Optimized COG with filters not found"
        
        comparison = self.compare_geotiff_files(legacy_cog, optimized_cog)
        
        assert comparison["data_match"], f"Filtered output data mismatch: {comparison['differences']}"
        assert comparison["metadata_match"], f"Filtered output metadata mismatch: {comparison['differences']}"
        
        print(f"\n✓ Filtered output files are identical")
    
    def test_output_reproducibility(self):
        sample_radar_file = Path("data/netcdf/RMA1_0315_01_20251208T191648Z.nc")
        """Test that running the same implementation twice produces identical output."""
        from radar_cog_processor.cache import GRID2D_CACHE, GRID3D_CACHE
        import shutil
        
        output_dir1 = Path("output/test_reproducibility_1")
        output_dir2 = Path("output/test_reproducibility_2")
        
        for d in [output_dir1, output_dir2]:
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)
        
        # Run 1
        GRID2D_CACHE.clear()
        GRID3D_CACHE.clear()
        
        result1 = process_radar_to_cog(
            str(sample_radar_file),
            product="PPI",
            field_requested="DBZH",
            elevation=0,
            output_dir=str(output_dir1)
        )
        
        # Run 2
        GRID2D_CACHE.clear()
        GRID3D_CACHE.clear()
        
        result2 = process_radar_to_cog(
            str(sample_radar_file),
            product="PPI",
            field_requested="DBZH",
            elevation=0,
            output_dir=str(output_dir2)
        )
        
        cog1 = Path(result1["image_url"])
        cog2 = Path(result2["image_url"])
        
        assert cog1.exists() and cog2.exists(), "Output files not created"
        
        comparison = self.compare_geotiff_files(cog1, cog2)
        
        assert comparison["data_match"], f"Reproducibility check failed: {comparison['differences']}"
        
        print(f"\n✓ Optimized implementation is reproducible (identical outputs on consecutive runs)")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])