"""
Batch processing example: Process multiple radar files.

This example demonstrates how to process multiple radar files
in batch mode, useful for operational processing or research.
"""
from radar_processor import process_radar_to_cog
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import time


class RangeFilter:
    """Simple filter class."""
    def __init__(self, field, min_val=None, max_val=None):
        self.field = field
        self.min = min_val
        self.max = max_val


def process_single_file(radar_file, output_dir, field="DBZH", product="PPI"):
    """Process a single radar file."""
    try:
        result = process_radar_to_cog(
            filepath=str(radar_file),
            product=product,
            field_requested=field,
            elevation=0,
            output_dir=str(output_dir)
        )
        return {
            "success": True,
            "file": radar_file.name,
            "result": result
        }
    except Exception as e:
        return {
            "success": False,
            "file": radar_file.name,
            "error": str(e)
        }


def batch_process_serial(radar_files, output_dir, field="DBZH"):
    """Process multiple files serially."""
    print(f"Processing {len(radar_files)} files serially...")
    start_time = time.time()
    
    results = []
    for i, radar_file in enumerate(radar_files, 1):
        print(f"Processing {i}/{len(radar_files)}: {radar_file.name}")
        result = process_single_file(radar_file, output_dir, field)
        results.append(result)
    
    elapsed = time.time() - start_time
    print(f"Completed in {elapsed:.2f} seconds")
    return results


def batch_process_parallel(radar_files, output_dir, field="DBZH", max_workers=4):
    """Process multiple files in parallel."""
    print(f"Processing {len(radar_files)} files in parallel (max_workers={max_workers})...")
    start_time = time.time()
    
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(process_single_file, f, output_dir, field): f
            for f in radar_files
        }
        
        # Collect results as they complete
        for i, future in enumerate(as_completed(future_to_file), 1):
            radar_file = future_to_file[future]
            try:
                result = future.result()
                results.append(result)
                status = "✓" if result["success"] else "✗"
                print(f"{status} {i}/{len(radar_files)}: {radar_file.name}")
            except Exception as e:
                print(f"✗ {i}/{len(radar_files)}: {radar_file.name} - {e}")
                results.append({
                    "success": False,
                    "file": radar_file.name,
                    "error": str(e)
                })
    
    elapsed = time.time() - start_time
    print(f"Completed in {elapsed:.2f} seconds")
    return results


def process_directory(input_dir, output_dir, pattern="*.nc", parallel=True):
    """Process all radar files in a directory."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all radar files
    radar_files = list(input_path.glob(pattern))
    
    if not radar_files:
        print(f"No files found matching pattern '{pattern}' in {input_dir}")
        return []
    
    print(f"Found {len(radar_files)} files")
    
    # Process files
    if parallel:
        results = batch_process_parallel(radar_files, output_path)
    else:
        results = batch_process_serial(radar_files, output_path)
    
    # Print summary
    successful = sum(1 for r in results if r["success"])
    failed = len(results) - successful
    
    print(f"\n=== Summary ===")
    print(f"Total files: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    if failed > 0:
        print("\nFailed files:")
        for r in results:
            if not r["success"]:
                print(f"  - {r['file']}: {r['error']}")
    
    return results


def process_with_multiple_products(radar_files, output_dir):
    """Process files with multiple products."""
    products = [
        ("PPI", {"elevation": 0}),
        ("PPI", {"elevation": 1}),
        ("CAPPI", {"cappi_height": 4000}),
        ("COLMAX", {}),
    ]
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    for product_name, kwargs in products:
        print(f"\n=== Processing {product_name} ===")
        product_dir = output_path / product_name.lower()
        product_dir.mkdir(exist_ok=True)
        
        for radar_file in radar_files:
            try:
                result = process_radar_to_cog(
                    filepath=str(radar_file),
                    product=product_name,
                    field_requested="DBZH",
                    output_dir=str(product_dir),
                    **kwargs
                )
                all_results.append({
                    "success": True,
                    "file": radar_file.name,
                    "product": product_name,
                    "result": result
                })
                print(f"✓ {radar_file.name}")
            except Exception as e:
                all_results.append({
                    "success": False,
                    "file": radar_file.name,
                    "product": product_name,
                    "error": str(e)
                })
                print(f"✗ {radar_file.name}: {e}")
    
    return all_results


def main():
    # Example 1: Process all files in a directory
    print("=== Example 1: Process Directory ===")
    results = process_directory(
        input_dir="path/to/radar/files",
        output_dir="output/batch",
        pattern="*.nc",
        parallel=True  # Use parallel processing
    )
    
    # Example 2: Process with multiple products
    print("\n=== Example 2: Multiple Products ===")
    input_path = Path("path/to/radar/files")
    radar_files = list(input_path.glob("*.nc"))[:5]  # First 5 files
    
    if radar_files:
        results = process_with_multiple_products(
            radar_files=radar_files,
            output_dir="output/multi_product"
        )


if __name__ == "__main__":
    main()
