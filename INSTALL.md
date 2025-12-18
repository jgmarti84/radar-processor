# Radar COG Processor - Installation & Usage Guide

## Quick Installation

### Option 1: Install from local source

```bash
cd radar-cog-processor
pip install -e .
```

### Option 2: Install with development dependencies

```bash
cd radar-cog-processor
pip install -e ".[dev]"
```

### Option 3: Install from requirements.txt

```bash
pip install -r requirements.txt
pip install -e .
```

## System Dependencies

### GDAL Installation

The package requires GDAL/rasterio which needs system-level dependencies.

**macOS (with Homebrew):**
```bash
brew install gdal
```

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install gdal-bin libgdal-dev libhdf5-dev libnetcdf-dev
```

**Conda (recommended for all platforms):**
```bash
conda create -n radar-cog python=3.11
conda activate radar-cog
conda install -c conda-forge gdal rasterio pyproj shapely
pip install -e .
```

## Verifying Installation

```python
import radar_cog_processor
print(radar_cog_processor.__version__)

# Test import of main function
from radar_cog_processor import process_radar_to_cog
print("Installation successful!")
```

## Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=radar_cog_processor --cov-report=html

# View coverage report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

## Running Examples

```bash
cd examples

# Basic usage (requires actual radar file)
python basic_usage.py

# Advanced features
python advanced_usage.py

# Batch processing
python batch_processing.py
```

**Note:** Examples require actual radar NetCDF files. Update file paths in the scripts.

## Building Distribution Packages

```bash
# Install build tools
pip install build twine

# Build source distribution and wheel
python -m build

# Check distribution
twine check dist/*

# Upload to PyPI (when ready)
# twine upload dist/*
```

## Project Structure

```
radar-cog-processor/
├── src/
│   └── radar_cog_processor/
│       ├── __init__.py          # Package initialization
│       ├── processor.py         # Main processing functions
│       ├── constants.py         # Field definitions and constants
│       ├── cache.py            # LRU cache implementation
│       ├── utils.py            # Helper functions
│       └── colormaps.py        # Custom colormap definitions
├── tests/
│   ├── conftest.py             # Pytest configuration
│   ├── test_cache.py           # Cache tests
│   ├── test_colormaps.py       # Colormap tests
│   └── test_utils.py           # Utility function tests
├── examples/
│   ├── basic_usage.py          # Simple example
│   ├── advanced_usage.py       # Advanced features
│   └── batch_processing.py     # Batch processing
├── pyproject.toml              # Package metadata and config
├── setup.py                    # Setup script
├── requirements.txt            # Dependencies
├── README.md                   # Main documentation
├── LICENSE                     # MIT License
└── CHANGELOG.md               # Version history
```

## Common Issues and Solutions

### Issue: GDAL import error
```
ImportError: GDAL library not found
```
**Solution:** Install GDAL system dependencies (see above)

### Issue: PyART import error
```
ModuleNotFoundError: No module named 'pyart'
```
**Solution:** Install PyART:
```bash
pip install arm-pyart
# Or use custom fork:
pip install git+https://github.com/IgnaCat/pyart.git@84f411ae86e05b14fc075b8f6535af84c8bba2c9
```

### Issue: Rasterio CRS error
```
CRSError: Invalid projection
```
**Solution:** Ensure pyproj and proj.db are correctly installed:
```bash
conda install -c conda-forge pyproj
```

### Issue: Out of memory with large files
**Solution:** Adjust cache sizes in code or process files sequentially:
```python
from radar_cog_processor.cache import GRID2D_CACHE, GRID3D_CACHE

# Reduce cache sizes
GRID2D_CACHE.maxsize = 100 * 1024 * 1024  # 100 MB
GRID3D_CACHE.maxsize = 300 * 1024 * 1024  # 300 MB
```

## Development Workflow

1. **Clone and install in editable mode:**
```bash
git clone <your-repo-url>
cd radar-cog-processor
pip install -e ".[dev]"
```

2. **Make changes to code**

3. **Run tests:**
```bash
pytest tests/
```

4. **Format code:**
```bash
black src/radar_cog_processor tests examples
```

5. **Check code quality:**
```bash
flake8 src/radar_cog_processor
```

## Using in Another Project

After installing the package, use it in your code:

```python
from radar_cog_processor import process_radar_to_cog

result = process_radar_to_cog(
    filepath="path/to/radar.nc",
    product="PPI",
    field_requested="DBZH",
    elevation=0,
    output_dir="output"
)
```

## Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Check existing documentation in README.md
- Review examples in the `examples/` directory

## Next Steps

1. Install the package following instructions above
2. Run tests to verify installation
3. Try the basic example with your radar data
4. Explore advanced features and batch processing
5. Customize for your specific use case

Happy radar processing! 🌩️📡
