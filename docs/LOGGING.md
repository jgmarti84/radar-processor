# Logging in radar_grid Module

The `radar_grid` module uses Python's built-in `logging` library for all output messages. This allows you to control the verbosity and destination of log messages.

## Basic Usage

By default, logging is not configured. To see log messages, you need to configure logging in your application:

```python
import logging
from radar_grid import compute_grid_geometry, load_geometry

# Configure logging to see INFO and above messages
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Now radar_grid operations will log their progress
geometry = compute_grid_geometry(radar, grid_shape, grid_limits)
```

## Log Levels

The `radar_grid` module uses the following log levels:

- **DEBUG**: Detailed information for each processing step (e.g., per-level pair counts)
- **INFO**: General progress messages (e.g., "Processing 10 z-levels", "Merge complete")
- **WARNING**: Warnings about potential issues (e.g., altitude outside grid range)
- **ERROR**: Errors that prevent operation completion

## Controlling Verbosity

### Show Only Important Messages (WARNING and above)
```python
logging.basicConfig(level=logging.WARNING)
```

### Show Progress Messages (INFO and above)
```python
logging.basicConfig(level=logging.INFO)
```

### Show Detailed Debug Information
```python
logging.basicConfig(level=logging.DEBUG)
```

## Advanced Configuration

### Log to File
```python
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='radar_processing.log',
    filemode='w'
)
```

### Log to Both Console and File
```python
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)

# File handler
file_handler = logging.FileHandler('radar_processing.log')
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)

# Add handlers
logger.addHandler(console_handler)
logger.addHandler(file_handler)
```

### Control Specific Module Logging
```python
import logging

# Set global level
logging.basicConfig(level=logging.WARNING)

# But enable INFO for radar_grid.compute
logging.getLogger('radar_grid.compute').setLevel(logging.INFO)
```

## Disabling All radar_grid Logs
```python
import logging

# Disable all radar_grid logging
logging.getLogger('radar_grid').setLevel(logging.CRITICAL + 1)
```

## Example: Processing with Different Verbosity

```python
import logging
from radar_grid import compute_grid_geometry, apply_geometry
import pyart

# Quiet mode - only show warnings and errors
logging.basicConfig(level=logging.WARNING)

# Process silently
radar = pyart.io.read('radar_file.nc')
geometry = compute_grid_geometry(radar, grid_shape, grid_limits)

# ---

# Verbose mode - show all progress
logging.basicConfig(level=logging.INFO, force=True)

# Now you'll see progress messages
field_data = get_field_data(radar, 'DBZH')
grid = apply_geometry(geometry, field_data)
```

## Logger Names

All radar_grid modules use logger names in the format `radar_grid.<module>`:

- `radar_grid.compute`
- `radar_grid.geometry`
- `radar_grid.products`
- `radar_grid.interpolate`
- `radar_grid.filters`
- `radar_grid.geotiff`
- `radar_grid.utils`
- `radar_grid.mpl_visualization`

This allows you to control logging for specific modules independently.
