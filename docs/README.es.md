# Radar COG Processor

Una biblioteca Python de alto rendimiento para procesar datos de radar meteorológico desde archivos NetCDF a GeoTIFFs optimizados para la nube (COGs).

## Descripción General

Radar COG Processor convierte datos de escaneo volumétrico de radar en imágenes ráster georreferenciadas, aptas para mapeo web, aplicaciones GIS y análisis automatizado. Soporta múltiples productos de radar, campos y incluye control de calidad sofisticado mediante filtros.

### Características Principales

- **Múltiples Tipos de Productos**: PPI, CAPPI y COLMAX
- **Campos de Doble Polarización**: DBZH, ZDR, RHOHV, KDP, VRAD, WRAD, PHIDP
- **Filtrado de Dos Niveles**: Filtros QC y Filtros Visuales
- **Caché Inteligente**: Caché de mallas 2D y 3D para rendimiento
- **Soporte de Mapas de Colores**: Múltiples mapas de colores por tipo de campo
- **Salida Optimizada para la Nube**: Salida GeoTIFF optimizada para servicio web

## Instalación

### Inicio Rápido

```bash
# Clonar el repositorio
git clone <repository-url>
cd radar-processor

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Instalar paquete
pip install -e .
```

### Requisitos

- Python 3.9+
- Bibliotecas GDAL/OGR
- ARM PyART (para procesamiento de radar)
- xarray, numpy, scipy

## Inicio Rápido

### Uso Básico

```python
from radar_processor import process_radar_to_cog

# Procesar un archivo de radar a COG
result = process_radar_to_cog(
    filepath="data/netcdf/archivo_radar.nc",
    product="PPI",
    field_requested="DBZH",
    elevation=0,
    output_dir="output"
)

print(f"Creado: {result['image_url']}")
```

### Con Filtrado de Calidad

```python
# Definir un filtro (debe tener atributos field, min, max)
class Filter:
    def __init__(self, field: str, min: float = None, max: float = None):
        self.field = field
        self.min = min
        self.max = max

# Aplicar control de calidad
result = process_radar_to_cog(
    filepath="data/netcdf/archivo_radar.nc",
    product="PPI",
    field_requested="DBZH",
    elevation=0,
    filters=[Filter("RHOHV", min=0.85)],  # Eliminar ecos no meteorológicos
    output_dir="output"
)
```

## Productos

La biblioteca soporta tres tipos de productos de radar, cada uno adecuado para diferentes aplicaciones:

### PPI (Indicador de Posición en Plano)

Un corte horizontal a un ángulo de elevación específico. Es la visualización de radar más común.

```python
result = process_radar_to_cog(
    filepath="archivo_radar.nc",
    product="PPI",
    field_requested="DBZH",
    elevation=0,  # Ángulo de elevación más bajo
    output_dir="output"
)
```

**Características:**
- La altura varía con la distancia al radar
- Muestra la vista del radar en un ángulo de escaneo
- Procesamiento rápido (solo malla 2D)

**Casos de Uso:**
- Monitoreo en tiempo real
- Seguimiento de tormentas
- Visión general del tiempo

### CAPPI (PPI de Altitud Constante)

Un corte horizontal a altura constante sobre el nivel del suelo/mar.

```python
result = process_radar_to_cog(
    filepath="archivo_radar.nc",
    product="CAPPI",
    field_requested="DBZH",
    cappi_height=3000,  # 3km de altitud
    output_dir="output"
)
```

**Características:**
- Altura consistente en todas partes
- Interpolado desde múltiples ángulos de elevación
- Requiere mallado 3D

**Casos de Uso:**
- Meteorología aeronáutica
- Aplicaciones de investigación
- Análisis de altura consistente

### COLMAX (Máximo de Columna)

Valor máximo en cada columna vertical.

```python
result = process_radar_to_cog(
    filepath="archivo_radar.nc",
    product="COLMAX",
    field_requested="DBZH",
    output_dir="output"
)
```

**Características:**
- Muestra el eco más fuerte sin importar la altura
- Bueno para detección de tiempo severo
- Requiere mallado 3D

**Casos de Uso:**
- Detección de tiempo severo
- Identificación de granizo
- Clasificación de intensidad de tormentas

## Campos

Campos de medición de radar disponibles:

| Campo | Nombre | Unidad | Descripción |
|-------|--------|--------|-------------|
| DBZH | Reflectividad Horizontal | dBZ | Intensidad de precipitación |
| DBZV | Reflectividad Vertical | dBZ | Reflectividad de polarización vertical |
| ZDR | Reflectividad Diferencial | dB | Indicador de forma/tamaño de gotas |
| RHOHV | Coeficiente de Correlación Cruzada | 0-1 | Indicador de calidad de datos |
| KDP | Fase Diferencial Específica | deg/km | Estimación de tasa de lluvia |
| VRAD | Velocidad Radial | m/s | Movimiento hacia/desde el radar |
| WRAD | Ancho Espectral | m/s | Indicador de turbulencia |
| PHIDP | Fase Diferencial | grados | Diferencia de fase acumulativa |

### Selección de Campos

```python
# Procesar diferentes campos
for field in ["DBZH", "ZDR", "VRAD"]:
    result = process_radar_to_cog(
        filepath="archivo_radar.nc",
        product="PPI",
        field_requested=field,
        elevation=0,
        output_dir=f"output/{field}"
    )
```

## Sistema de Filtrado

La biblioteca usa un **sistema de filtrado de dos niveles** para control de calidad:

### 1. Filtros QC (Control de Calidad)

Los filtros QC afectan el **proceso de mallado/interpolación**. Usan campos como RHOHV para excluir puntos de datos de baja calidad antes de crear la malla.

**Campos QC soportados actualmente:** `RHOHV`

```python
# El filtro RHOHV elimina ecos no meteorológicos
filtro_qc = Filter("RHOHV", min=0.85)
```

**Cómo funcionan los filtros QC:**
1. Se aplican durante la fase de construcción de la malla
2. Las puertas (gates) con valores fuera del rango del filtro se excluyen de la interpolación
3. Afectan la clave de caché (diferentes filtros = diferentes mallas)

### 2. Filtros Visuales

Los filtros visuales se aplican **después del mallado** para enmascarar valores en el producto final. Funcionan sobre el campo principal que se está visualizando.

```python
# Mostrar solo precipitación moderada a intensa
filtro_visual = Filter("DBZH", min=20, max=55)
```

**Cómo funcionan los filtros visuales:**
1. Se aplican a los datos mallados finales
2. Enmascaran valores fuera del rango especificado
3. No afectan el caché

### Filtrado Combinado

Puedes usar ambos tipos de filtro juntos:

```python
# Filtro QC (RHOHV) + Filtro visual (rango DBZH)
filters = [
    Filter("RHOHV", min=0.85),     # Eliminar clutter, pájaros, etc.
    Filter("DBZH", min=15, max=60) # Mostrar solo precipitación significativa
]

result = process_radar_to_cog(
    filepath="archivo_radar.nc",
    product="PPI",
    field_requested="DBZH",
    elevation=0,
    filters=filters,
    output_dir="output"
)
```

### Recomendaciones de Filtros

| Caso de Uso | Filtros Recomendados |
|-------------|---------------------|
| Monitoreo general | `RHOHV >= 0.85` |
| Investigación (alta calidad) | `RHOHV >= 0.92` |
| Tiempo severo | `RHOHV >= 0.85, DBZH >= 40` |
| Precipitación ligera | `RHOHV >= 0.85, DBZH 0-30` |
| Solo precipitación | `RHOHV >= 0.85, DBZH >= 10` |

## Caché

La biblioteca implementa caché inteligente para evitar cálculos redundantes:

### Caché de Dos Niveles

1. **GRID2D_CACHE**: Cachea mallas 2D (para PPI)
2. **GRID3D_CACHE**: Cachea mallas 3D (para CAPPI/COLMAX)

### Comportamiento del Caché

- La misma malla se reutiliza cuando solo cambian los filtros visuales
- Los filtros QC afectan la clave de caché (diferente QC = diferente malla)
- Diferentes productos pueden compartir la misma malla 3D

### Gestión del Caché

```python
from radar_processor.cache import GRID2D_CACHE, GRID3D_CACHE

# Limpiar cachés entre archivos o para gestión de memoria
GRID2D_CACHE.clear()
GRID3D_CACHE.clear()
```

## Opciones Avanzadas

### Mapas de Colores Personalizados

Anular el mapa de colores predeterminado para un campo:

```python
result = process_radar_to_cog(
    filepath="archivo_radar.nc",
    product="PPI",
    field_requested="DBZH",
    elevation=0,
    colormap_overrides={"DBZH": "pyart_HomeyerRainbow"},
    output_dir="output"
)
```

### Configuración de Malla

Personalizar los parámetros de la malla de salida:

```python
result = process_radar_to_cog(
    filepath="archivo_radar.nc",
    product="PPI",
    field_requested="DBZH",
    elevation=0,
    grid_shape=(500, 500),  # Resolución de salida
    grid_limits=((-250000, 250000), (-250000, 250000)),  # Extensión en metros
    output_dir="output"
)
```

## Referencia de API

### Función Principal

```python
process_radar_to_cog(
    filepath: str,
    product: str = "PPI",
    field_requested: str = "DBZH",
    elevation: int = 0,
    cappi_height: float = 4000,
    filters: List[Filter] = None,
    colormap_overrides: Dict[str, str] = None,
    grid_shape: Tuple[int, int] = None,
    grid_limits: Tuple[Tuple[float, float], Tuple[float, float]] = None,
    output_dir: str = "output"
) -> Dict[str, Any]
```

**Parámetros:**
- `filepath`: Ruta al archivo NetCDF de radar
- `product`: Tipo de producto ("PPI", "CAPPI", "COLMAX")
- `field_requested`: Campo de radar a procesar
- `elevation`: Índice de elevación para PPI (0 = más bajo)
- `cappi_height`: Altura en metros para CAPPI
- `filters`: Lista de objetos de filtro
- `colormap_overrides`: Dict mapeando nombres de campo a nombres de mapa de colores
- `grid_shape`: Dimensiones de la malla de salida (y, x)
- `grid_limits`: Extensión de la malla en metros ((y_min, y_max), (x_min, x_max))
- `output_dir`: Ruta del directorio de salida

**Retorna:**
Diccionario conteniendo:
- `image_url`: Ruta al archivo COG generado
- Metadatos adicionales

## Ejemplos

Ver el directorio `examples/` para ejemplos completos funcionales:

- `basic_usage.py` - Ejemplos simples de procesamiento
- `advanced_usage.py` - Configuración avanzada
- `filter_examples.py` - Demostraciones completas de filtros
- `all_options.py` - Todas las combinaciones de producto/campo/filtro
- `products_explained.py` - Explicación detallada de tipos de producto
- `fields_explained.py` - Descripciones de campos de radar
- `batch_processing.py` - Procesamiento de múltiples archivos

## Rendimiento

### Consejos de Optimización

1. **Reutilizar mallas**: Procesar múltiples campos desde la misma malla
2. **Limpiar cachés**: Limpiar cachés entre diferentes archivos
3. **Usar el producto apropiado**: PPI es más rápido que CAPPI/COLMAX
4. **Filtrar temprano**: Los filtros QC reducen datos antes del mallado

### Benchmarking

La biblioteca incluye herramientas de benchmarking:

```bash
python scripts/benchmark_compare.py --runs 3 data/netcdf/*.nc
```

## Pruebas

```bash
# Ejecutar todas las pruebas
pytest tests/

# Ejecutar con cobertura
pytest tests/ --cov=radar_processor

# Ejecutar archivo de prueba específico
pytest tests/test_processor_phases.py
```

## Estructura del Proyecto

```
radar-processor/
├── src/radar_processor/
│   ├── __init__.py         # Exportaciones del paquete
│   ├── processor.py        # Pipeline de procesamiento principal
│   ├── processor_legacy.py # Implementación legacy (referencia)
│   ├── utils.py            # Funciones utilitarias
│   ├── colormaps.py        # Definiciones de mapas de colores
│   ├── constants.py        # Constantes de configuración
│   └── cache.py            # Sistema de caché de mallas
├── examples/               # Ejemplos de uso
├── tests/                  # Suite de pruebas
├── scripts/                # Scripts de benchmarking
├── data/                   # Datos de ejemplo
└── docs/                   # Documentación
```

## Contribuir

1. Hacer fork del repositorio
2. Crear una rama de característica
3. Hacer tus cambios
4. Ejecutar pruebas: `pytest tests/`
5. Enviar un pull request

## Licencia

Ver archivo LICENSE para detalles.

## Registro de Cambios

Ver CHANGELOG.md para historial de versiones.
