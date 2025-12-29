# Radar COG Processor - Referencia de API

Una biblioteca Python para convertir archivos NetCDF de radar meteorológico a GeoTIFFs Optimizados para la Nube (COGs).

## Tabla de Contenidos

1. [Descripción General](#descripción-general)
2. [Instalación](#instalación)
3. [Inicio Rápido](#inicio-rápido)
4. [Funciones Principales](#funciones-principales)
5. [Productos](#productos)
6. [Campos](#campos)
7. [Filtros](#filtros)
8. [Mapas de Colores](#mapas-de-colores)
9. [Caché](#caché)
10. [Ejemplos](#ejemplos)

---

## Descripción General

La biblioteca `radar_processor` convierte datos de radar meteorológico desde formato NetCDF (comúnmente usado por sistemas de radar como RMA/ARM) a GeoTIFFs Optimizados para la Nube (COGs). Los COGs están optimizados para mapeo web y almacenamiento en la nube, permitiendo acceso eficiente basado en tiles.

### Características Principales

- **Múltiples Productos**: Visualizaciones PPI, CAPPI y COLMAX
- **Múltiples Campos**: DBZH (reflectividad), ZDR, RHOHV, KDP, VRAD y más
- **Filtrado de Datos**: Filtros de control de calidad (QC) y filtros de rango visual
- **Caché de Dos Niveles**: Rendimiento optimizado con caché de mallas 3D y 2D
- **Mapas de Colores Profesionales**: Mapas de colores personalizados para radar meteorológico

### Arquitectura

```
┌──────────────────────────────────────────────────────────────────┐
│                    process_radar_to_cog()                        │
├──────────────────────────────────────────────────────────────────┤
│  Fase 1-2: Configuración y Validación                            │
│  Fase 3: Config de Renderizado (mapas de colores, vmin/vmax)    │
│  Fase 4-5: Preparación de Datos y Config de Producto            │
│  Fase 6: Configuración de Malla                                  │
│  Fase 7: Clasificación de Filtros (QC vs Visual)                │
│  Fase 8-9: Malla 3D → Colapso 2D (cacheado)                     │
│  Fase 10-11: Aplicar Filtros (post-mallado)                     │
│  Fase 12: Exportar a GeoTIFF → COG                              │
└──────────────────────────────────────────────────────────────────┘
```

---

## Instalación

```bash
pip install -e .
```

### Dependencias

- Python 3.10+
- PyART (ARM Radar Toolkit)
- GDAL 3.x
- NumPy, Rasterio, PyProj

---

## Inicio Rápido

```python
from radar_processor import process_radar_to_cog

# Uso básico
result = process_radar_to_cog(
    filepath="archivo_radar.nc",
    product="PPI",
    field_requested="DBZH",
    elevation=0,
    output_dir="output"
)

print(result['image_url'])  # Ruta al archivo COG creado
```

---

## Funciones Principales

### `process_radar_to_cog()`

La función principal de procesamiento.

```python
def process_radar_to_cog(
    filepath,                    # Ruta al archivo NetCDF de radar
    product="PPI",               # Tipo de producto: 'PPI', 'CAPPI', 'COLMAX'
    field_requested="DBZH",      # Campo a procesar
    cappi_height=4000,           # Altura (m) para producto CAPPI
    elevation=0,                 # Índice de elevación para PPI
    filters=None,                # Lista de objetos de filtro
    output_dir="output",         # Directorio de salida
    volume=None,                 # Identificador de volumen para resolución
    colormap_overrides=None      # Mapeo de mapa de colores personalizado
) -> dict
```

#### Retorna

```python
{
    "image_url": "/ruta/a/output/radar_DBZH_PPI_nofilter_0_abc123.tif",
    "field": "DBZH",
    "source_file": "/ruta/a/input.nc",
    "tilejson_url": "placeholder/tilejson.json?url=..."
}
```

---

## Productos

La biblioteca soporta tres productos de visualización de radar:

### PPI - Indicador de Posición en Plano

Un corte horizontal a través del volumen de radar en un ángulo de elevación específico.

```python
result = process_radar_to_cog(
    filepath="radar.nc",
    product="PPI",
    field_requested="DBZH",
    elevation=0,       # 0 = elevación más baja, 1 = segunda, etc.
    output_dir="output"
)
```

**Caso de Uso**: Ver patrones de precipitación en un ángulo de escaneo específico (inclinación).

### CAPPI - Indicador de Posición en Plano de Altitud Constante

Un corte horizontal a una altitud constante sobre el nivel del suelo.

```python
result = process_radar_to_cog(
    filepath="radar.nc",
    product="CAPPI",
    field_requested="DBZH",
    cappi_height=4000,  # Altura en metros sobre el radar
    output_dir="output"
)
```

**Caso de Uso**: Vista de altitud consistente útil para aviación y análisis atmosférico.

### COLMAX - Máximo de Columna

Valor máximo en cada columna vertical (como una vista de máximo "desde arriba").

```python
result = process_radar_to_cog(
    filepath="radar.nc",
    product="COLMAX",
    field_requested="DBZH",
    output_dir="output"
)
```

**Caso de Uso**: Identificar la intensidad máxima en cualquier lugar de la columna atmosférica.

---

## Campos

Los datos de radar contienen múltiples campos de medición:

| Campo   | Nombre                              | Unidades | Rango Típico  | Caso de Uso                        |
|---------|-------------------------------------|----------|---------------|-------------------------------------|
| DBZH    | Reflectividad Horizontal            | dBZ      | -30 a 70      | Intensidad de precipitación         |
| DBZV    | Reflectividad Vertical              | dBZ      | -30 a 70      | Precipitación (pol vertical)        |
| ZDR     | Reflectividad Diferencial           | dB       | -5 a 10.5     | Tamaño/forma de gotas               |
| RHOHV   | Coeficiente de Correlación Cruzada  | -        | 0 a 1         | Tipo/calidad de precipitación       |
| KDP     | Fase Diferencial Específica         | deg/km   | 0 a 8         | Estimación de tasa de lluvia        |
| VRAD    | Velocidad Radial                    | m/s      | -35 a 35      | Detección de viento/movimiento      |
| WRAD    | Ancho Espectral                     | m/s      | 0 a 10        | Indicación de turbulencia           |
| PHIDP   | Fase Diferencial                    | deg      | 0 a 360       | Longitud de trayectoria de precip   |

### Alias de Campos

La biblioteca soporta múltiples convenciones de nombres:

```python
# Todos estos funcionan para el mismo campo:
field_requested="DBZH"
field_requested="corrected_reflectivity_horizontal"
```

---

## Filtros

Los filtros son **la característica más importante** para controlar la calidad de datos y visualización.

### Entendiendo los Filtros

Los filtros enmascaran (ocultan) valores de datos basados en condiciones. Hay **dos tipos**:

```
┌────────────────────────────────────────────────────────────────────────┐
│                           TIPOS DE FILTROS                              │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  1. FILTROS QC (Control de Calidad)                             │   │
│  │                                                                  │   │
│  │  • Aplicados DURANTE el mallado (afecta interpolación)         │   │
│  │  • Filtran basados en campos QC como RHOHV                      │   │
│  │  • Eliminan ecos no meteorológicos (pájaros, clutter, ruido)   │   │
│  │  • Solo RHOHV es actualmente un campo QC                        │   │
│  │                                                                  │   │
│  │  Ejemplo: RHOHV < 0.8 → probablemente NO precipitación → OCULTAR│   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  2. FILTROS VISUALES (Rango de Valores)                         │   │
│  │                                                                  │   │
│  │  • Aplicados DESPUÉS del mallado (post-procesamiento)          │   │
│  │  • Filtran basados en el MISMO campo siendo visualizado        │   │
│  │  • Recortan rangos de valores para visualización               │   │
│  │  • Útiles para destacar rangos de intensidad específicos       │   │
│  │                                                                  │   │
│  │  Ejemplo: Mostrar solo DBZH entre 20 y 50 dBZ                   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└────────────────────────────────────────────────────────────────────────┘
```

### Creando Objetos de Filtro

Los filtros son objetos con tres atributos: `field`, `min`, `max`.

```python
# Clase de filtro simple (puedes usar cualquier clase con estos atributos)
class Filter:
    def __init__(self, field, min=None, max=None):
        self.field = field
        self.min = min
        self.max = max

# O usar una tupla nombrada
from collections import namedtuple
Filter = namedtuple('Filter', ['field', 'min', 'max'])

# O una dataclass
from dataclasses import dataclass

@dataclass
class Filter:
    field: str
    min: float = None
    max: float = None
```

### Filtro QC: Ejemplo RHOHV

RHOHV (Coeficiente de Correlación Cruzada) es el campo principal de control de calidad.

**¿Qué es RHOHV?**
- Los valores van de 0 a 1
- **RHOHV > 0.85-0.90**: Probablemente objetivos meteorológicos (lluvia, nieve)
- **RHOHV < 0.80**: Probablemente no meteorológico (clutter de tierra, pájaros, insectos)

```python
# Eliminar ecos de no-precipitación
filtro_rhohv = Filter(field="RHOHV", min=0.85, max=1.0)

result = process_radar_to_cog(
    filepath="radar.nc",
    product="PPI",
    field_requested="DBZH",
    filters=[filtro_rhohv],
    output_dir="output"
)
```

**Efecto**: Cualquier píxel de radar donde RHOHV < 0.85 o RHOHV > 1.0 será enmascarado (transparente/oculto).

### Filtro Visual: Ejemplo de Rango DBZH

```python
# Mostrar solo precipitación moderada a intensa (20-50 dBZ)
filtro_dbzh = Filter(field="DBZH", min=20, max=50)

result = process_radar_to_cog(
    filepath="radar.nc",
    product="PPI",
    field_requested="DBZH",
    filters=[filtro_dbzh],
    output_dir="output"
)
```

**Efecto**: 
- Valores < 20 dBZ (llovizna ligera) → enmascarado
- Valores > 50 dBZ (extremadamente intenso) → enmascarado
- Valores 20-50 dBZ → visible

### Combinando Múltiples Filtros

```python
filters = [
    # QC: Eliminar ecos no meteorológicos
    Filter(field="RHOHV", min=0.85, max=1.0),
    
    # Visual: Mostrar solo precipitación significativa
    Filter(field="DBZH", min=15, max=60),
]

result = process_radar_to_cog(
    filepath="radar.nc",
    product="PPI",
    field_requested="DBZH",
    filters=filters,
    output_dir="output"
)
```

### Lógica de Clasificación de Filtros

La biblioteca clasifica automáticamente los filtros:

```python
# Lógica interna (simplificada):
AFFECTS_INTERP_FIELDS = {"RHOHV"}  # Campos QC

for filter in filters:
    if filter.field.upper() in AFFECTS_INTERP_FIELDS:
        qc_filters.append(filter)      # Aplicado durante mallado
    else:
        visual_filters.append(filter)  # Aplicado después del mallado
```

### ¿Por Qué Dos Tipos de Filtros?

```
Filtros QC (durante mallado):
─────────────────────────────
• Aplicados ANTES de la interpolación a la malla Cartesiana
• Afecta qué puertas de radar polares contribuyen a los píxeles mallados
• Importante para eliminar datos contaminados antes de la interpolación
• Actualmente solo RHOHV (identificación de eco meteorológico)

Filtros Visuales (después del mallado):
────────────────────────────────────────
• Aplicados DESPUÉS de que la malla es computada
• Solo enmascara valores de visualización, no afecta interpolación
• Puede filtrar en cualquier campo
• Más rápido de aplicar (no requiere re-mallado)
```

### Casos de Uso de Filtros

| Escenario | Filtro | Por Qué |
|-----------|--------|---------|
| Eliminar clutter de tierra | `Filter("RHOHV", 0.85, 1.0)` | RHOHV bajo = no meteorológico |
| Ocultar precipitación ligera | `Filter("DBZH", 10, None)` | Enfocarse en ecos significativos |
| Mostrar solo lluvia moderada | `Filter("DBZH", 20, 45)` | ~4-50 mm/hr de lluvia |
| Destacar regiones de granizo | `Filter("DBZH", 55, None)` | Reflectividad muy alta |
| Umbral de análisis de viento | `Filter("VRAD", -25, 25)` | Eliminar velocidades extremas |

---

## Mapas de Colores

### Mapas de Colores Predeterminados por Campo

| Campo | Mapa de Colores | Descripción |
|-------|-----------------|-------------|
| DBZH  | `grc_th`        | Reflectividad estándar (verde-amarillo-rojo) |
| ZDR   | `grc_zdr2`      | Reflectividad diferencial |
| RHOHV | `grc_rho`       | Coeficiente de correlación |
| KDP   | `grc_rain`      | Fase diferencial específica |
| VRAD  | `NWSVel`        | Velocidad (azul-blanco-rojo) |
| WRAD  | `Oranges`       | Ancho espectral |

### Anulando Mapas de Colores

```python
result = process_radar_to_cog(
    filepath="radar.nc",
    product="PPI",
    field_requested="DBZH",
    colormap_overrides={"DBZH": "pyart_HomeyerRainbow"},
    output_dir="output"
)
```

### Mapas de Colores Disponibles

**Mapas de colores de radar personalizados (grc_*):**
- `grc_th`, `grc_th2` - Reflectividad
- `grc_zdr`, `grc_zdr2` - Reflectividad diferencial
- `grc_rho` - Coeficiente de correlación
- `grc_rain` - Tasa de lluvia

**Mapas de colores PyART (pyart_*):**
- `pyart_NWSRef` - Reflectividad NWS
- `pyart_HomeyerRainbow` - Reflectividad arcoíris
- `pyart_BuDRd18` - Velocidad

**Mapas de colores Matplotlib:**
- `viridis`, `plasma`, `Oranges`, `RdBu_r`, etc.

---

## Caché

La biblioteca implementa caché de dos niveles para rendimiento:

### Niveles de Caché

1. **Caché de Malla 3D** - Cacheado después de mallar datos polares a Cartesianos
2. **Caché de Malla 2D** - Cacheado después de colapsar 3D a 2D

### Claves de Caché

Las claves de caché se basan en:
- Hash del archivo (MD5)
- Tipo de producto (PPI/CAPPI/COLMAX)
- Nombre del campo
- Elevación/altura
- Resolución de malla
- Firma de filtro QC

### Limpiando el Caché

```python
from radar_processor.cache import GRID2D_CACHE, GRID3D_CACHE

# Limpiar todas las mallas cacheadas
GRID2D_CACHE.clear()
GRID3D_CACHE.clear()
```

---

## Ejemplos

Ver el directorio `examples/` para ejemplos completos funcionales:

- `examples/basic_usage.py` - Procesamiento simple de un archivo
- `examples/advanced_usage.py` - Filtros, mapas de colores, múltiples productos
- `examples/batch_processing.py` - Procesar múltiples archivos
- `examples/filter_examples.py` - Demostraciones completas de filtros
- `examples/all_options.py` - Cada combinación de opciones

---

## Manejo de Errores

```python
from radar_processor import process_radar_to_cog

try:
    result = process_radar_to_cog(...)
except ValueError as e:
    # Elevación, producto o configuración inválidos
    print(f"Error de configuración: {e}")
except KeyError as e:
    # Campo no encontrado en el archivo
    print(f"Campo no disponible: {e}")
except FileNotFoundError as e:
    # Archivo de entrada no encontrado
    print(f"Archivo no encontrado: {e}")
```

---

## Consejos de Rendimiento

1. **Usar caché**: Procesar mismo archivo con diferentes visualizaciones en secuencia
2. **Limpiar caché entre archivos**: `GRID2D_CACHE.clear()` entre archivos no relacionados
3. **Usar filtros visuales sobre filtros QC**: Los filtros visuales no requieren re-mallado
4. **Procesar productos relacionados juntos**: Mismo archivo, diferentes elevaciones comparten caché

---

## Glosario

| Término | Definición |
|---------|------------|
| **COG** | Cloud-Optimized GeoTIFF - Formato eficiente para mapeo web |
| **PPI** | Indicador de Posición en Plano - Barrido horizontal de radar |
| **CAPPI** | PPI de Altitud Constante - Corte a altura fija |
| **COLMAX** | Máximo de Columna - Valor máximo en columna vertical |
| **DBZH** | Reflectividad horizontal en dBZ |
| **RHOHV** | Coeficiente de correlación cruzada (indicador de calidad) |
| **QC** | Control de Calidad |
| **Enmascaramiento** | Ocultar datos basado en condiciones |
| **Mallado** | Proceso de interpolar datos polares a malla Cartesiana |
| **Gate/Puerta** | Muestra individual de radar en coordenadas polares |
