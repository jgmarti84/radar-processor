# Entendiendo los Filtros en Radar COG Processor

Este documento proporciona una explicación profunda del sistema de filtrado, explicando exactamente cómo funcionan los filtros en cada etapa del procesamiento.

## Tabla de Contenidos

1. [Conceptos Básicos de Filtros](#conceptos-básicos-de-filtros)
2. [Tipos de Filtros](#tipos-de-filtros)
3. [Cómo Funcionan los Filtros Internamente](#cómo-funcionan-los-filtros-internamente)
4. [Ejemplos Prácticos](#ejemplos-prácticos)
5. [Errores Comunes](#errores-comunes)
6. [Referencia de Filtros](#referencia-de-filtros)

---

## Conceptos Básicos de Filtros

### ¿Qué es un Filtro?

Un filtro es un objeto que especifica condiciones para incluir o excluir datos de radar. Los filtros ayudan a eliminar ruido, ecos no meteorológicos y a enfocarse en rangos de datos específicos.

### Estructura del Objeto Filtro

Un objeto filtro debe tener estos atributos:
- `field`: El campo de radar sobre el cual filtrar (ej. "RHOHV", "DBZH")
- `min`: Valor mínimo a incluir (opcional)
- `max`: Valor máximo a incluir (opcional)

```python
class Filter:
    def __init__(self, field: str, min: float = None, max: float = None):
        self.field = field
        self.min = min
        self.max = max

# Ejemplos
filtro_qc = Filter("RHOHV", min=0.85)        # Mantener RHOHV >= 0.85
filtro_rango = Filter("DBZH", min=10, max=60) # Mantener 10 <= DBZH <= 60
filtro_max = Filter("DBZH", max=50)           # Mantener DBZH <= 50
```

---

## Tipos de Filtros

La biblioteca usa un **sistema de filtrado de dos niveles**. Esto es crucial de entender:

### 1. Filtros QC (Control de Calidad)

**Propósito:** Eliminar datos de baja calidad ANTES del mallado/interpolación.

**Campos Afectados:** Actualmente solo `RHOHV`

**Cuándo se Aplican:** Durante la fase de construcción de la malla (función `map_gates_to_grid` de PyART)

**Efecto:** Las puertas (gates - muestras de radar) que no pasan el filtro se excluyen completamente de la interpolación.

**Afecta el Caché:** SÍ - Diferentes filtros QC producen diferentes mallas, por lo que cambian la clave de caché.

```
┌─────────────────────────────────────────────────────────────────────┐
│                      FLUJO DE FILTRO QC                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   Datos de Radar Crudos                                              │
│        │                                                             │
│        ▼                                                             │
│   ┌──────────────────────────────────────────────────────────────┐  │
│   │  Creación de Filtro de Puertas (utils.build_gatefilter)      │  │
│   │  • Para cada filtro QC en RHOHV:                              │  │
│   │    - Marcar puertas donde RHOHV < min como "excluidas"        │  │
│   │    - Marcar puertas donde RHOHV > max como "excluidas"        │  │
│   └──────────────────────────────────────────────────────────────┘  │
│        │                                                             │
│        ▼                                                             │
│   ┌──────────────────────────────────────────────────────────────┐  │
│   │  Construcción de Malla (pyart.map.grid_from_radars)           │  │
│   │  • Las puertas excluidas NO se interpolan                     │  │
│   │  • Solo las puertas válidas contribuyen a la malla            │  │
│   └──────────────────────────────────────────────────────────────┘  │
│        │                                                             │
│        ▼                                                             │
│   Datos Mallados (QC aplicado)                                       │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 2. Filtros Visuales

**Propósito:** Enmascarar valores en el producto FINAL visualizado.

**Campos Afectados:** Cualquier campo (típicamente el campo principal que se está mostrando)

**Cuándo se Aplican:** Después del mallado, al arreglo 2D final

**Efecto:** Los valores fuera del rango se enmascaran (transparente/sin-datos)

**Afecta el Caché:** NO - Se usa la misma malla, solo cambia el enmascaramiento

```
┌─────────────────────────────────────────────────────────────────────┐
│                      FLUJO DE FILTRO VISUAL                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   Datos Mallados (del caché o recién calculados)                     │
│        │                                                             │
│        ▼                                                             │
│   ┌──────────────────────────────────────────────────────────────┐  │
│   │  Aplicación de Filtro (processor._apply_filter_masks)         │  │
│   │  • Para cada filtro visual:                                   │  │
│   │    - Crear máscara donde valor < min                          │  │
│   │    - Crear máscara donde valor > max                          │  │
│   │    - Aplicar máscaras al arreglo de datos                     │  │
│   └──────────────────────────────────────────────────────────────┘  │
│        │                                                             │
│        ▼                                                             │
│   Datos Enmascarados (listos para aplicación de mapa de colores)     │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Lógica de Clasificación de Filtros

Cuando pasas filtros a `process_radar_to_cog()`, la biblioteca los clasifica automáticamente:

```python
# Dentro de processor._build_processing_config():

AFFECTS_INTERP_FIELDS = {"RHOHV"}  # Campos QC

for filter in filters:
    if filter.field in AFFECTS_INTERP_FIELDS:
        qc_filters.append(filter)      # Va a construcción de malla
    else:
        visual_filters.append(filter)  # Se aplica después del mallado
```

---

## Cómo Funcionan los Filtros Internamente

### Pipeline de Procesamiento Completo con Filtros

```
ENTRADA: Archivo de radar, filtros, configuración de producto
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ FASE 1-2: Configuración y Validación                                 │
│ • Analizar parámetros de entrada                                     │
│ • Validar objetos de filtro                                          │
└─────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ FASE 7: Clasificación de Filtros                                     │
│ • Separar filtros en QC (RHOHV) vs Visual (otros)                   │
│ • Filtros QC → afectan interpolación                                 │
│ • Filtros visuales → máscaras de post-procesamiento                  │
└─────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ FASE 8: Generación de Clave de Caché                                 │
│ • Construir clave de caché de: archivo + producto + elevación +     │
│   FILTROS_QC                                                         │
│ • Nota: Filtros visuales NO están en la clave de caché              │
│ • Verificar si la malla existe en caché                              │
└─────────────────────────────────────────────────────────────────────┘
                    │
        ┌──────────┴──────────┐
        │                     │
   ACIERTO CACHÉ         FALLO CACHÉ
        │                     │
        │                     ▼
        │     ┌─────────────────────────────────────────────────────┐
        │     │ FASE 9: Construcción de Malla                       │
        │     │ • Crear GateFilter desde filtros QC                 │
        │     │ • Ejecutar pyart.map.grid_from_radars()             │
        │     │ • Puertas excluidas no contribuyen a la malla       │
        │     │ • Almacenar resultado en caché                      │
        │     └─────────────────────────────────────────────────────┘
        │                     │
        └──────────┬──────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│ FASE 10-11: Aplicar Filtros Visuales                                 │
│ • Obtener datos de malla para el campo solicitado                    │
│ • Para cada filtro visual:                                           │
│   - Enmascarar valores fuera del rango [min, max]                   │
│ • Aplicar máscara de campo QC (RHOHV 2D colapsado si presente)      │
└─────────────────────────────────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│ FASE 12: Exportación                                                 │
│ • Aplicar mapa de colores                                            │
│ • Escribir GeoTIFF optimizado para la nube                          │
└─────────────────────────────────────────────────────────────────────┘
                   │
                   ▼
             SALIDA: Archivo COG
```

### Detalles de Implementación del Filtro QC

El filtro QC crea un objeto `GateFilter` de PyART:

```python
# En utils.build_gatefilter():

def build_gatefilter(radar, filters):
    """Construir GateFilter de PyART desde especificaciones de filtro."""
    gatefilter = pyart.filters.GateFilter(radar)
    
    for filt in filters:
        # Excluir puertas donde RHOHV está debajo del mínimo
        if filt.min is not None:
            gatefilter.exclude_below(filt.field, filt.min)
        
        # Excluir puertas donde RHOHV está arriba del máximo
        if filt.max is not None:
            gatefilter.exclude_above(filt.field, filt.max)
    
    return gatefilter
```

Este GateFilter se pasa al constructor de mallas de PyART:

```python
grid = pyart.map.grid_from_radars(
    radars,
    gatefilters=[gatefilter],  # Puertas excluidas no se interpolarán
    ...
)
```

### Detalles de Implementación del Filtro Visual

Los filtros visuales enmascaran el arreglo de datos final:

```python
# En processor._apply_filter_masks():

def _apply_filter_masks(data, filters, grid, config):
    """Aplicar máscaras de filtro visual a datos mallados."""
    
    for filt in filters:
        # Obtener los datos del campo de filtro (ej. valores DBZH)
        filter_data = grid.fields[filt.field]['data']
        
        # Crear máscaras para valores fuera de rango
        if filt.min is not None:
            mask = filter_data < filt.min
            data = np.ma.masked_where(mask, data)
        
        if filt.max is not None:
            mask = filter_data > filt.max
            data = np.ma.masked_where(mask, data)
    
    return data
```

---

## Ejemplos Prácticos

### Ejemplo 1: Eliminar Clutter de Tierra con RHOHV

El clutter de tierra (edificios, montañas) típicamente tiene valores bajos de RHOHV:

```python
filters = [Filter("RHOHV", min=0.85)]

result = process_radar_to_cog(
    filepath="radar.nc",
    product="PPI",
    field_requested="DBZH",
    elevation=0,
    filters=filters,
    output_dir="output"
)
```

**Qué sucede:**
1. Filtro clasificado como QC (RHOHV está en AFFECTS_INTERP_FIELDS)
2. Puertas donde RHOHV < 0.85 excluidas de la construcción de malla
3. Esas áreas aparecen como transparentes en la salida

### Ejemplo 2: Mostrar Solo Precipitación Intensa

```python
filters = [
    Filter("RHOHV", min=0.85),    # QC: Eliminar clutter
    Filter("DBZH", min=40)         # Visual: Solo precipitación intensa
]

result = process_radar_to_cog(
    filepath="radar.nc",
    product="PPI", 
    field_requested="DBZH",
    elevation=0,
    filters=filters,
    output_dir="output"
)
```

**Qué sucede:**
1. Filtro RHOHV → QC (afecta construcción de malla)
2. Filtro DBZH → Visual (aplicado después del mallado)
3. Resultado: Datos limpios mostrando solo DBZH ≥ 40 dBZ

### Ejemplo 3: Aislar Rango de Precipitación

```python
filters = [
    Filter("RHOHV", min=0.85),     # Eliminar no-meteorológico
    Filter("DBZH", min=20, max=45) # Solo precipitación moderada
]
```

**Qué sucede:**
- DBZH < 20 dBZ: enmascarado (muy ligero)
- DBZH > 45 dBZ: enmascarado (muy intenso)
- Solo se muestran valores de 20-45 dBZ

### Ejemplo 4: Procesar Múltiples Productos con el Mismo QC

Debido a que los filtros QC afectan el caché, puedes procesar eficientemente múltiples productos:

```python
from radar_cog_processor.cache import GRID3D_CACHE

# Definir filtro QC una vez
qc = [Filter("RHOHV", min=0.85)]

# Procesar CAPPI (crea y cachea malla 3D)
cappi_result = process_radar_to_cog(
    filepath="radar.nc",
    product="CAPPI",
    field_requested="DBZH",
    cappi_height=3000,
    filters=qc,
    output_dir="output/cappi"
)

# Procesar COLMAX (reutiliza malla 3D cacheada - ¡mismo filtro QC!)
colmax_result = process_radar_to_cog(
    filepath="radar.nc", 
    product="COLMAX",
    field_requested="DBZH",
    filters=qc,  # Mismo QC = misma clave de caché
    output_dir="output/colmax"
)
```

### Ejemplo 5: Diferentes Filtros Visuales, Misma Malla

Los filtros visuales no afectan el caché, así que puedes aplicar diferentes filtros visuales a la misma malla cacheada:

```python
qc = [Filter("RHOHV", min=0.85)]

# Precipitación ligera
light = process_radar_to_cog(
    filepath="radar.nc",
    product="PPI",
    field_requested="DBZH",
    filters=qc + [Filter("DBZH", min=10, max=30)],
    output_dir="output/ligera"
)

# Precipitación intensa (usa misma malla cacheada)
heavy = process_radar_to_cog(
    filepath="radar.nc",
    product="PPI",
    field_requested="DBZH",
    filters=qc + [Filter("DBZH", min=40)],  # Diferente filtro visual
    output_dir="output/intensa"
)
```

---

## Errores Comunes

### Error 1: Esperar que Campos No-RHOHV Afecten el Mallado

❌ **Expectativa incorrecta:**
```python
# ZDR NO es un campo QC, así que esto no excluye puertas durante el mallado
filters = [Filter("ZDR", min=-1, max=4)]
```

✅ **Lo que realmente sucede:**
- El filtro ZDR se clasifica como Visual
- Todas las puertas se interpolan
- El rango ZDR se aplica como máscara DESPUÉS del mallado

**Si necesitas filtrado de puertas basado en ZDR**, necesitarías modificar `AFFECTS_INTERP_FIELDS` en `constants.py`.

### Error 2: Asumir que los Filtros Visuales Ahorran Tiempo de Procesamiento

❌ **Suposición incorrecta:**
```python
# Ambos hacen la misma cantidad de cómputo de malla
filters_estrecho = [Filter("DBZH", min=40, max=50)]  # Rango estrecho
filters_amplio = [Filter("DBZH", min=0, max=70)]     # Rango amplio
```

✅ **Realidad:**
- Los filtros visuales no reducen el cómputo de malla
- Solo enmascaran la salida
- Para procesamiento más rápido, usa filtros QC (RHOHV)

### Error 3: Olvidar Limpiar el Caché Entre Diferentes Archivos

❌ **Problema potencial:**
```python
# Procesar archivo A
process_radar_to_cog("archivoA.nc", filters=filters, ...)

# Procesar archivo B - pero el caché del archivo A podría persistir
process_radar_to_cog("archivoB.nc", filters=filters, ...)
```

✅ **Mejor práctica:**
```python
from radar_cog_processor.cache import GRID2D_CACHE, GRID3D_CACHE

# Limpiar entre diferentes archivos de radar
GRID2D_CACHE.clear()
GRID3D_CACHE.clear()

process_radar_to_cog("archivoB.nc", ...)
```

### Error 4: Usar Nombres de Atributos Incorrectos en el Filtro

❌ **Incorrecto:**
```python
class FiltroMalo:
    def __init__(self, field, minimo=None, maximo=None):
        self.field = field
        self.minimo = minimo  # ¡Incorrecto! Debería ser 'min'
        self.maximo = maximo  # ¡Incorrecto! Debería ser 'max'
```

✅ **Correcto:**
```python
class Filter:
    def __init__(self, field, min=None, max=None):
        self.field = field
        self.min = min   # Nombre de atributo correcto
        self.max = max   # Nombre de atributo correcto
```

---

## Referencia de Filtros

### Campos QC (Afectan Interpolación)

| Campo | Descripción | Valores Típicos de Filtro |
|-------|-------------|--------------------------|
| RHOHV | Coeficiente de correlación cruzada | min=0.80 (permisivo), min=0.85 (estándar), min=0.92 (estricto) |

### Campos Visuales (Máscaras Post-Mallado)

| Campo | Descripción | Filtros de Ejemplo |
|-------|-------------|-------------------|
| DBZH | Reflectividad | min=10 (precip significativa), min=40 (intensa), max=60 |
| ZDR | Reflectividad diferencial | min=-1, max=5 |
| VRAD | Velocidad radial | min=-30, max=30 |
| KDP | Fase diferencial específica | min=0, max=10 |
| WRAD | Ancho espectral | max=8 |

### Guía de Filtros RHOHV

| Umbral RHOHV | Caso de Uso | Qué se Filtra |
|--------------|-------------|---------------|
| ≥ 0.70 | Muy permisivo | Solo no-met obvio |
| ≥ 0.80 | Permisivo | Mayoría de clutter de tierra, insectos |
| ≥ 0.85 | Estándar (recomendado) | Clutter, pájaros, fase mixta |
| ≥ 0.90 | Estricto | Todo excepto lluvia/nieve pura |
| ≥ 0.95 | Muy estricto | Solo precipitación uniforme |

### Guía de Filtros DBZH

| Rango DBZH | Tipo de Precipitación |
|------------|----------------------|
| 0-15 dBZ | Muy ligera, neblina |
| 15-30 dBZ | Lluvia/nieve ligera |
| 30-40 dBZ | Lluvia moderada |
| 40-50 dBZ | Lluvia intensa |
| 50-60 dBZ | Lluvia muy intensa, posible granizo |
| >60 dBZ | Granizo grande, tormentas severas |

---

## Resumen

1. **Dos Tipos de Filtros:**
   - Filtros QC (RHOHV): Afectan construcción de malla, cambian clave de caché
   - Filtros visuales (otros): Máscaras de post-procesamiento, no afectan caché

2. **Requisitos del Objeto Filtro:**
   - Debe tener atributos `.field`, `.min`, `.max`
   - `min` y `max` son opcionales (pueden ser None)

3. **Implicaciones de Rendimiento:**
   - Mismos filtros QC = pueden compartir mallas cacheadas
   - Filtros visuales no afectan tiempo de cómputo

4. **Mejores Prácticas:**
   - Siempre usar filtro QC RHOHV para datos de calidad
   - Limpiar cachés entre diferentes archivos de radar
   - Usar filtros visuales para personalización de visualización
