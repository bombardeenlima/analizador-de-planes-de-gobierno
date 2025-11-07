# Analizador de Planes de Gobierno

Herramientas para extraer, verificar y visualizar propuestas a partir de planes de gobierno en formato PDF/TXT.

## Componentes clave
- `scripts/analizar_planes.py`: limpia los documentos fuente (`datos/propuestas/`), segmenta propuestas y genera `propuestas.csv`, bitácora (`log.txt`) y extractos (`extractos/`).
- `scripts/evaluar_propuestas.py`: cruza `propuestas.csv` contra los indicadores ubicados en `datos/informacion/` (cualquier CSV) para emitir `evaluacion_propuestas.csv`. Los archivos se detectan automáticamente por su contenido, por lo que no es necesario respetar nombres específicos.
- `scripts/graficos_propuestas.py`: crea visualizaciones enriquecidas a partir de ambos CSV (barras, heatmaps, stacked bars, timelines y nube/top de palabras).
- `pipeline.py`: orquestador que permite ejecutar los pasos anteriores de forma secuencial u opciones puntuales mediante `--steps`.
- `datos/verbos.txt`: banco externo de verbos y patrones que controlan la detección de propuestas; se puede ampliar sin tocar el código.

## Requisitos
Python 3.10+ y las dependencias opcionales:

```bash
pip install pdfminer.six PyPDF2 pandas matplotlib seaborn wordcloud
```

- `pdfminer.six` y `PyPDF2` habilitan la extracción de texto desde PDF.
- `pandas` facilita el tratamiento de datos tabulares.
- `matplotlib`, `seaborn` y `wordcloud` son necesarias para los gráficos (sin `wordcloud` se genera un top de palabras).

## Estructura de carpetas
```
datos/
├─ propuestas/              # PDFs / TXTs originales
├─ informacion/             # Indicadores y fuentes externas
└─ verbos.txt               # Bancos de verbos y patrones
graficos/                   # Salida de visualizaciones
propuestas.csv              # Resultado de extracción
evaluacion_propuestas.csv   # Resultado de verificación
scripts/                    # Herramientas principales
```

## Ejecución manual por paso
### 1. Extraer propuestas
```bash
python scripts/analizar_planes.py \
  --input datos/propuestas \
  --output propuestas.csv
```
Opciones relevantes:
- `--input`: carpeta con PDFs/TXT (default `datos/propuestas`).
- `--output`: CSV resultante (default `propuestas.csv`).
- `--skip-charts`: evita crear gráficos automáticamente.

### 2. Verificar propuestas con data externa
```bash
python scripts/evaluar_propuestas.py
```
Genera/actualiza `evaluacion_propuestas.csv` con el estado de cada propuesta (`cumplida`, `en_progreso`, `no_cumplida`, `sin_datos`) y un detalle justificando la clasificación. El script inspecciona `datos/informacion/` y usa heurísticas sobre encabezados/contenido de cada CSV para mapearlos a indicadores (presupuestos, denuncias, infobras, etc.), de modo que basta con colocar datos relevantes en esa carpeta para incorporarlos.

### 3. Generar gráficos
```bash
python scripts/graficos_propuestas.py \
  --csv propuestas.csv \
  --evaluacion evaluacion_propuestas.csv \
  --out graficos
```
Produce visualizaciones comparables entre distritos y métricas (barras, heatmaps, stacked bars, timelines, nube/top palabras). Los avisos en consola indican dependencias faltantes o datos insuficientes.

## Personalización de detección
Edita `datos/verbos.txt` para añadir o ajustar:
- `[ACTION_VERBS]`: verbos que suelen encabezar propuestas.
- `[ACTION_NOMINALIZATIONS]`: patrones nominales de acción.
- `[HARD_SIGNS]`, `[MEDIUM_SIGNS]`, `[LOW_SIGNS]`: pistas utilizadas para estimar factibilidad.

Los cambios se aplican en la siguiente ejecución sin modificar el código fuente.

## Datos externos flexibles
- Cada CSV dentro de `datos/informacion/` se analiza automáticamente intentando varios formatos (codificación/separador) hasta poder cargarse.
- Los encabezados se normalizan (sin acentos, símbolos ni mayúsculas) para inferir a qué indicador pertenece el archivo (p. ej. columnas con `avance` + `municipalidad` → ejecución presupuestal; `dist_hecho` + `cantidad` → denuncias policiales; columnas con `estado` → infobras).
- Para Infobras se admite que un solo archivo contenga varios distritos; el contenido se fracciona por la columna de distrito/municipalidad disponible.
- Si quieres sumar nuevos tipos de indicadores basta con ampliar la función de clasificación en `scripts/evaluar_propuestas.py`, sin cambiar la forma en que se almacenan los datos.

## Registros y validación
- `log.txt` recoge advertencias/errores de extracción.
- `extractos/` guarda fragmentos de texto por distrito para auditoría.
- Se recomienda revisar manualmente muestras al azar para validar la calidad de las propuestas detectadas y los estados asignados.
