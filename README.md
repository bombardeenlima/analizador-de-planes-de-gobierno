# Analizador de Planes de Gobierno

Herramientas para extraer, verificar y visualizar propuestas a partir de planes de gobierno en formato PDF/TXT.

## Componentes clave
- `scripts/analizar_planes.py`: limpia los documentos fuente (`datos/propuestas/`), segmenta propuestas y genera `propuestas.csv`, bitácora (`log.txt`) y extractos (`extractos/`).
- `scripts/evaluar_propuestas.py`: cruza `propuestas.csv` contra los indicadores en `datos/informacion/` para emitir `evaluacion_propuestas.csv`.
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
Genera/actualiza `evaluacion_propuestas.csv` con el estado de cada propuesta (`cumplida`, `en_progreso`, `no_cumplida`, `sin_datos`) y un detalle justificando la clasificación.

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

## Registros y validación
- `log.txt` recoge advertencias/errores de extracción.
- `extractos/` guarda fragmentos de texto por distrito para auditoría.
- Se recomienda revisar manualmente muestras al azar para validar la calidad de las propuestas detectadas y los estados asignados.
