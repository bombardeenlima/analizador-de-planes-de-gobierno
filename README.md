# Analizador de Planes de Gobierno

Herramientas para extraer y visualizar propuestas a partir de planes de gobierno en formato PDF/TXT.

## Componentes clave
- `analizar_planes.py`: limpia los documentos, segmenta propuestas y genera `propuestas.csv`, bitácora (`log.txt`) y extractos por distrito (`extractos/`).
- `graficos_propuestas.py`: crea visualizaciones a partir del CSV (barras, pastel, timeline y nube de palabras/Top tokens).
- `verbos.txt`: banco externo de verbos y patrones que controlan la detección de propuestas; se puede ampliar sin tocar el código.
- `propuestas/`: carpeta de entrada por defecto para los planes de gobierno.

## Requisitos
Python 3.10+ y las dependencias opcionales:

```bash
pip install pdfminer.six PyPDF2 matplotlib wordcloud pandas
```

- `pdfminer.six` y `PyPDF2` habilitan la extracción de texto desde PDF.
- `matplotlib` y `wordcloud` son necesarias para los gráficos (sin `wordcloud` se genera un gráfico de barras de palabras).
- `pandas` acelera la lectura del CSV (si falta se usa `csv.DictReader`).

## Uso básico
```bash
python3 analizar_planes.py \
  --input propuestas \
  --output propuestas.csv
```

### Opciones relevantes
- `--input`: carpeta con PDFs/TXT (default `./propuestas`).
- `--output`: nombre del CSV resultante (default `propuestas.csv`).
- `--skip-charts`: evita la generación automática de gráficos.

El script muestra el progreso por archivo y resume el total de propuestas, factibilidad y timeline. Al finalizar crea la carpeta `graficos/` con las figuras cuando no se usa `--skip-charts`.

## Generación de gráficos independiente
```bash
python3 graficos_propuestas.py --csv propuestas.csv --out graficos
```
Produce:
- `propuestas_por_distrito.png`
- `factibilidad.png`
- `timeline.png`
- `nube_palabras.png` (o `top_palabras.png` si falta `wordcloud`)

Los avisos en consola indican dependencias faltantes o datos insuficientes.

## Personalización de detección
Edita `verbos.txt` para añadir o ajustar:
- `[ACTION_VERBS]`: verbos que suelen encabezar propuestas.
- `[ACTION_NOMINALIZATIONS]`: patrones nominales de acción.
- `[HARD_SIGNS]`, `[MEDIUM_SIGNS]`, `[LOW_SIGNS]`: pistas utilizadas para estimar factibilidad.

Los cambios se aplican en la siguiente ejecución sin modificar el código fuente.

## Registros y validación
- `log.txt` recoge advertencias/errores de extracción.
- Se recomienda revisar manualmente muestras al azar para validar la calidad de las propuestas detectadas.
