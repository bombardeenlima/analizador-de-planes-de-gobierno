#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generación de gráficos a partir de propuestas.csv.

Se puede ejecutar como script independiente:
    python3 graficos_propuestas.py --csv propuestas.csv --out graficos

También expone generate_all_charts para ser reutilizado desde otros módulos.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Tuple, Dict


STOPWORDS = {
    "de", "la", "el", "los", "las", "y", "en", "del", "para", "con", "por",
    "una", "un", "se", "al", "a", "que", "su", "sus", "desde", "sobre", "o",
    "como", "más", "menos", "sera", "será", "seran", "serán",
    "ser", "es", "son", "municipal", "gestion", "gestión", "gestionar",
    "plan", "gobierno"
}


class ChartDependencyError(RuntimeError):
    """Fallo por dependencia faltante al generar gráficos."""


def ensure_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")  # backend sin interfaz
        import matplotlib.pyplot as plt  # noqa
        return plt
    except Exception as exc:  # pragma: no cover - dependencias externas
        raise ChartDependencyError("Se requiere 'matplotlib' para generar gráficos.") from exc


def load_records(csv_path: Path) -> List[Dict[str, str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo CSV: {csv_path}")
    try:
        import pandas as pd  # type: ignore
    except Exception:
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            return [dict(row) for row in reader]
    else:  # pragma: no cover - depende de pandas
        df = pd.read_csv(csv_path)
        return df.fillna("").to_dict(orient="records")


def tokenize_words(records: Iterable[Dict[str, str]]) -> List[str]:
    tokens: List[str] = []
    word_pat = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]{3,}")
    for row in records:
        text = str(row.get("propuesta", ""))
        for word in word_pat.findall(text.lower()):
            if word not in STOPWORDS and len(word) > 2:
                tokens.append(word)
    return tokens


def plot_proposals_by_district(records: List[Dict[str, str]], out_dir: Path):
    plt = ensure_matplotlib()
    counts = Counter((row.get("distrito") or "").strip() for row in records)
    counts.pop("", None)
    if not counts:
        raise ValueError("No hay datos de distrito para graficar.")
    labels, values = zip(*sorted(counts.items(), key=lambda kv: kv[1], reverse=True))
    positions = list(range(len(labels)))
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.5), 6))
    ax.bar(positions, values, color="#2a9d8f")
    ax.set_ylabel("Número de propuestas")
    ax.set_title("Propuestas por distrito")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    fig.tight_layout()
    out_path = out_dir / "propuestas_por_distrito.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_factibilidad(records: List[Dict[str, str]], out_dir: Path):
    plt = ensure_matplotlib()
    counts = Counter((row.get("factibilidad") or "No definida").strip() for row in records)
    labels, values = zip(*sorted(counts.items(), key=lambda kv: kv[0]))
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=140, colors=["#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51"])
    ax.set_title("Distribución de factibilidad")
    out_path = out_dir / "factibilidad.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_timeline(records: List[Dict[str, str]], out_dir: Path):
    plt = ensure_matplotlib()
    def bucket(value: str) -> str:
        if not value or str(value).strip() in {"", "nan", "NaN"}:
            return "No definido"
        text = str(value)
        if re.search(r"20\d{2}–20\d{2}", text):
            return "Rango"
        if re.search(r"\b20\d{2}\b", text):
            return "Año"
        return "Otro"

    counts = Counter(bucket(row.get("timeline", "")) for row in records)
    labels, values = zip(*sorted(counts.items(), key=lambda kv: kv[0]))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, values, color="#457b9d")
    ax.set_ylabel("Número de propuestas")
    ax.set_title("Clasificación de timeline")
    for idx, val in enumerate(values):
        ax.text(idx, val + max(values) * 0.02, str(val), ha="center")
    fig.tight_layout()
    out_path = out_dir / "timeline.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_word_visual(tokens: List[str], out_dir: Path):
    if not tokens:
        raise ValueError("No hay tokens disponibles para la nube de palabras.")
    try:
        from wordcloud import WordCloud  # type: ignore
    except Exception:
        plt = ensure_matplotlib()
        top = Counter(tokens).most_common(20)
        labels, values = zip(*top)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(labels[::-1], values[::-1], color="#8ab17d")
        ax.set_title("Top 20 palabras (sin wordcloud)")
        ax.set_xlabel("Frecuencia")
        fig.tight_layout()
        out_path = out_dir / "top_palabras.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return out_path, "WordCloud no disponible, se generó histograma de palabras."

    plt = ensure_matplotlib()
    text = " ".join(tokens)
    wc = WordCloud(width=1400, height=900, background_color="white", colormap="viridis").generate(text)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("Nube de palabras de propuestas")
    fig.tight_layout()
    out_path = out_dir / "nube_palabras.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path, None


def generate_all_charts(csv_path: Path, output_dir: Path = Path("graficos")) -> Tuple[List[Path], List[str]]:
    records = load_records(csv_path)
    if not records:
        return [], ["El CSV no contiene registros para graficar."]

    output_dir.mkdir(parents=True, exist_ok=True)
    generated: List[Path] = []
    warnings: List[str] = []

    chart_tasks = [
        ("Propuestas por distrito", plot_proposals_by_district),
        ("Factibilidad", plot_factibilidad),
        ("Timeline", plot_timeline),
    ]

    for label, func in chart_tasks:
        try:
            path = func(records, output_dir)
        except ChartDependencyError as dep_err:
            warnings.append(str(dep_err))
            break  # sin matplotlib no tiene sentido continuar
        except ValueError as exc:
            warnings.append(f"{label}: {exc}")
        else:
            generated.append(path)

    if not warnings or all(not isinstance(w, str) or "matplotlib" not in w.lower() for w in warnings):
        try:
            tokens = tokenize_words(records)
            word_path, note = plot_word_visual(tokens, output_dir)
            generated.append(word_path)
            if note:
                warnings.append(note)
        except ChartDependencyError as dep_err:
            warnings.append(str(dep_err))
        except ValueError as exc:
            warnings.append(f"Nube de palabras: {exc}")

    return generated, warnings


def run_cli():
    parser = argparse.ArgumentParser(description="Genera gráficos a partir de propuestas.csv.")
    parser.add_argument("--csv", type=Path, default=Path("propuestas.csv"),
                        help="Ruta al CSV de propuestas (default: propuestas.csv)")
    parser.add_argument("--out", type=Path, default=Path("graficos"),
                        help="Carpeta donde guardar los gráficos (default: ./graficos)")
    args = parser.parse_args()

    generated, warnings = generate_all_charts(args.csv, args.out)
    if generated:
        print("[GRAFICOS] Archivos generados:")
        for path in generated:
            print(f"  - {path}")
    else:
        print("[GRAFICOS] No se generaron gráficos.")
    if warnings:
        print("[GRAFICOS] Advertencias:")
        for msg in warnings:
            print(f"  - {msg}")


if __name__ == "__main__":
    run_cli()
