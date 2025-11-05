#!/usr/bin/env python3
"""
Genera un set de gráficos enriquecidos y estéticos a partir de:
  - propuestas.csv
  - evaluacion_propuestas.csv (opcional)

Ejemplo:
    python3 graficos_propuestas.py --csv propuestas.csv \
        --evaluacion evaluacion_propuestas.csv --out graficos
"""
from __future__ import annotations

import argparse
import math
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_PROPOSALS_CSV = ROOT_DIR / "propuestas.csv"
DEFAULT_EVALUACION_CSV = ROOT_DIR / "evaluacion_propuestas.csv"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "graficos"

PALETTE = ["#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51", "#8ecae6", "#ffb703"]
BACKGROUND = "#f8f9fa"


class ChartDependencyError(RuntimeError):
    """Se dispara si falta alguna dependencia gráfica."""


def ensure_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - dependencias externas
        raise ChartDependencyError("Se requiere 'matplotlib' para generar gráficos.") from exc

    plt.rcParams.update(
        {
            "figure.facecolor": BACKGROUND,
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.2,
            "grid.linestyle": "--",
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
        }
    )
    return plt


def ensure_seaborn(plt):
    try:
        import seaborn as sns  # type: ignore
    except Exception:
        return None
    sns.set_theme(style="whitegrid", context="talk", palette=PALETTE)
    return sns


def ensure_wordcloud():
    try:
        from wordcloud import WordCloud  # type: ignore
    except Exception:
        return None
    return WordCloud


def load_dataframe(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    return df.fillna("")


def normalize_timeline(text: str) -> str:
    if not text or str(text).strip() in {"", "NaN", "nan"}:
        return "No definido"
    text = str(text).strip()
    if re.search(r"20\d{2}\s*–\s*20\d{2}", text):
        return "Rango multianual"
    if re.match(r"^\d{4}$", text):
        return f"Año {text}"
    if re.search(r"(primer|segundo|tercer).+año", text.lower()):
        return "Primeros años"
    if re.search(r"(mes|semestre)", text.lower()):
        return "Periodicidad breve"
    if re.search(r"20\d{2}", text):
        return "Referencia temporal"
    return "Otro"


def tokenize_words(texts: Iterable[str]) -> List[str]:
    tokens: List[str] = []
    word_pat = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]{4,}")
    stopwords = {
        "de",
        "la",
        "el",
        "los",
        "las",
        "y",
        "en",
        "del",
        "para",
        "con",
        "por",
        "una",
        "un",
        "se",
        "al",
        "a",
        "que",
        "su",
        "sus",
        "desde",
        "sobre",
        "como",
        "mas",
        "más",
        "menos",
        "sera",
        "será",
        "seran",
        "serán",
        "ser",
        "es",
        "son",
        "municipal",
        "gestion",
        "gestión",
        "gestionar",
        "plan",
        "gobierno",
        "distrito",
    }
    for text in texts:
        for word in word_pat.findall(str(text).lower()):
            if word not in stopwords:
                tokens.append(word)
    return tokens


def annotate_bar(ax, orient: str = "v") -> None:
    for patch in ax.patches:
        if orient == "v":
            height = patch.get_height()
            if math.isnan(height):
                continue
            ax.text(
                patch.get_x() + patch.get_width() / 2,
                height + ax.get_ylim()[1] * 0.01,
                f"{int(height)}",
                ha="center",
                va="bottom",
                fontsize=10,
            )
        else:
            width = patch.get_width()
            if math.isnan(width):
                continue
            ax.text(
                width + ax.get_xlim()[1] * 0.01,
                patch.get_y() + patch.get_height() / 2,
                f"{int(width)}",
                va="center",
                fontsize=10,
            )


def plot_top_districts(df: pd.DataFrame, out_dir: Path, sns, plt):
    counts = df["distrito"].str.strip().value_counts().head(10)
    if counts.empty:
        raise ValueError("No hay datos de distrito para graficar.")
    fig, ax = plt.subplots(figsize=(9, 6))
    if sns:
        sns.barplot(x=counts.values, y=counts.index, palette=PALETTE, ax=ax)
    else:
        ax.barh(counts.index, counts.values, color=PALETTE[1])
    ax.set_xlabel("Número de propuestas")
    ax.set_ylabel("")
    ax.set_title("Top 10 distritos con más propuestas")
    annotate_bar(ax, orient="h")
    fig.tight_layout()
    out_path = out_dir / "propuestas_top_distritos.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_factibilidad_heatmap(df: pd.DataFrame, out_dir: Path, sns, plt):
    top_districts = df["distrito"].str.strip().value_counts().head(12).index
    subset = (
        df.assign(
            distrito=df["distrito"].str.strip(),
            factibilidad=df["factibilidad"].replace("", "No definida"),
        )
        .query("distrito != ''")
        .loc[lambda _df: _df["distrito"].isin(top_districts)]
    )
    if subset.empty:
        raise ValueError("No hay datos suficientes para el mapa de calor.")
    pivot = (
        subset.groupby(["distrito", "factibilidad"])
        .size()
        .unstack(fill_value=0)
        .loc[top_districts]
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    if sns:
        sns.heatmap(pivot, annot=True, fmt="d", cmap="YlGnBu", linewidths=0.5, ax=ax)
    else:
        im = ax.imshow(pivot.values, cmap="YlGnBu")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                ax.text(j, i, int(pivot.values[i, j]), ha="center", va="center", color="black")
    ax.set_title("Factibilidad declarada por distrito (Top 12)")
    fig.tight_layout()
    out_path = out_dir / "factibilidad_por_distrito.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_timeline_distribution(df: pd.DataFrame, out_dir: Path, sns, plt):
    timeline = df["timeline"].apply(normalize_timeline).value_counts().head(12)
    fig, ax = plt.subplots(figsize=(9, 5))
    if sns:
        sns.barplot(x=timeline.index, y=timeline.values, palette=PALETTE, ax=ax)
    else:
        ax.bar(timeline.index, timeline.values, color=PALETTE[2])
    ax.set_ylabel("Número de propuestas")
    ax.set_xlabel("Categoría temporal declarada")
    ax.set_title("Distribución de horizontes temporales")
    ax.tick_params(axis="x", rotation=30, ha="right")
    annotate_bar(ax, orient="v")
    fig.tight_layout()
    out_path = out_dir / "timeline_detallado.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_word_visual(df: pd.DataFrame, out_dir: Path, plt):
    tokens = tokenize_words(df["propuesta"])
    if not tokens:
        raise ValueError("No se encontraron palabras suficientes para la nube de palabras.")
    wordcloud_cls = ensure_wordcloud()
    fig, ax = plt.subplots(figsize=(11, 7))
    if wordcloud_cls:
        text = " ".join(tokens)
        wc = wordcloud_cls(width=1600, height=950, background_color="white", colormap="viridis").generate(text)
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title("Nube de palabras de propuestas")
        out_path = out_dir / "nube_palabras.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        return out_path, None

    counts = Counter(tokens).most_common(20)
    labels, values = zip(*counts)
    ax.barh(labels[::-1], values[::-1], color=PALETTE[0])
    ax.set_title("Top 20 palabras (WordCloud no disponible)")
    ax.set_xlabel("Frecuencia")
    fig.tight_layout()
    out_path = out_dir / "top_palabras.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path, "WordCloud no disponible, se generó un top de palabras."


def plot_status_distribution(eval_df: pd.DataFrame, out_dir: Path, sns, plt):
    counts = eval_df["status"].value_counts()
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = PALETTE[: len(counts)]
    if sns:
        sns.barplot(x=counts.index, y=counts.values, palette=colors, ax=ax)
    else:
        ax.bar(counts.index, counts.values, color=colors)
    ax.set_ylabel("Número de propuestas")
    ax.set_xlabel("Estado evaluado")
    ax.set_title("Distribución de estados de evaluación")
    annotate_bar(ax, orient="v")
    fig.tight_layout()
    out_path = out_dir / "evaluacion_estados.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_status_by_district(eval_df: pd.DataFrame, out_dir: Path, sns, plt):
    top_districts = eval_df["distrito"].str.strip().value_counts().head(12).index
    pivot = (
        eval_df.assign(distrito=eval_df["distrito"].str.strip())
        .loc[lambda df_: df_["distrito"].isin(top_districts)]
        .groupby(["distrito", "status"])
        .size()
        .unstack(fill_value=0)
        .loc[top_districts]
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    bottom = pd.Series([0] * len(pivot), index=pivot.index)
    colors = {status: PALETTE[idx % len(PALETTE)] for idx, status in enumerate(pivot.columns)}
    for status in pivot.columns:
        ax.bar(
            pivot.index,
            pivot[status],
            bottom=bottom,
            label=status,
            color=colors[status],
        )
        bottom = bottom + pivot[status]
    ax.set_ylabel("Número de propuestas")
    ax.set_xlabel("")
    ax.set_title("Estados por distrito (Top 12)")
    ax.legend(title="Estado", loc="upper right")
    ax.tick_params(axis="x", rotation=30, ha="right")
    fig.tight_layout()
    out_path = out_dir / "evaluacion_estados_por_distrito.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_status_vs_factibilidad(eval_df: pd.DataFrame, out_dir: Path, sns, plt):
    table = (
        eval_df.assign(factibilidad=eval_df["factibilidad"].replace("", "No definida"))
        .groupby(["status", "factibilidad"])
        .size()
        .unstack(fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    if sns:
        sns.heatmap(table, annot=True, fmt="d", cmap="rocket_r", linewidths=0.5, ax=ax)
    else:
        im = ax.imshow(table.values, cmap="rocket_r")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(range(len(table.columns)))
        ax.set_xticklabels(table.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(table.index)))
        ax.set_yticklabels(table.index)
        for i in range(table.shape[0]):
            for j in range(table.shape[1]):
                ax.text(j, i, int(table.values[i, j]), ha="center", va="center", color="white")
    ax.set_title("Cruce entre estado evaluado y factibilidad declarada")
    fig.tight_layout()
    out_path = out_dir / "evaluacion_estado_vs_factibilidad.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_status_vs_timeline(eval_df: pd.DataFrame, out_dir: Path, sns, plt):
    table = (
        eval_df.assign(timeline_categoria=eval_df["timeline"].apply(normalize_timeline))
        .groupby(["status", "timeline_categoria"])
        .size()
        .unstack(fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(11, 6))
    if sns:
        sns.heatmap(table, annot=True, fmt="d", cmap="crest", linewidths=0.5, ax=ax)
    else:
        im = ax.imshow(table.values, cmap="crest")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(range(len(table.columns)))
        ax.set_xticklabels(table.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(table.index)))
        ax.set_yticklabels(table.index)
        for i in range(table.shape[0]):
            for j in range(table.shape[1]):
                ax.text(j, i, int(table.values[i, j]), ha="center", va="center")
    ax.set_title("Estados evaluados por categoría temporal declarada")
    fig.tight_layout()
    out_path = out_dir / "evaluacion_estado_vs_timeline.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def generate_all_charts(
    csv_path: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    eval_csv: Optional[Path] = None,
) -> Tuple[List[Path], List[str]]:
    plt = ensure_matplotlib()
    sns = ensure_seaborn(plt)

    df = load_dataframe(csv_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: List[Path] = []
    warnings: List[str] = []

    proposal_tasks = [
        ("Top distritos", plot_top_districts),
        ("Factibilidad por distrito", plot_factibilidad_heatmap),
        ("Distribución de timeline", plot_timeline_distribution),
    ]

    for label, func in proposal_tasks:
        try:
            path = func(df, output_dir, sns, plt)
        except ValueError as exc:
            warnings.append(f"{label}: {exc}")
        else:
            generated.append(path)

    # Word visual handled separately because it may require WordCloud
    try:
        word_path, note = plot_word_visual(df, output_dir, plt)
        generated.append(word_path)
        if note:
            warnings.append(note)
    except ValueError as exc:
        warnings.append(f"Nube de palabras: {exc}")

    if eval_csv and eval_csv.exists():
        eval_df = load_dataframe(eval_csv)
        eval_tasks = [
            ("Distribución de estados", plot_status_distribution),
            ("Estados por distrito", plot_status_by_district),
            ("Estado vs factibilidad", plot_status_vs_factibilidad),
            ("Estado vs timeline", plot_status_vs_timeline),
        ]
        for label, func in eval_tasks:
            try:
                path = func(eval_df, output_dir, sns, plt)
            except ValueError as exc:
                warnings.append(f"{label}: {exc}")
            else:
                generated.append(path)
    elif eval_csv:
        warnings.append(f"No se encontró el archivo de evaluación: {eval_csv}")

    return generated, warnings


def run_cli():
    parser = argparse.ArgumentParser(description="Genera gráficos enriquecidos para las propuestas y su evaluación.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_PROPOSALS_CSV, help="Ruta al CSV de propuestas.")
    parser.add_argument(
        "--evaluacion",
        type=Path,
        default=DEFAULT_EVALUACION_CSV,
        help="Ruta al CSV de evaluación (opcional).",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_DIR, help="Carpeta de salida.")
    args = parser.parse_args()

    try:
        generated, warnings = generate_all_charts(args.csv, args.out, args.evaluacion)
    except ChartDependencyError as exc:
        print(f"[GRAFICOS] Error crítico: {exc}")
        return
    except FileNotFoundError as exc:
        print(f"[GRAFICOS] {exc}")
        return

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
