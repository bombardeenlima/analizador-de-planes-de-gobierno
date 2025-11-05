#!/usr/bin/env python3
"""
Orquestador para el flujo completo:

1. Extraer propuestas desde los planes (propuestas.csv).
2. Verificar las propuestas con la data disponible (evaluacion_propuestas.csv).
3. Generar visualizaciones basadas en ambos CSV.

Por defecto ejecuta los tres pasos de manera secuencial; opcionalmente se
pueden especificar etapas concretas usando `--steps`.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Callable, List, Tuple


ROOT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = ROOT_DIR / "scripts"


def run_script(script_name: str, *args: str) -> int:
    script_path = SCRIPTS_DIR / script_name
    if not script_path.exists():
        print(f"[ERROR] No se encontró el script {script_path}")
        return 1
    cmd = [sys.executable, str(script_path), *args]
    print(f"\n[PIPELINE] Ejecutando: {' '.join(cmd)}")
    try:
        proc = subprocess.run(cmd, cwd=ROOT_DIR, check=False)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[PIPELINE] Falló la ejecución de {script_name}: {exc}")
        return 1
    if proc.returncode == 0:
        print(f"[PIPELINE] {script_name} finalizó correctamente.")
    else:
        print(f"[PIPELINE] {script_name} terminó con código {proc.returncode}.")
    return proc.returncode


def run_extract() -> int:
    """Genera/actualiza propuestas.csv a partir de los archivos fuente."""
    return run_script("analizar_planes.py")


def run_verify() -> int:
    """Ejecuta la verificación de propuestas (evaluacion_propuestas.csv)."""
    return run_script("evaluar_propuestas.py")


def run_graphics() -> int:
    """Genera los gráficos en la carpeta ./graficos."""
    csv = ROOT_DIR / "propuestas.csv"
    eval_csv = ROOT_DIR / "evaluacion_propuestas.csv"
    args: List[str] = ["--csv", str(csv)]
    if eval_csv.exists():
        args.extend(["--evaluacion", str(eval_csv)])
    args.extend(["--out", str(ROOT_DIR / "graficos")])
    return run_script("graficos_propuestas.py", *args)


def main() -> None:
    actions: List[Tuple[str, Callable[[], int]]] = [
        ("extract", run_extract),
        ("verify", run_verify),
        ("graphics", run_graphics),
    ]

    parser = argparse.ArgumentParser(description="Ejecuta el pipeline de análisis de propuestas.")
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=[label for label, _ in actions],
        help="Pasos a ejecutar (extract, verify, graphics). Por defecto se ejecutan todos en orden.",
    )
    args = parser.parse_args()

    if args.steps:
        selected = [action for action in actions if action[0] in args.steps]
    else:
        selected = actions

    print("\n[PIPELINE] Inicio de ejecución\n" + "-" * 40)
    failures = 0
    for key, func in selected:
        print(f"\n[PIPELINE] Paso {key}")
        result = func()
        if result != 0:
            failures += 1
            print(f"[PIPELINE] Paso {key} finalizó con errores.")
    print("\n[PIPELINE] Resumen\n" + "-" * 40)
    if failures:
        print(f"Finalizado con {failures} error(es). Revise los mensajes anteriores.")
    else:
        print("Todos los procesos seleccionados finalizaron correctamente.")


if __name__ == "__main__":
    main()
