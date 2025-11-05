#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analizador de planes de gobierno (PDF/TXT) -> propuestas.csv  (v3 mejorado)

Mejoras clave vs v2:
- Segmentación por bloques (no solo líneas).
- Reconocimiento de subtítulos y su fusión con el primer párrafo.
- Viñetas multilínea completas.
- Tokenización de oraciones en español para evitar cortes.
- Reconoce nominalizaciones y verbos modales.
- Más tolerancia ante texto desordenado por OCR.
"""

from __future__ import annotations
import os
import re
import csv
import sys
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Iterable

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_DIR = BASE_DIR / "datos" / "propuestas"
OUTPUT_CSV = BASE_DIR / "propuestas.csv"
EXTRACTS_DIR = BASE_DIR / "extractos"
LOG_FILE = BASE_DIR / "log.txt"
KEYWORD_FILE = BASE_DIR / "datos" / "verbos.txt"

# =============================
# Bancos de palabras (por defecto)
# =============================

DEFAULT_ACTION_VERBS = [
    "implementar","promover","fortalecer","construir","fomentar","garantizar",
    "mejorar","crear","desarrollar","optimizar","ampliar","modernizar",
    "impulsar","coordinar","articular","reducir","prevenir","asegurar",
    "potenciar","consolidar","mantener","rehabilitar","instalar","adquirir",
    "dotar","capacitar","regular","incentivar","digitalizar","integrar",
    "monitorear","evaluar","transparentar","actualizar","resguardar",
    "ordenar","planificar","restaurar","recuperar","gestionar","asegurar",
    "sostener","apoyar","articular","proteger","impulsar","financiar",
    "supervisar"
]
DEFAULT_ACTION_NOMINALIZATIONS = [
    r"implementaci[oó]n\s+de", r"mejora\s+de", r"fortalecimiento\s+de", r"creaci[oó]n\s+de",
    r"desarrollo\s+de", r"ampliaci[oó]n\s+de", r"modernizaci[oó]n\s+de", r"impulso\s+de",
    r"digitalizaci[oó]n\s+de", r"instalaci[oó]n\s+de", r"adquisici[oó]n\s+de",
    r"recuperaci[oó]n\s+de", r"implementaci[oó]n\s+integral\s+de", r"renovaci[oó]n\s+de",
    r"fortalecimiento\s+institucional\s+para"
]

DEFAULT_HARD_SIGNS = [
    r"\bpresupuesto\w*", r"\bfinanciam\w*", r"\bcronograma\w*", r"\bconvenio\w*",
    r"\bresponsable\w*|\bencargad[oa]s?\b", r"\bequipo\s+t[eé]cnico\w*",
    r"\b(etapas?|hitos?)\b", r"\bmetodolog\w*", r"\bindicador(?:es)?\b",
    r"\bfuente(?:s)?\s+de\s+financiamiento\b", r"\bexpediente\s+t[eé]cnico\b",
    r"\bmatriz\s+de\s+seguimiento\b", r"\bplan\s+operativo\b"
]
DEFAULT_MEDIUM_SIGNS = [
    r"\b(se\s+)?promover\w*\b", r"\b(se\s+)?buscar\w*\b", r"\b(se\s+)?evaluar\w*\b",
    r"\b(se\s+)?gestionar\w*\b", r"\b(se\s+)?impulsar\w*\b", r"\b(se\s+)?fomentar\w*\b",
    r"\b(se\s+)?incentivar\w*\b", r"\b(se\s+)?priorizar\w*\b", r"\b(se\s+)?coordinar\w*\b"
]
DEFAULT_LOW_SIGNS = [
    r"\bse\s+propone\b", r"\b(se\s+)?plantea\w*\b", r"\baspira\w*\b", r"\bintenci[oó]n\s+de\b",
    r"\b(se\s+)?busca\w*\b", r"\b(se\s+)?pretende\w*\b"
]

def load_keyword_banks(path: Path) -> Dict[str, List[str]]:
    banks = {
        "ACTION_VERBS": list(DEFAULT_ACTION_VERBS),
        "ACTION_NOMINALIZATIONS": list(DEFAULT_ACTION_NOMINALIZATIONS),
        "HARD_SIGNS": list(DEFAULT_HARD_SIGNS),
        "MEDIUM_SIGNS": list(DEFAULT_MEDIUM_SIGNS),
        "LOW_SIGNS": list(DEFAULT_LOW_SIGNS),
    }
    if not path.exists():
        return banks

    current = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1].strip().upper()
            if current not in banks:
                banks[current] = []
            else:
                banks[current].clear()
            continue
        if current:
            value = line.split("#", 1)[0].strip()
            if value:
                banks.setdefault(current, []).append(value)
    return banks

KEYWORD_BANKS = load_keyword_banks(KEYWORD_FILE)

ACTION_VERBS = KEYWORD_BANKS.get("ACTION_VERBS", list(DEFAULT_ACTION_VERBS)) or list(DEFAULT_ACTION_VERBS)
ACTION_NOMINALIZATIONS = KEYWORD_BANKS.get("ACTION_NOMINALIZATIONS", list(DEFAULT_ACTION_NOMINALIZATIONS)) or list(DEFAULT_ACTION_NOMINALIZATIONS)
HARD_SIGNS = KEYWORD_BANKS.get("HARD_SIGNS", list(DEFAULT_HARD_SIGNS)) or list(DEFAULT_HARD_SIGNS)
MEDIUM_SIGNS = KEYWORD_BANKS.get("MEDIUM_SIGNS", list(DEFAULT_MEDIUM_SIGNS)) or list(DEFAULT_MEDIUM_SIGNS)
LOW_SIGNS = KEYWORD_BANKS.get("LOW_SIGNS", list(DEFAULT_LOW_SIGNS)) or list(DEFAULT_LOW_SIGNS)

ACTION_VERBS_RE = re.compile(r"^\s*(?:" + "|".join(re.escape(v) for v in ACTION_VERBS) + r")\b", re.I)

# =============================
# Viñetas, encabezados, y frases
# =============================

PAGE_MARK_PAT = re.compile(r"^\s*(p[aá]gina\s*\d+|\d+)\s*$", re.I)
HEADER_FOOTER_PAT = re.compile(r"(municipalidad|plan de gobierno|partido|movimiento|alcald[ií]a)", re.I)

HEADING_END = re.compile(r".*[:：]$")
ALL_CAPS = re.compile(r"^[A-ZÁÉÍÓÚÜÑ0-9\s\-/]+$")
TITLE_CASE = re.compile(r"^(?:[A-ZÁÉÍÓÚ][a-záéíóúüñ]+(?:\s+|$)){1,8}$")
VERB_FINITE_HEAD = re.compile(
    r"^\s*(?:se\s+(?:va(?:\s+a)?|van(?:\s+a)?)\s+a|vamos\s+a|se\s+|[a-záéíóúüñ]+(?:remos|rán|rá|ré|ría|rían)\b)",
    re.I,
)
BULLET_PAT_STRICT = re.compile(
    r"""^\s*(?:[-–—•●■◦▪]|(?:\d{1,3}|[ivxlcdm]+|[a-zA-Z])[\.)])(?:\s+|$)""", re.I
)

SENT_END = re.compile(r"([.!?؛…]|(?<=\))\.)\s+")
ABBREV = re.compile(
    r"\b(?:Sr|Sra|Sres|Dr|Dra|Ing|Lic|Mg|PhD|p\.\s?ej|p\.e\.|etc|aprox|Av|No|N°|n°)\.$", re.I
)

# =============================
# Timeline / Factibilidad / Observadores
# =============================

YEAR_RANGE_PAT = re.compile(r"\b(20\d{2})\s*(?:–|-|—|-|to|al?|a)\s*(20\d{2})\b", re.I)
SINGLE_YEAR_PAT = re.compile(r"\b(20\d{2})\b")
RELATIVE_PATS = [
    re.compile(r"\b(primer|segundo|tercer|cuarto)\s+a[nñ]o(?:\s+de\s+(gesti[oó]n|gobierno))?\b", re.I),
    re.compile(r"\bprimer\s+semestre\b", re.I),
    re.compile(r"\bprimeros?\s+(?:seis|6|doce|12)\s+meses\b", re.I),
    re.compile(r"\b(en|durante)\s+los\s+primeros?\s+\w+\b", re.I),
    re.compile(r"\b(corto|mediano|largo)\s+plazo\b", re.I),
    re.compile(r"\bmensual(?:mente)?\b|\btrimestral(?:mente)?\b|\bsemestral(?:mente)?\b|\banual(?:mente)?\b", re.I),
]

OBS_MAP = {
    "participación": [r"\bparticipaci[oó]n\b", r"\bvecin[oa]s?\b", r"\bcomunidad(?:es)?\b", r"\borganizaciones?\b"],
    "monitoreo": [r"\bmonitoreo\b", r"\bmonitor(?:ear|eo)\b"],
    "evaluación": [r"\bevaluaci[oó]n\b", r"\bevaluar\b"],
    "rendición de cuentas": [r"\brendici[oó]n\s+de\s+cuentas\b", r"\btransparenc\w*\b"],
    "sostenibilidad": [r"\bsostenibil\w*\b", r"\bsostenible\w*\b"],
    "indicadores": [r"\bindicador(?:es)?\b", r"\bkpi\b", r"\bmetas?\b"],
    "beneficiarios": [r"\bbeneficiari\w*\b", r"\bcobertura\b", r"\balcance\b"],
    "costo-beneficio": [r"\bcosto[- ]beneficio\b", r"\beficiencia\b", r"\broi\b"],
}

# =============================
# Extracción y normalización
# =============================

def extract_text_pdf(path: Path) -> str:
    text = ""
    try:
        from pdfminer.high_level import extract_text as pdfminer_extract
        text = pdfminer_extract(str(path)) or ""
        if text.strip():
            return text
    except Exception as e:
        logging.warning(f"pdfminer falló en {path.name}: {e}")
    try:
        import PyPDF2
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            pages = [page.extract_text() or "" for page in reader.pages]
            text = "\n".join(pages)
    except Exception as e:
        logging.error(f"PyPDF2 también falló en {path.name}: {e}")
        text = ""
    return text

def extract_text_file(path: Path) -> str:
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            return path.read_text(encoding=enc)
        except Exception:
            continue
    with open(path, "rb") as f:
        raw = f.read()
    return raw.decode("utf-8", errors="ignore")

def extract_text_any(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return extract_text_pdf(path)
    if path.suffix.lower() == ".txt":
        return extract_text_file(path)
    return ""

def normalize_text(text: str) -> str:
    if not text: return ""
    text = text.replace("\r", "\n")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    lines = []
    for raw in text.split("\n"):
        line = raw.strip()
        if PAGE_MARK_PAT.match(line): continue
        if len(line) <= 60 and HEADER_FOOTER_PAT.search(line): continue
        line = re.sub(r"\s+", " ", line).strip()
        lines.append(line)
    out = []
    last_blank = False
    for ln in lines:
        if not ln:
            if not last_blank: out.append("")
            last_blank = True
        else:
            out.append(ln); last_blank = False
    return "\n".join(out).strip()

# =============================
# Segmentación avanzada
# =============================

def is_bullet(line: str) -> bool:
    return bool(BULLET_PAT_STRICT.match(line.strip()))

def is_heading(line: str) -> bool:
    s = line.strip()
    if not s: return False
    if HEADING_END.search(s): return True
    if ALL_CAPS.match(s) and len(s.split()) <= 12: return True
    if TITLE_CASE.match(s) and not VERB_FINITE_HEAD.match(s): return True
    return False

def split_sentences(text: str) -> List[str]:
    if not text: return []
    TMP = "§§DOT§§"
    def _protect_abbrev(pattern: re.Pattern[str], src: str) -> str:
        return pattern.sub(lambda m: m.group(1) + TMP, src)

    primary_abbrev = re.compile(r"(\b(?:Sr|Sra|Sres|Dr|Dra|Ing|Lic|Mg|PhD|p)\.)(\s*)", re.I)
    secondary_abbrev = re.compile(r"(\b(?:Av|No|N°|n°|p\.\s?ej|p\.e|etc|aprox)\.)(\s*)", re.I)

    protected = _protect_abbrev(primary_abbrev, text)
    protected = _protect_abbrev(secondary_abbrev, protected)
    parts = re.split(SENT_END, protected)
    out, buf = [], ""
    for chunk in parts:
        if not chunk: continue
        buf += chunk
        if SENT_END.match(chunk[-2:]) or re.search(r"[.!?…]$", chunk):
            out.append(buf.strip())
            buf = ""
    if buf.strip():
        out.append(buf.strip())
    out = [s.replace(TMP, " ") for s in out]
    return [s for s in out if len(s.strip()) >= 2]

def looks_like_proposal_block(text: str) -> bool:
    t = text.strip()
    if not t: return False
    if is_bullet(t): return True
    if ACTION_VERBS_RE.match(t): return True
    if VERB_FINITE_HEAD.match(t): return True
    if any(re.search(p, t, flags=re.I) for p in ACTION_NOMINALIZATIONS): return True
    words = t.split()
    if words and words[0][0].isupper():
        head = " ".join(words[:4]).lower()
        if any(head.startswith(v) for v in ACTION_VERBS): return True
    return False

def segment_blocks(text: str) -> List[str]:
    lines = [ln.strip() for ln in text.split("\n")]
    blocks = []
    i, n = 0, len(lines)

    def collect_until_stop(start: int, stop_pred) -> Tuple[str, int]:
        buf, j = [], start
        while j < n:
            ln = lines[j]
            if stop_pred(j, ln): break
            if ln: buf.append(ln)
            j += 1
        return " ".join(buf).strip(), j

    while i < n:
        ln = lines[i]
        if not ln:
            i += 1; continue

        if is_bullet(ln):
            ln = re.sub(BULLET_PAT_STRICT, "", ln).strip()
            buf = [ln] if ln else []
            i += 1
            while i < n and lines[i] and not is_bullet(lines[i]) and not is_heading(lines[i]):
                buf.append(lines[i]); i += 1
            blocks.append(" ".join(buf).strip())
            continue

        if is_heading(ln):
            head = ln
            i += 1
            para, i2 = collect_until_stop(i, lambda j, l: not l or is_heading(l) or is_bullet(l))
            if para:
                sents = split_sentences(para)
                tail = " ".join(sents[:2]) if sents else para
                blocks.append(f"{head} {tail}".strip(" :"))
                i = i2
            else:
                blocks.append(head.strip(" :"))
            continue

        para, i2 = collect_until_stop(i, lambda j, l: not l or is_heading(l) or is_bullet(l))
        blocks.append(para if para else ln)
        i = i2

    return [re.sub(r"\s+", " ", b).strip() for b in blocks if b.strip()]

def segment_proposals(text: str, max_sentences_per_prop: int = 3) -> List[str]:
    blocks = segment_blocks(text)
    proposals = []
    for blk in blocks:
        if not blk or len(blk) < 5: continue
        if looks_like_proposal_block(blk):
            sents = split_sentences(blk)
            if not sents:
                if len(blk) >= 25: proposals.append(blk)
                continue
            take = min(max_sentences_per_prop, len(sents))
            prop = " ".join(sents[:take]).strip()
            if len(prop) < 25 and len(sents) > take:
                prop = " ".join(sents[:take+1]).strip()
            if len(prop) >= 20:
                proposals.append(prop)
            continue
        if proposals and blk and blk[0].islower():
            prev = proposals.pop()
            merged = f"{prev} {blk}"
            sents = split_sentences(merged)
            prop = " ".join(sents[:max_sentences_per_prop+1]) if sents else merged
            proposals.append(prop.strip())
    seen, dedup = set(), []
    for p in proposals:
        key = re.sub(r"\W+", "", p.lower())[:220]
        if key not in seen:
            seen.add(key); dedup.append(p)
    return dedup

# =============================
# Funciones auxiliares
# =============================

def extract_timeline(text: str) -> str:
    t = str(text)
    m = YEAR_RANGE_PAT.search(t)
    if m: return f"{m.group(1)}–{m.group(2)}"
    for rx in RELATIVE_PATS:
        m = rx.search(t)
        if m: return m.group(0)
    m = SINGLE_YEAR_PAT.search(t)
    if m: return m.group(1)
    return "No definido"

def score_feasibility(text: str) -> str:
    t = str(text).lower()
    hard = sum(1 for pat in HARD_SIGNS if re.search(pat, t))
    med  = sum(1 for pat in MEDIUM_SIGNS if re.search(pat, t))
    low  = sum(1 for pat in LOW_SIGNS if re.search(pat, t))
    if hard >= 2: return "Alta"
    if hard == 1 or med >= 2: return "Media"
    if med == 1 or low >= 1: return "Baja"
    return "No definida"

def extract_observers(text: str) -> str:
    t = str(text)
    found = []
    for label, pats in OBS_MAP.items():
        if any(re.search(p, t, flags=re.I) for p in pats):
            found.append(label)
    return ", ".join(found)

def infer_district_from_filename(path: Path) -> str:
    base = re.sub(r"[_\-]+", " ", path.stem).strip()
    base = re.sub(r"\s+", " ", base)
    base = re.sub(r"^(distrito\s+de\s+)", "", base, flags=re.I)
    return " ".join(w.capitalize() for w in base.split())

def save_extract(district: str, idx: int, text: str):
    d = EXTRACTS_DIR / district
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{idx:04d}.txt").write_text(text, encoding="utf-8")

# =============================
# Procesamiento principal
# =============================

def process_file(path: Path) -> List[Dict[str, str]]:
    raw = extract_text_any(path)
    if not raw.strip():
        logging.error(f"Texto vacío o no extraíble: {path.name}")
        return []
    clean = normalize_text(raw)
    if not clean:
        logging.warning(f"Texto vacío tras limpieza: {path.name}")
        return []
    props = segment_proposals(clean, max_sentences_per_prop=3)
    if not props:
        logging.warning(f"No se detectaron propuestas en: {path.name}")
        return []
    district = infer_district_from_filename(path)
    rows = []
    for i, prop in enumerate(props, 1):
        rows.append({
            "distrito": district,
            "propuesta": prop,
            "timeline": extract_timeline(prop),
            "factibilidad": score_feasibility(prop),
            "observadores_de_exito": extract_observers(prop)
        })
        save_extract(district, i, prop)
    return rows

def write_csv(rows: List[Dict[str, str]], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["distrito","propuesta","timeline","factibilidad","observadores_de_exito"]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows: w.writerow(r)

def iter_input_files(root: Path) -> Iterable[Path]:
    for p in sorted(root.glob("*")):
        if p.is_file() and p.suffix.lower() in {".pdf",".txt"}:
            yield p

def main():
    import argparse
    logging.basicConfig(filename=str(LOG_FILE), filemode="w", level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Analiza planes de gobierno y extrae propuestas -> CSV (v3).")
    parser.add_argument("-i","--input", type=str, default=str(DEFAULT_INPUT_DIR),
                        help="Carpeta con .pdf/.txt (por defecto: ./datos/propuestas)")
    parser.add_argument("-o","--output", type=str, default=str(OUTPUT_CSV),
                        help="CSV de salida (por defecto: ./propuestas.csv)")
    parser.add_argument("--skip-charts", action="store_true",
                        help="No generar gráficos a partir del CSV resultante.")
    args = parser.parse_args()

    root = Path(args.input)
    if not root.exists():
        print(f"[ERROR] Carpeta no encontrada: {root}")
        sys.exit(1)

    files = list(iter_input_files(root))
    total_files = len(files)
    all_rows = []
    if total_files == 0:
        print("[AVISO] No se encontraron archivos .pdf/.txt en la carpeta de entrada.")
        sys.exit(0)

    for idx, path in enumerate(files, 1):
        progress_prefix = f"[{idx}/{total_files}]"
        print(f"{progress_prefix} Procesando {path.name}", end="\r", flush=True)
        try:
            file_rows = process_file(path)
            all_rows.extend(file_rows)
            print(f"{progress_prefix} Procesado {path.name} | propuestas: {len(file_rows)}")
        except Exception as e:
            logging.exception(f"Fallo procesando {path.name}: {e}")
            print(f"{progress_prefix} Error procesando {path.name} (ver log.txt)")

    if not all_rows:
        print("[AVISO] No se detectaron propuestas. Revisa log.txt y ajusta verbos/patrones.")
    else:
        write_csv(all_rows, Path(args.output))
        import collections
        by_fact = collections.Counter(r["factibilidad"] for r in all_rows)
        by_tl = collections.Counter("No definido" if r["timeline"]=="No definido"
                                    else ("Rango" if re.search(r"20\d{2}–20\d{2}", r["timeline"]) else "Otro")
                                    for r in all_rows)
        print(f"[OK] CSV: {args.output} | filas: {len(all_rows)} | archivos: {total_files}")
        print(f"Factibilidad: {dict(by_fact)} | Timeline: {dict(by_tl)}")
        if args.skip_charts:
            print("[GRAFICOS] Omitido por bandera --skip-charts.")
        else:
            print(f"[GRAFICOS] Generando gráficos a partir de {args.output}...", flush=True)
            try:
                from graficos_propuestas import generate_all_charts
                charts, warnings = generate_all_charts(Path(args.output))
            except Exception as exc:
                logging.exception("Error generando gráficos: %s", exc)
                print(f"[GRAFICOS] Error generando gráficos: {exc}")
            else:
                if charts:
                    for path in charts:
                        print(f"[GRAFICOS] ✓ {path}")
                else:
                    print("[GRAFICOS] No se generaron gráficos.")
                for warn in warnings:
                    print(f"[GRAFICOS] Aviso: {warn}")
        print("El código detecta las propuestas con regrex, recomiendo de todas formas hacer una revisión humana (al azar, por ejemplo)")

if __name__ == "__main__":
    main()
