#!/usr/bin/env python3
"""
Tooling to contrast the municipal proposals listed in `propuestas.csv`
against the evidence available in ./datos/informacion/*.csv.  For each proposal we try
to infer a relevant dataset based on keywords and then classify its status as
`cumplida`, `en_progreso`, `no_cumplida` o `sin_datos`.

The script writes the evaluation to `evaluacion_propuestas.csv` alongside a
textual rationale for the assigned status.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATOS_DIR = BASE_DIR / "datos" / "informacion"
PROPOSALS_PATH = BASE_DIR / "propuestas.csv"
OUTPUT_PATH = BASE_DIR / "evaluacion_propuestas.csv"

StatusResult = Tuple[str, str]
StatusFunc = Callable[[str], Optional[StatusResult]]

# Status labels we use in downstream logic.
ACHIEVED = "cumplida"
IN_TRACK = "en_progreso"
NOT_ACHIEVED = "no_cumplida"
NO_DATA = "sin_datos"

INFOBRAS_FILENAME_MAP: Dict[str, str] = {
    # Explicit aliases help when the dataset usa abreviaturas poco comunes.
    "ate": "ate",
    "chaclacayo": "chaclacayo",
    "cieneguilla": "cieneguilla",
    "el agustino": "el agustino",
    "la molina": "la molina",
    "lurigancho": "lurigancho",
    "san isidro": "san isidro",
    "san juan de lurigancho": "san juan d lurigancho",
    "santa anita": "santa anita",
    "santiago de surco": "surco",
    "surquillo": "surquillo",
}


def _strip_accents(value: str) -> str:
    """Return a lowercase, accent-free version of the provided text."""
    normalized = unicodedata.normalize("NFD", value or "")
    without_marks = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    return without_marks.lower()


def _normalize_column_label(label: str) -> str:
    """Utility to normalize column labels for fuzzy matching."""
    text = _strip_accents(str(label))
    text = text.replace("%", " porcentaje ")
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def _keyword_present(text: str, tokens: Iterable[str]) -> bool:
    """Check if any token is present in the normalized text."""
    norm_text = _strip_accents(text)
    return any(token in norm_text for token in tokens)


def _load_csv(path: Path, **kwargs) -> pd.DataFrame:
    """Read a CSV file with pandas and bubble up a clearer error on failure."""
    try:
        return pd.read_csv(path, **kwargs)
    except Exception as exc:  # pragma: no cover - defensive programming
        raise RuntimeError(f"Unable to read CSV file at {path}") from exc


def _load_csv_dynamic(path: Path) -> Optional[pd.DataFrame]:
    """
    Best-effort CSV reader that tries several encoding/separator combinations.
    Returns None when every attempt fails so the caller can skip the file.
    """
    attempts = [
        {"encoding": "utf-8", "on_bad_lines": "skip", "low_memory": False},
        {"encoding": "utf-8-sig", "on_bad_lines": "skip", "low_memory": False},
        {"encoding": "latin-1", "on_bad_lines": "skip", "low_memory": False},
        {"encoding": "latin-1", "sep": ";", "engine": "python", "on_bad_lines": "skip", "low_memory": False},
        {"encoding": "utf-8", "sep": ";", "engine": "python", "on_bad_lines": "skip", "low_memory": False},
        {"sep": None, "engine": "python", "on_bad_lines": "skip", "low_memory": False},
    ]
    for opts in attempts:
        try:
            return pd.read_csv(path, **opts)
        except Exception:
            continue
    return None


def _normalize_municipality_name(raw_name: str) -> str:
    """
    Take raw entries such as '150103-301252: MUNICIPALIDAD DISTRITAL DE ATE - VITARTE'
    and return a normalized key usable for matching.
    """
    if not isinstance(raw_name, str):
        return ""
    name = raw_name.split(":")[-1].strip()
    return _strip_accents(name)


def _threshold_classification(
    value: float,
    achieved_threshold: float,
    in_track_threshold: float,
    invert: bool = False,
) -> str:
    """
    Helper to transform a scalar metric into one of the three ordered statuses.

    Parameters
    ----------
    value:
        Metric to evaluate.
    achieved_threshold:
        Boundary at or above which we mark the proposal as achieved (or below
        if `invert` is True).
    in_track_threshold:
        Boundary for distinguishing between `in_track` and `not_achieved`.
        Must be <= achieved_threshold when `invert` is False.
    invert:
        When True we consider lower values better (useful for crime metrics).
    """
    if value is None or pd.isna(value):
        return NO_DATA

    if invert:
        if value <= achieved_threshold:
            return ACHIEVED
        if value <= in_track_threshold:
            return IN_TRACK
        return NOT_ACHIEVED

    if value >= achieved_threshold:
        return ACHIEVED
    if value >= in_track_threshold:
        return IN_TRACK
    return NOT_ACHIEVED


class DatasetDiscovery:
    """
    Scans ./datos/informacion looking for CSV files and classifies them based on
    their headers/content so downstream logic can stay agnostic to filenames.
    """

    def __init__(self, root: Path):
        self.root = root
        self.frames: Dict[str, pd.DataFrame] = {}
        self.infobras_tables: List[Tuple[Path, pd.DataFrame]] = []
        if not root.exists():
            return
        self._scan_directory()

    def get(self, name: str) -> pd.DataFrame:
        frame = self.frames.get(name)
        if frame is None:
            return pd.DataFrame()
        return frame.copy(deep=False)

    def build_infobras_lookup(self) -> Dict[str, pd.DataFrame]:
        lookup: Dict[str, pd.DataFrame] = {}
        for path, table in self.infobras_tables:
            split = self._split_by_district(table)
            if not split:
                key = _strip_accents(path.stem)
                if not key:
                    continue
                lookup[key] = self._merge_frames(lookup.get(key), table)
                continue
            for district_key, subset in split.items():
                lookup[district_key] = self._merge_frames(lookup.get(district_key), subset)
        return {key: df.reset_index(drop=True) for key, df in lookup.items()}

    def _scan_directory(self) -> None:
        for path in sorted(self.root.rglob("*.csv")):
            frame = _load_csv_dynamic(path)
            if frame is None or frame.empty:
                continue
            category = self._classify(path, frame)
            if category is None:
                continue
            if category == "infobras":
                self.infobras_tables.append((path, frame))
                continue
            self.frames[category] = self._merge_frames(self.frames.get(category), frame)

    def _classify(self, path: Path, frame: pd.DataFrame) -> Optional[str]:
        columns = [_normalize_column_label(col) for col in frame.columns]
        normalized_path = _normalize_column_label(path.stem)

        def has_fragment(fragment: str) -> bool:
            fragment = fragment.lower()
            return any(fragment in col for col in columns)

        def has_all(*fragments: str) -> bool:
            return all(has_fragment(fragment) for fragment in fragments)

        if has_fragment("violencia") or "violencia" in normalized_path:
            if has_fragment("cantidad"):
                return "violence_reports"

        if has_fragment("dist_hecho") or (has_fragment("dist") and has_fragment("hecho")):
            if has_fragment("cantidad") or has_fragment("numero_casos"):
                return "police_reports"

        if has_fragment("arbol"):
            return "tree_management"

        if has_fragment("agenda") or has_fragment("parque") or "cultural" in normalized_path:
            return "cultural_agenda"

        if has_fragment("control") and has_fragment("informe"):
            return "control_services"

        if has_fragment("pim") and (has_fragment("recaud") or has_fragment("ingreso")):
            return "revenue_execution"

        if has_fragment("municipalidad") and has_fragment("avance") and (
            has_fragment("gasto") or "presupuesto" in normalized_path or has_fragment("presupuesto")
        ):
            return "budget_execution"

        if has_fragment("municipalidad") and has_fragment("avance") and (
            has_fragment("inversion") or has_fragment("proyecto") or has_fragment("obra")
        ):
            return "investment_execution"

        if has_fragment("discap"):
            return "disability_spending"

        if has_fragment("ninez") or has_fragment("infancia") or has_fragment("adolescen") or has_fragment("juventud"):
            return "childhood_spending"

        if has_fragment("estado") and has_fragment("obra"):
            return "infobras"

        return None

    @staticmethod
    def _merge_frames(existing: Optional[pd.DataFrame], new: pd.DataFrame) -> pd.DataFrame:
        if existing is None or existing.empty:
            return new.copy()
        return pd.concat([existing, new], ignore_index=True, sort=False)

    @staticmethod
    def _split_by_district(frame: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        for column in frame.columns:
            normalized = _normalize_column_label(column)
            if any(keyword in normalized for keyword in ("distrit", "municipalidad", "localidad")):
                groups: Dict[str, pd.DataFrame] = {}
                for key, subset in frame.groupby(column):
                    normalized_key = _strip_accents(str(key).strip())
                    if not normalized_key:
                        continue
                    groups[normalized_key] = subset.copy()
                if groups:
                    return groups
        return {}


@dataclass
class MunicipalDatasets:
    """Container for the pre-processed indicators sourced from ./datos/informacion."""

    budget_execution: pd.DataFrame
    investment_execution: pd.DataFrame
    disability_spending: pd.DataFrame
    childhood_spending: pd.DataFrame
    revenue_execution: pd.DataFrame
    police_reports: pd.DataFrame
    violence_reports: pd.DataFrame
    tree_management: pd.DataFrame
    cultural_agenda: pd.DataFrame
    control_services: pd.DataFrame
    infobras_lookup: Dict[str, pd.DataFrame]

    @classmethod
    def load(cls) -> "MunicipalDatasets":
        discovery = DatasetDiscovery(DATOS_DIR)
        return cls(
            budget_execution=discovery.get("budget_execution"),
            investment_execution=discovery.get("investment_execution"),
            disability_spending=discovery.get("disability_spending"),
            childhood_spending=discovery.get("childhood_spending"),
            revenue_execution=discovery.get("revenue_execution"),
            police_reports=discovery.get("police_reports"),
            violence_reports=discovery.get("violence_reports"),
            tree_management=discovery.get("tree_management"),
            cultural_agenda=discovery.get("cultural_agenda"),
            control_services=discovery.get("control_services"),
            infobras_lookup=discovery.build_infobras_lookup(),
        )


class ProposalEvaluator:
    """Encapsulates the keyword mapping and status evaluation logic."""

    def __init__(self, datasets: MunicipalDatasets, known_districts: Iterable[str]):
        self.datasets = datasets
        self.known_districts = [
            _strip_accents(str(district).strip())
            for district in known_districts
            if str(district).strip()
        ]
        self._prepare_municipality_index()
        self.keyword_map = self._build_keyword_map()
        self.handlers = self._build_handlers()

    def _prepare_municipality_index(self) -> None:
        """Create normalized views for a fast match when filtering by district."""
        def build_lookup(frame: pd.DataFrame, column: str) -> Dict[str, pd.DataFrame]:
            lookup: Dict[str, list] = {}
            if column not in frame.columns:
                return {}
            for _, row in frame.iterrows():
                raw = str(row[column])
                normalized_raw = _strip_accents(raw)
                for district_key in self.known_districts:
                    if district_key and district_key in normalized_raw:
                        lookup.setdefault(district_key, []).append(row.to_dict())
                        break
            return {key: pd.DataFrame(rows) for key, rows in lookup.items()}

        self.budget_lookup = build_lookup(self.datasets.budget_execution, "Municipalidad")
        self.investment_lookup = build_lookup(self.datasets.investment_execution, "Municipalidad")
        self.disability_lookup = build_lookup(self.datasets.disability_spending, "Municipalidad")
        self.childhood_lookup = build_lookup(self.datasets.childhood_spending, "Municipalidad")
        self.revenue_lookup = build_lookup(self.datasets.revenue_execution, "Municipalidad")
        self.control_lookup = build_lookup(self.datasets.control_services, "ENTIDAD")

        # Tree management uses 'DISTRITO' as identifier.
        tree_frame = self.datasets.tree_management.copy()
        tree_lookup: Dict[str, list] = {}
        if "DISTRITO" in tree_frame.columns:
            for _, row in tree_frame.iterrows():
                raw = str(row["DISTRITO"])
                normalized_raw = _strip_accents(raw)
                for district_key in self.known_districts:
                    if district_key and district_key in normalized_raw:
                        tree_lookup.setdefault(district_key, []).append(row.to_dict())
                        break
        self.tree_lookup = {key: pd.DataFrame(rows) for key, rows in tree_lookup.items()}

        # Police and violence datasets identify the district in DIST_HECHO / DIST_HECHO etc.
        police = self.datasets.police_reports.copy()
        if "DIST_HECHO" in police.columns:
            police["_normalized_key"] = police["DIST_HECHO"].apply(_strip_accents)
            self.police_lookup = {
                key: subset.drop(columns=["_normalized_key"])
                for key, subset in police.groupby("_normalized_key")
            }
        else:
            self.police_lookup = {}

        violence = self.datasets.violence_reports.copy()
        violence_column = None
        for candidate in ("DIST_HECHO", "DIST. HECHO", "Provincia", "DIST"):
            if candidate in violence.columns:
                violence_column = candidate
                break
        if violence_column:
            violence["_normalized_key"] = violence[violence_column].apply(_strip_accents)
            self.violence_lookup = {
                key: subset.drop(columns=["_normalized_key"])
                for key, subset in violence.groupby("_normalized_key")
            }
        else:
            self.violence_lookup = {}

        agenda = self.datasets.cultural_agenda.copy()
        potential_columns = [
            col for col in agenda.columns
            if any(word in col.lower() for word in ("lugar", "direccion", "parque"))
        ]
        if potential_columns:
            locations = []
            for _, row in agenda.iterrows():
                for col in potential_columns:
                    value = str(row.get(col, "")).strip()
                    if not value or value.lower() == "nan":
                        continue
                    locations.append(_strip_accents(value))
            self.agenda_locations = locations
        else:
            self.agenda_locations = []

    def _build_keyword_map(self) -> Dict[str, Iterable[str]]:
        return {
            "violence_women": (
                "violencia contra la mujer",
                "violencia familiar",
                "violencia de genero",
                "feminicidio",
                "mujer",
                "genero",
            ),
            "police_reports": (
                "seguridad",
                "delincuencia",
                "criminal",
                "patrull",
                "serenazgo",
                "videovigilancia",
                "camara",
                "drone",
                "dron",
                "alarmas",
                "policia",
                "pandill",
                "criminalidad",
            ),
            "budget_execution": (
                "presupuesto",
                "gasto",
                "pim",
                "pia",
                "financ",
                "ejecucion presupuestal",
                "certificacion",
                "compromiso",
                "tesorer",
                "tributario",
            ),
            "revenue_execution": (
                "ingreso",
                "recaudacion",
                "impuesto",
                "predial",
                "tributo",
                "cobranza",
            ),
            "investment_execution": (
                "obra",
                "infraestructura",
                "pista",
                "vereda",
                "carretera",
                "puente",
                "colegio",
                "hospital",
                "proyecto de inversion",
                "inversion publica",
            ),
            "infobras": (
                "infobras",
                "ejecucion de obra",
                "avance de obra",
            ),
            "disability_spending": (
                "discapacidad",
                "inclusion",
                "diversidad funcional",
            ),
            "childhood_spending": (
                "ninez",
                "niñez",
                "adolescencia",
                "infancia",
                "jovenes",
                "juventud",
            ),
            "tree_management": (
                "arbol",
                "arbolado",
                "forestacion",
                "areas verdes",
                "parque",
                "jardin",
            ),
            "cultural_agenda": (
                "cultura",
                "cultural",
                "evento cultural",
                "arte",
                "artesania",
                "deporte",
                "recreacion",
            ),
            "control_services": (
                "control",
                "contraloria",
                "fiscalizacion",
                "transparencia",
                "auditoria",
            ),
        }

    def _build_handlers(self) -> Dict[str, StatusFunc]:
        return {
            "budget_execution": self._handle_budget_execution,
            "investment_execution": self._handle_investment_execution,
            "disability_spending": self._handle_disability_spending,
            "childhood_spending": self._handle_childhood_spending,
            "revenue_execution": self._handle_revenue_execution,
            "violence_women": self._handle_violence_reports,
            "police_reports": self._handle_police_reports,
            "tree_management": self._handle_tree_management,
            "cultural_agenda": self._handle_cultural_agenda,
            "control_services": self._handle_control_services,
            "infobras": self._handle_infobras,
        }

    def find_category(self, proposal_text: str) -> Optional[str]:
        normalized = _strip_accents(proposal_text or "")
        for category in (
            "violence_women",
            "police_reports",
            "budget_execution",
            "revenue_execution",
            "investment_execution",
            "infobras",
            "disability_spending",
            "childhood_spending",
            "tree_management",
            "cultural_agenda",
            "control_services",
        ):
            if category not in self.keyword_map:
                continue
            if _keyword_present(normalized, self.keyword_map[category]):
                return category
        return None

    def evaluate(self, district: str, proposal_text: str) -> StatusResult:
        category = self.find_category(proposal_text)
        handler = self.handlers.get(category or "")
        if category and handler:
            result = handler(district)
            if result:
                return result
        return NO_DATA, "No contamos con indicadores cuantitativos asociados a esta propuesta."

    def _latest_record(self, lookup: Dict[str, pd.DataFrame], district: str) -> Optional[pd.Series]:
        key = _strip_accents(district)
        if key not in lookup:
            return None
        frame = lookup[key].copy()
        if frame.empty:
            return None
        if "Year" in frame.columns:
            frame = frame.sort_values("Year", ascending=False)
        elif "ANIO" in frame.columns:
            frame = frame.sort_values("ANIO", ascending=False)
        return frame.iloc[0]

    def _handle_budget_execution(self, district: str) -> Optional[StatusResult]:
        record = self._latest_record(self.budget_lookup, district)
        if record is None:
            return None
        avance = float(record.get("Avance %", float("nan")))
        year = int(record.get("Year", 0)) if not pd.isna(record.get("Year", float("nan"))) else ""
        status = _threshold_classification(avance, achieved_threshold=85.0, in_track_threshold=70.0)
        detail = f"Ejecución del gasto {avance:.1f}% ({year})."
        return status, detail

    def _handle_investment_execution(self, district: str) -> Optional[StatusResult]:
        record = self._latest_record(self.investment_lookup, district)
        if record is None:
            return None
        avance = float(record.get("Avance %", float("nan")))
        year = int(record.get("Year", 0)) if not pd.isna(record.get("Year", float("nan"))) else ""
        status = _threshold_classification(avance, achieved_threshold=80.0, in_track_threshold=55.0)
        detail = f"Ejecución de proyectos de inversión {avance:.1f}% ({year})."
        return status, detail

    def _handle_disability_spending(self, district: str) -> Optional[StatusResult]:
        record = self._latest_record(self.disability_lookup, district)
        if record is None:
            return None
        avance = float(record.get("Avance %", float("nan")))
        year = int(record.get("Year", 0)) if not pd.isna(record.get("Year", float("nan"))) else ""
        status = _threshold_classification(avance, achieved_threshold=70.0, in_track_threshold=45.0)
        detail = f"Gasto orientado a personas con discapacidad {avance:.1f}% ({year})."
        return status, detail

    def _handle_childhood_spending(self, district: str) -> Optional[StatusResult]:
        record = self._latest_record(self.childhood_lookup, district)
        if record is None:
            return None
        avance = float(record.get("Avance %", float("nan")))
        year = int(record.get("Year", 0)) if not pd.isna(record.get("Year", float("nan"))) else ""
        status = _threshold_classification(avance, achieved_threshold=70.0, in_track_threshold=45.0)
        detail = f"Recursos para niñez y adolescencia ejecutados {avance:.1f}% ({year})."
        return status, detail

    def _handle_revenue_execution(self, district: str) -> Optional[StatusResult]:
        record = self._latest_record(self.revenue_lookup, district)
        if record is None:
            return None
        try:
            pim = float(record["PIM"])
            recaudado = float(record["Recaudado"])
        except KeyError:
            return None
        ratio = recaudado / pim if pim else float("nan")
        status = _threshold_classification(ratio, achieved_threshold=0.95, in_track_threshold=0.7)
        detail = f"Recaudación/PIM = {ratio:.2f}."
        return status, detail

    def _handle_police_reports(self, district: str) -> Optional[StatusResult]:
        key = _strip_accents(district)
        subset = self.police_lookup.get(key)
        if subset is None or subset.empty:
            return None
        # Aggregate yearly totals and compare the two most recent years.
        subset = subset.copy()
        subset["cantidad"] = pd.to_numeric(subset["cantidad"], errors="coerce")
        by_year = subset.groupby("ANIO")["cantidad"].sum().dropna()
        if by_year.empty:
            return None
        sorted_years = sorted(by_year.index)
        if len(sorted_years) == 1:
            latest_year = sorted_years[-1]
            total = by_year.iloc[-1]
            status = IN_TRACK if total == 0 else NOT_ACHIEVED
            detail = f"Solo disponible {latest_year}; total denuncias {int(total)}."
            return status, detail
        latest_year = sorted_years[-1]
        previous_year = sorted_years[-2]
        latest_value = by_year.loc[latest_year]
        previous_value = by_year.loc[previous_year]
        ratio = latest_value / previous_value if previous_value else float("inf")
        status = _threshold_classification(ratio, achieved_threshold=0.85, in_track_threshold=0.95, invert=True)
        detail = (
            f"Denuncias policiales {latest_year}: {int(latest_value)} vs "
            f"{previous_year}: {int(previous_value)}."
        )
        return status, detail

    def _handle_violence_reports(self, district: str) -> Optional[StatusResult]:
        key = _strip_accents(district)
        subset = self.violence_lookup.get(key)
        if subset is None or subset.empty:
            return None
        subset = subset.copy()
        quantity_column = None
        for candidate in ("CANTIDAD", "Cantidad", "cantidad"):
            if candidate in subset.columns:
                quantity_column = candidate
                break
        if not quantity_column:
            return None
        subset[quantity_column] = pd.to_numeric(subset[quantity_column], errors="coerce")
        year_column = None
        for candidate in ("AÑO", "ANIO", "Año", "Anio"):
            if candidate in subset.columns:
                year_column = candidate
                break
        if not year_column:
            return None
        by_year = subset.groupby(year_column)[quantity_column].sum().dropna()
        if by_year.empty:
            return None
        sorted_years = sorted(by_year.index)
        if len(sorted_years) == 1:
            latest_year = sorted_years[-1]
            total = by_year.iloc[-1]
            status = IN_TRACK if total == 0 else NOT_ACHIEVED
            detail = f"Solo disponible {latest_year}; casos reportados {int(total)}."
            return status, detail
        latest_year = sorted_years[-1]
        previous_year = sorted_years[-2]
        latest_value = by_year.loc[latest_year]
        previous_value = by_year.loc[previous_year]
        ratio = latest_value / previous_value if previous_value else float("inf")
        status = _threshold_classification(ratio, achieved_threshold=0.8, in_track_threshold=0.95, invert=True)
        detail = (
            f"Violencia contra la mujer {latest_year}: {int(latest_value)} vs "
            f"{previous_year}: {int(previous_value)}."
        )
        return status, detail

    def _handle_tree_management(self, district: str) -> Optional[StatusResult]:
        key = _strip_accents(district)
        subset = self.tree_lookup.get(key)
        if subset is None or subset.empty:
            return None
        total_authorized = 0
        for col in subset.columns:
            if "CANTIDAD_DE_ARBOLES" in col:
                total_authorized += pd.to_numeric(subset[col], errors="coerce").fillna(0).sum()
        status = ACHIEVED if total_authorized >= 100 else IN_TRACK if total_authorized >= 20 else NOT_ACHIEVED
        detail = f"Gestión arborizada: {int(total_authorized)} intervenciones registradas."
        return status, detail

    def _handle_cultural_agenda(self, district: str) -> Optional[StatusResult]:
        key = _strip_accents(district)
        matches = sum(1 for location in getattr(self, "agenda_locations", []) if key in location)
        if matches == 0:
            return None
        status = ACHIEVED if matches >= 8 else IN_TRACK if matches >= 3 else NOT_ACHIEVED
        detail = f"Actividades culturales registradas en parques zonales: {matches}."
        return status, detail

    def _handle_control_services(self, district: str) -> Optional[StatusResult]:
        record = self._latest_record(self.control_lookup, district)
        if record is None:
            return None
        date_column = None
        for candidate in ("FECHA DE CONCLUSIÓN", "FECHA DE EMISIÓN", "FECHA DE PUBLICACIÓN"):
            if candidate in record.index:
                date_column = candidate
                break
        detail = f"Informe de control identificado ({record.get('NÚMERO DE INFORME', 's/d')})."
        status = ACHIEVED if date_column else IN_TRACK
        return status, detail

    def _handle_infobras(self, district: str) -> Optional[StatusResult]:
        key = _strip_accents(district)
        frame = self._get_infobras_frame(key)
        if frame is None or frame.empty:
            return None
        state_column = None
        for col in frame.columns:
            normalized = _normalize_column_label(col)
            if "estado" in normalized or "situacion" in normalized:
                state_column = col
                break
        if not state_column or state_column not in frame.columns:
            return None
        counts = frame[state_column].fillna("Sin estado").value_counts()
        total = counts.sum()
        if total == 0:
            return None
        finished = 0
        in_progress = 0
        for state, value in counts.items():
            norm_state = _strip_accents(str(state))
            if "finaliz" in norm_state:
                finished += value
            elif "ejec" in norm_state or "avance" in norm_state:
                in_progress += value
        completion_ratio = finished / total if total else 0.0
        progress_ratio = (finished + in_progress) / total if total else 0.0
        if completion_ratio >= 0.6:
            status = ACHIEVED
        elif progress_ratio >= 0.4:
            status = IN_TRACK
        else:
            status = NOT_ACHIEVED
        detail = (
            f"Obras finalizadas {finished}/{total}; en ejecución {in_progress}."
        )
        return status, detail

    def _get_infobras_frame(self, key: str) -> Optional[pd.DataFrame]:
        lookup = getattr(self.datasets, "infobras_lookup", {}) or {}
        if not lookup:
            return None
        variations = {
            key,
            key.replace(" ", ""),
        }
        alias = INFOBRAS_FILENAME_MAP.get(key)
        if alias:
            alias_key = _strip_accents(alias)
            variations.add(alias_key)
            variations.add(alias_key.replace(" ", ""))
        for variation in list(variations):
            if variation in lookup:
                return lookup[variation]
        for candidate_key, frame in lookup.items():
            compact_candidate = candidate_key.replace(" ", "")
            if compact_candidate in variations or any(
                variation in candidate_key or candidate_key in variation for variation in variations
            ):
                return frame
        return None


def evaluate_proposals(args: argparse.Namespace) -> None:
    if not PROPOSALS_PATH.exists():
        raise FileNotFoundError(f"No encontramos {PROPOSALS_PATH}")
    proposals = _load_csv(PROPOSALS_PATH)
    datasets = MunicipalDatasets.load()
    known_districts = sorted(proposals["distrito"].dropna().unique())
    evaluator = ProposalEvaluator(datasets, known_districts)

    records = []
    for _, row in proposals.iterrows():
        district = str(row.get("distrito", "")).strip()
        text = str(row.get("propuesta", "")).strip()
        status, detail = evaluator.evaluate(district, text)
        records.append(
            {
                "distrito": district,
                "propuesta": text,
                "timeline": row.get("timeline", ""),
                "factibilidad": row.get("factibilidad", ""),
                "observadores_de_exito": row.get("observadores_de_exito", ""),
                "status": status,
                "detalle": detail,
            }
        )

    output_df = pd.DataFrame.from_records(records)
    output_df.to_csv(OUTPUT_PATH, index=False, quoting=csv.QUOTE_NONNUMERIC)

    summary = output_df["status"].value_counts()
    print("Evaluación completada. Distribución de estados:")
    for status, count in summary.items():
        print(f"  - {status}: {count}")
    print(f"\nResultados guardados en {OUTPUT_PATH}")


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluar el cumplimiento de propuestas municipales.")
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    evaluate_proposals(args)


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1:])
