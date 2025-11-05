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
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Tuple

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
    infobras_files: Dict[str, Path]

    @classmethod
    def load(cls) -> "MunicipalDatasets":
        budget_execution = _load_csv(DATOS_DIR / "ejecucion_del_gasto.csv")
        investment_execution = _load_csv(DATOS_DIR / "ejecucion_de_proyectos_de_inversion.csv")
        disability_spending = _load_csv(DATOS_DIR / "gastos_orientados_a_personas_con_discapacidad.csv")
        childhood_spending = _load_csv(DATOS_DIR / "recursos_para_gastos_en_ninez_y_adolescencia.csv")
        revenue_execution = _load_csv(DATOS_DIR / "presupuesto_y_ejecucion_de_ingresos.csv")
        police_reports = _load_csv(DATOS_DIR / "denuncias_policiales_2018_2025.csv")
        violence_reports = _load_csv(DATOS_DIR / "violencia_contra_la_mujer_2018_2025.csv")
        tree_management = _load_csv(
            DATOS_DIR / "gestion_del_arboleado_urbano.csv",
            sep=";",
            encoding="latin-1",
        )
        cultural_agenda = _load_csv(
            DATOS_DIR / "agendas_culturales_en_parques_zonales.csv",
            encoding="utf-8-sig",
        )
        control_services = _load_csv(
            DATOS_DIR / "servicios_de_control.csv",
            encoding="latin-1",
            engine="python",
            usecols=range(14),
            on_bad_lines="skip",
        )

        infobras_root = DATOS_DIR / "infobras"
        infobras_files: Dict[str, Path] = {}
        for file in infobras_root.glob("*.csv"):
            stem = file.stem.lower()
            infobras_files[stem] = file
            infobras_files[stem.replace(" ", "")] = file
            normalized = _strip_accents(stem)
            infobras_files.setdefault(normalized, file)
            infobras_files.setdefault(normalized.replace(" ", ""), file)

        return cls(
            budget_execution=budget_execution,
            investment_execution=investment_execution,
            disability_spending=disability_spending,
            childhood_spending=childhood_spending,
            revenue_execution=revenue_execution,
            police_reports=police_reports,
            violence_reports=violence_reports,
            tree_management=tree_management,
            cultural_agenda=cultural_agenda,
            control_services=control_services,
            infobras_files=infobras_files,
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
        path: Optional[Path] = None
        filename_hint = INFOBRAS_FILENAME_MAP.get(key)
        candidate_keys = []
        if filename_hint:
            normalized_hint = _strip_accents(filename_hint)
            candidate_keys.extend([
                filename_hint.lower(),
                filename_hint.replace(" ", "").lower(),
                normalized_hint,
                normalized_hint.replace(" ", ""),
            ])
        normalized_key = key.replace(" ", "")
        candidate_keys.extend([
            key,
            normalized_key,
            key.replace(" ", ""),
        ])
        for candidate in candidate_keys:
            candidate = candidate.lower()
            if candidate in self.datasets.infobras_files:
                path = self.datasets.infobras_files[candidate]
                break
        if path is None:
            for stem, candidate_path in self.datasets.infobras_files.items():
                stem_compact = stem.replace(" ", "")
                if normalized_key in stem_compact or stem_compact in normalized_key:
                    path = candidate_path
                    break
        if path is None:
            return None
        frame = _load_csv(path)
        if frame.empty or "Estado de la obra" not in [col.lower() for col in frame.columns]:
            # Make the column lookup accent/spacing insensitive.
            state_column = None
            for col in frame.columns:
                if _strip_accents(col) == "estado de obra" or "estado" in _strip_accents(col):
                    state_column = col
                    break
        else:
            state_column = "Estado de obra"
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
