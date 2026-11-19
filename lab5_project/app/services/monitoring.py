import asyncio
import json
import sqlite3
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DB_PATH = BASE_DIR / "astro_vision.db"
REPORTS_DIR = BASE_DIR / "reports"
RUNS_PATH = BASE_DIR / "logs" / "traffic_runs.json"
MODEL_NAME = "RandomForest_API"

# Ознаки з ЛР4. Galaxy_Type має єдине значення (Spiral) -> дрейф для константи не визначений, тому виключено.
CATEGORICAL = ["Spectral_Class", "Star_Color", "Constellation", "Variable_Star"]
NUMERICAL = ["Temperature_K", "Luminosity_L_Sun", "Radius_R_Sun", "Mass_M_Sun", "Distance_LY",
             "Apparent_Magnitude", "Absolute_Magnitude", "Age_Million_Years", "Metallicity",
             "Rotation_Speed_KM_S", "Surface_Gravity"]
FEATURES = NUMERICAL + CATEGORICAL
REPORT_TYPES = ("drift", "target", "performance", "quality")


def _query(sql: str, params: tuple = ()) -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(sql, conn, params=params)


def _finalize(df: pd.DataFrame) -> pd.DataFrame:
    """Однакова структура для reference і current: типи, назви target/prediction, порядок стовпців."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    for c in NUMERICAL:
        df[c] = df[c].astype(float)
    for c in CATEGORICAL:
        df[c] = df[c].astype(int)
    df["prediction"] = df["prediction"].astype(int)
    df["target"] = df["target"].astype("Int64")   # у current можуть бути пропуски (запити без ground truth)
    return df[["timestamp"] + FEATURES + ["target", "prediction"]].reset_index(drop=True)


def load_reference() -> pd.DataFrame:
    """Reference = записи, на яких навчалась модель (predictions.source='train') разом з ознаками зі stars_data_cleaned."""
    cols = ", ".join(f'c."{f}"' for f in FEATURES)
    sql = (f'SELECT c."timestamp" AS timestamp, {cols}, c."Star_Type" AS target, p."prediction" AS prediction '
           f'FROM predictions p JOIN stars_data_cleaned c ON c."Star_ID" = p."Star_ID" '
           f"WHERE p.source = 'train' AND p.model_name = ?")
    return _finalize(_query(sql, (MODEL_NAME,)))


def load_runs() -> dict:
    return json.loads(RUNS_PATH.read_text(encoding="utf-8")) if RUNS_PATH.exists() else {}


def load_current(run: str = "all") -> pd.DataFrame:
    """Current = запити до /predict (inference_inputs + predictions.source='inference').
    target = true_label (відомий лише для запитів за star_id), для решти — NA.
    run: 'all' або назва прогону з logs/traffic_runs.json ('normal', 'drift') — відбір за часом запиту."""
    cols = ", ".join(f'i."{f}"' for f in FEATURES)
    sql = (f'SELECT i."timestamp" AS timestamp, {cols}, p."true_label" AS target, p."prediction" AS prediction '
           f"FROM inference_inputs i JOIN predictions p ON p.input_id = i.id "
           f"WHERE p.source = 'inference' AND p.model_name = ?")
    params: list = [MODEL_NAME]
    if run != "all":
        runs = load_runs()
        if run not in runs:
            raise ValueError(f"Прогін '{run}' не знайдено в {RUNS_PATH.name}: спершу запустіть src/simulate_traffic.py")
        sql += " AND i.\"timestamp\" BETWEEN ? AND ?"
        params += [runs[run]["start"], runs[run]["end"]]
    df = _query(sql, tuple(params))
    if df.empty:
        raise ValueError("Немає inference-даних: надішліть запити на /predict (src/simulate_traffic.py)")
    return _finalize(df)


def drift_summary(ref: pd.DataFrame, cur: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """KS-тест (числові) та хі-квадрат (категоріальні). Перехресна перевірка до звіту Evidently + числа для висновку."""
    rows = []
    for f in NUMERICAL:
        stat, p = ks_2samp(ref[f], cur[f])
        rows.append((f, "num", "KS", stat, p, ref[f].mean(), cur[f].mean()))
    for f in CATEGORICAL:
        table = pd.concat([ref[f].value_counts(), cur[f].value_counts()], axis=1).fillna(0)
        chi, p, *_ = chi2_contingency(table.values)
        rows.append((f, "cat", "chi2", chi, p, np.nan, np.nan))
    out = pd.DataFrame(rows, columns=["feature", "type", "test", "statistic", "p_value", "ref_mean", "cur_mean"])
    out["drift"] = out["p_value"] < alpha
    return out.sort_values("p_value").reset_index(drop=True)


def class_shares(ref: pd.DataFrame, cur: pd.DataFrame) -> pd.DataFrame:
    """Частка кожного класу: reference.target, current.target (де відомий), current.prediction."""
    return pd.DataFrame({
        "reference_target": ref["target"].value_counts(normalize=True),
        "current_target": cur["target"].dropna().value_counts(normalize=True),
        "current_prediction": cur["prediction"].value_counts(normalize=True),
    }).fillna(0).sort_index().round(4)


def performance_summary(ref: pd.DataFrame, cur: pd.DataFrame) -> pd.DataFrame:
    from sklearn.metrics import accuracy_score, f1_score
    rows = []
    for name, df in (("reference (train)", ref), ("current (з ground truth)", cur.dropna(subset=["target"]))):
        if df.empty:
            rows.append((name, 0, np.nan, np.nan)); continue
        y, p = df["target"].astype(int), df["prediction"]
        rows.append((name, len(df), accuracy_score(y, p), f1_score(y, p, average="macro")))
    return pd.DataFrame(rows, columns=["набір", "n", "accuracy", "f1_macro"]).round(4)


def _evidently_new(kind: str, ref: pd.DataFrame, cur: pd.DataFrame, path: Path) -> None:
    """Evidently >= 0.7."""
    from evidently import DataDefinition, Dataset, MulticlassClassification, Report
    from evidently.presets import ClassificationPreset, DataDriftPreset, DataSummaryPreset

    ref, cur = ref.copy(), cur.copy()
    if kind in ("target", "performance"):
        cur = cur.dropna(subset=["target"])
        ref["target"], cur["target"] = ref["target"].astype(int), cur["target"].astype(int)
    else:
        ref, cur = ref.drop(columns="target"), cur.drop(columns="target")

    if kind == "performance":
        dd = DataDefinition(numerical_columns=NUMERICAL, categorical_columns=CATEGORICAL,
                            classification=[MulticlassClassification(target="target", prediction_labels="prediction")])
        report = Report([ClassificationPreset()])
    elif kind == "target":
        dd = DataDefinition(numerical_columns=NUMERICAL, categorical_columns=CATEGORICAL + ["target", "prediction"])
        report = Report([DataDriftPreset(columns=["target", "prediction"])])
    elif kind == "drift":
        ref, cur = ref.drop(columns="prediction"), cur.drop(columns="prediction")
        dd = DataDefinition(numerical_columns=NUMERICAL, categorical_columns=CATEGORICAL)
        report = Report([DataDriftPreset()])
    else:  # quality
        ref, cur = ref.drop(columns=["prediction"]), cur.drop(columns=["prediction"])
        dd = DataDefinition(numerical_columns=NUMERICAL, categorical_columns=CATEGORICAL)
        report = Report([DataSummaryPreset()])
    snapshot = report.run(Dataset.from_pandas(cur, data_definition=dd),
                          Dataset.from_pandas(ref, data_definition=dd))   # run(current, reference)
    snapshot.save_html(str(path))


def _evidently_legacy(kind: str, ref: pd.DataFrame, cur: pd.DataFrame, path: Path) -> None:
    """Evidently 0.4.x."""
    from evidently import ColumnMapping
    from evidently.metric_preset import (ClassificationPreset, DataDriftPreset, DataQualityPreset,
                                         TargetDriftPreset)
    from evidently.report import Report

    ref, cur = ref.copy(), cur.copy()
    with_truth = kind in ("target", "performance")
    if with_truth:                                   # ground truth є лише в частині current-запитів
        cur = cur.dropna(subset=["target"])
        ref["target"], cur["target"] = ref["target"].astype(int), cur["target"].astype(int)
    else:
        ref, cur = ref.drop(columns=["target", "prediction"]), cur.drop(columns=["target", "prediction"])
    cm = ColumnMapping(target="target" if with_truth else None,
                       prediction="prediction" if with_truth else None,
                       datetime="timestamp", numerical_features=NUMERICAL, categorical_features=CATEGORICAL)
    preset = {"drift": DataDriftPreset(), "target": TargetDriftPreset(),
              "performance": ClassificationPreset(), "quality": DataQualityPreset()}[kind]
    report = Report(metrics=[preset])
    report.run(reference_data=ref, current_data=cur, column_mapping=cm)
    report.save_html(str(path))


def build_report(kind: str, ref: pd.DataFrame, cur: pd.DataFrame, out_path: Optional[Path] = None) -> Path:
    """Будує один Evidently HTML-звіт ('drift' | 'target' | 'performance' | 'quality') і повертає шлях до файлу."""
    if kind not in REPORT_TYPES:
        raise ValueError(f"kind має бути одним із {REPORT_TYPES}")
    try:
        import evidently  # noqa: F401
    except ImportError as e:
        raise RuntimeError("Evidently не встановлено: pip install evidently") from e
    out_path = Path(out_path) if out_path else REPORTS_DIR / f"{kind}_report.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from evidently import Dataset  # noqa: F401  (є лише в >= 0.7)
        _evidently_new(kind, ref, cur, out_path)
    except ImportError:
        _evidently_legacy(kind, ref, cur, out_path)
    return out_path


def generate(kind: str = "drift", run: str = "all") -> Path:
    """Повний цикл для /monitor: дані з БД -> Evidently -> HTML у reports/<kind>_<run>.html."""
    # FastAPI виконує sync-ендпоінти в окремому потоці, де немає asyncio event loop, а Evidently 0.4.x його потребує
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())
    ref, cur = load_reference(), load_current(run)
    return build_report(kind, ref, cur, REPORTS_DIR / f"{kind}_{run}.html")