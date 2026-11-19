"""Імітація потоку запитів до POST /predict (ЛР5): формує current-дані для моніторингу.

Режими:
  normal — 1000 відкладених записів new_input надсилаються за star_id (ground truth відомий; розподіл як у train);
  drift  — ЗМОДЕЛЬОВАНИЙ зсув: ті самі зорі з навмисно зміщеними ознаками надсилаються повним JSON
           разом із true_label (імітація того, що фактичний тип зорі став відомий пізніше). Мітки не змінюються, тому
           падіння accuracy у цьому режимі — наслідок зсуву ознак.

Перед запуском: uvicorn app.main:app --reload  і  POST /train-model.
Запуск:  python src/simulate_traffic.py normal
         python src/simulate_traffic.py drift --n 500
Час початку/кінця кожного прогону пишеться в logs/traffic_runs.json — за ним monitoring відділяє normal від drift.
"""
import argparse
import json
import sqlite3
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "astro_vision.db"
RUNS_PATH = BASE_DIR / "logs" / "traffic_runs.json"
FEATURES = ["Temperature_K", "Luminosity_L_Sun", "Radius_R_Sun", "Mass_M_Sun", "Distance_LY",
            "Apparent_Magnitude", "Absolute_Magnitude", "Spectral_Class", "Star_Color", "Age_Million_Years",
            "Metallicity", "Rotation_Speed_KM_S", "Constellation", "Galaxy_Type", "Variable_Star",
            "Surface_Gravity"]

# Параметри синтетичного дрейфу (ознаки в БД — z-оцінки з ЛР2, тому зсуви задано в одиницях σ)
DRIFT_SHIFTS = {"Temperature_K": +1.5, "Luminosity_L_Sun": +1.0, "Absolute_Magnitude": -0.8, "Metallicity": -1.0}
DRIFT_SPECTRAL_CODE, DRIFT_SPECTRAL_SHARE = 1, 0.5      # 50 % запитів отримують спектральний клас з кодом 1
DRIFT_VARIABLE_SHARE = 0.6                              # 60 % запитів стають змінними зорями


def post(url: str, payload: dict) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def load_new_input() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query("SELECT * FROM new_input", conn)


def make_drifted(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = df.copy()
    for col, shift in DRIFT_SHIFTS.items():
        out[col] = out[col] + shift
    out.loc[rng.random(len(out)) < DRIFT_SPECTRAL_SHARE, "Spectral_Class"] = DRIFT_SPECTRAL_CODE
    out.loc[rng.random(len(out)) < DRIFT_VARIABLE_SHARE, "Variable_Star"] = 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["normal", "drift"])
    ap.add_argument("--n", type=int, default=None, help="кількість запитів (normal: 1000, drift: 500 за замовчуванням)")
    ap.add_argument("--url", default="http://127.0.0.1:8000/predict")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    pool = load_new_input()
    n = min(args.n or (1000 if args.mode == "normal" else 500), len(pool))
    sample = pool.sample(n=n, random_state=args.seed).reset_index(drop=True)
    if args.mode == "drift":
        sample = make_drifted(sample, args.seed)

    time.sleep(1.1)                                      # щоб вікна normal/drift не перетиналися (timestamp з точністю до секунди)
    start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ok = fail = 0
    for _, row in sample.iterrows():
        payload = {"star_id": row["Star_ID"]} if args.mode == "normal" else \
                  {**{f: float(row[f]) for f in FEATURES}, "true_label": int(row["Star_Type"])}
        try:
            post(args.url, payload); ok += 1
        except (urllib.error.URLError, TimeoutError) as e:
            fail += 1
            if fail == 1:
                print("Помилка запиту:", e)
    end = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    time.sleep(1.1)

    runs = json.loads(RUNS_PATH.read_text(encoding="utf-8")) if RUNS_PATH.exists() else {}
    runs[args.mode] = {"start": start, "end": end, "n_ok": ok, "n_failed": fail, "seed": args.seed}
    RUNS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RUNS_PATH.write_text(json.dumps(runs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{args.mode}: успішно {ok}, помилок {fail}, вікно {start} — {end}")


if __name__ == "__main__":
    main()
