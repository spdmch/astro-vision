import logging
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sqlalchemy import text

from app.database.database import engine
from app.database.models import FEATURE_COLUMNS
from app.services.utils import save_meta, load_meta

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = BASE_DIR / "models" / "best_star_classifier_rf.pkl"
IMPORTANCE_PATH = BASE_DIR / "models" / "feature_importances.csv"
LOG_PATH = BASE_DIR / "logs" / "training.log"
FEATURE_TABLE = "stars_data_cleaned"
MODEL_NAME = "RandomForest_API"
# Найкращі гіперпараметри з ЛР3 (GridSearchCV) + class_weight для дисбалансу класів
BEST_PARAMS = dict(n_estimators=50, max_depth=20, min_samples_split=2,
                   class_weight="balanced", random_state=42, n_jobs=-1)

LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(filename=LOG_PATH, level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s", encoding="utf-8")
log = logging.getLogger("training")


def _label_map() -> dict:
    """Кодування Star_Type (LabelEncoder із ЛР2) відновлюємо зі зв'язку 1:1 stars <-> stars_data_cleaned."""
    q = f'SELECT DISTINCT c."Star_Type" AS code, s."Star_Type" AS name ' \
        f'FROM {FEATURE_TABLE} c JOIN stars s ON s."Star_ID" = c."Star_ID" ORDER BY code'
    df = pd.read_sql(q, engine)
    return {int(r.code): str(r.name) for r in df.itertuples()}


def train_model() -> dict:
    # 1. читаємо всі записи з таблиці ознак
    df = pd.read_sql(f"SELECT * FROM {FEATURE_TABLE}", engine)
    log.info("Прочитано %d записів з %s", len(df), FEATURE_TABLE)

    # 2. ділимо 90:10 на train та new_input (стратифіковано за цільовою змінною)
    train_df, new_input_df = train_test_split(
        df, test_size=0.10, random_state=42, stratify=df["Star_Type"])
    X_train, y_train = train_df[FEATURE_COLUMNS], train_df["Star_Type"].astype(int)
    X_new, y_new = new_input_df[FEATURE_COLUMNS], new_input_df["Star_Type"].astype(int)

    # 3. тренуємо модель
    clf = RandomForestClassifier(**BEST_PARAMS).fit(X_train, y_train)

    # 4. оцінюємо на train та на відкладених 10 % (new_input)
    train_acc = float(accuracy_score(y_train, clf.predict(X_train)))
    new_pred = clf.predict(X_new)
    new_acc = float(accuracy_score(y_new, new_pred))
    new_f1 = float(f1_score(y_new, new_pred, average="weighted"))
    log.info("train acc=%.4f | new_input acc=%.4f f1_w=%.4f", train_acc, new_acc, new_f1)

    # 5. зберігаємо модель, важливість ознак і метадані (порядок ознак, словник класів)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    pd.DataFrame({"Feature": FEATURE_COLUMNS, "Importance": clf.feature_importances_}) \
        .sort_values("Importance", ascending=False).to_csv(IMPORTANCE_PATH, index=False)
    save_meta({"features": FEATURE_COLUMNS, "labels": _label_map(), "model_name": MODEL_NAME,
               "params": {k: v for k, v in BEST_PARAMS.items() if k != "n_jobs"},
               "trained_at": datetime.now().isoformat(timespec="seconds")})

    # 6. зберігаємо вибірку new_input (джерело даних для /predict з star_id)
    new_input_df.to_sql("new_input", engine, if_exists="replace", index=False)

    # 7. передбачення на train -> predictions (source="train"); попередній запуск цієї моделі перезаписуємо
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = pd.DataFrame({
        "Star_ID": train_df["Star_ID"].values, "model_name": MODEL_NAME,
        "true_label": y_train.values, "prediction": clf.predict(X_train),
        "source": "train", "timestamp": now})
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM predictions WHERE source='train' AND model_name=:m"), {"m": MODEL_NAME})
    rows.to_sql("predictions", engine, if_exists="append", index=False)
    log.info("У predictions записано %d рядків (source=train)", len(rows))

    return dict(n_train=len(train_df), n_new_input=len(new_input_df), train_accuracy=round(train_acc, 4),
                new_input_accuracy=round(new_acc, 4), new_input_f1_weighted=round(new_f1, 4),
                model_path=str(MODEL_PATH), train_predictions_logged=len(rows))


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Модель ще не натренована: викличте POST /train-model")
    return joblib.load(MODEL_PATH)


def predict(features: pd.DataFrame):
    meta = load_meta()
    model = load_model()
    X = features[meta["features"]]          # той самий набір і порядок ознак, що й під час навчання
    proba = model.predict_proba(X)[0]
    code = int(model.classes_[proba.argmax()])
    return code, meta["labels"].get(str(code), f"Unknown ({code})"), float(proba.max()), meta["model_name"]
