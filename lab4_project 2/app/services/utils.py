import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
META_PATH = BASE_DIR / "models" / "model_meta.json"


def save_meta(meta: dict) -> None:
    META_PATH.parent.mkdir(parents=True, exist_ok=True)
    META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def load_meta() -> dict:
    if not META_PATH.exists():
        raise FileNotFoundError("Модель ще не натренована: викличте POST /train-model")
    return json.loads(META_PATH.read_text(encoding="utf-8"))
