from datetime import datetime
from typing import List

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.database.database import get_db
from app.database.models import FEATURE_COLUMNS, InferenceInput, Prediction
from app.schemas import PredictRequest, PredictResponse
from app.services.model import predict

router = APIRouter()


@router.post("/predict", response_model=PredictResponse)
def run_inference(req: PredictRequest, db: Session = Depends(get_db)):
    """Передбачає тип зорі за JSON з 16 ознаками або за star_id (ознаки беруться з БД)."""
    true_label = None
    if req.star_id and any(getattr(req, c) is None for c in FEATURE_COLUMNS):
        row = db.execute(text('SELECT * FROM stars_data_cleaned WHERE "Star_ID"=:i'),
                         {"i": req.star_id}).mappings().first()
        if row is None:
            raise HTTPException(404, f"Зорю {req.star_id} не знайдено в stars_data_cleaned")
        values = {c: float(row[c]) for c in FEATURE_COLUMNS}
        true_label = int(row["Star_Type"])
    else:
        values = {c: float(getattr(req, c)) for c in FEATURE_COLUMNS}

    try:
        code, label, prob, model_name = predict(pd.DataFrame([values]))
    except FileNotFoundError as e:
        raise HTTPException(409, str(e))

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    inp = InferenceInput(star_id=req.star_id, timestamp=now, **values)
    db.add(inp); db.flush()
    pred = Prediction(Star_ID=req.star_id, model_name=model_name, true_label=true_label,
                      prediction=code, source="inference", timestamp=now, input_id=inp.id)
    db.add(pred); db.commit()
    return PredictResponse(prediction_id=pred.id, input_id=inp.id, predicted_type=code,
                           predicted_label=label, source="inference", probability=round(prob, 4))


@router.get("/predictions")
def history(limit: int = 20, source: str = "inference", db: Session = Depends(get_db)):
    rows = db.query(Prediction).filter(Prediction.source == source).order_by(Prediction.id.desc()).limit(limit).all()
    return [{"id": r.id, "Star_ID": r.Star_ID, "model_name": r.model_name, "true_label": r.true_label,
             "prediction": r.prediction, "source": r.source, "timestamp": r.timestamp, "input_id": r.input_id}
            for r in rows]
