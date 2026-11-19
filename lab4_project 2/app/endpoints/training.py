from fastapi import APIRouter, HTTPException
from app.schemas import TrainingResponse
from app.services.model import train_model

router = APIRouter()


@router.post("/train-model", response_model=TrainingResponse)
def train_model_endpoint():
    """Читає stars_data_cleaned, ділить 90:10, тренує RandomForest, зберігає модель у models/
    та записує передбачення на train у predictions (source='train')."""
    try:
        return TrainingResponse(message="Модель натреновано та збережено.", **train_model())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Помилка під час навчання: {e}")
