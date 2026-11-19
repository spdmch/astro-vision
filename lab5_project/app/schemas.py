from typing import Optional, Dict
from pydantic import BaseModel, Field, model_validator
from app.database.models import FEATURE_COLUMNS


class PredictRequest(BaseModel):
    """Або star_id (беремо ознаки з БД), або всі 16 ознак (у масштабованому/закодованому вигляді з ЛР2)."""
    star_id: Optional[str] = Field(None, examples=["STAR_00001"])
    true_label: Optional[int] = Field(None, description="Фактичний тип зорі (ground truth), якщо він уже відомий — потрібен для моніторингу якості (ЛР5)")
    Temperature_K: Optional[float] = None
    Luminosity_L_Sun: Optional[float] = None
    Radius_R_Sun: Optional[float] = None
    Mass_M_Sun: Optional[float] = None
    Distance_LY: Optional[float] = None
    Apparent_Magnitude: Optional[float] = None
    Absolute_Magnitude: Optional[float] = None
    Spectral_Class: Optional[float] = None
    Star_Color: Optional[float] = None
    Age_Million_Years: Optional[float] = None
    Metallicity: Optional[float] = None
    Rotation_Speed_KM_S: Optional[float] = None
    Constellation: Optional[float] = None
    Galaxy_Type: Optional[float] = None
    Variable_Star: Optional[float] = None
    Surface_Gravity: Optional[float] = None

    @model_validator(mode="after")
    def check_not_empty(self):
        provided = [c for c in FEATURE_COLUMNS if getattr(self, c) is not None]
        if self.star_id is None and not provided:
            raise ValueError("Порожній запит: передайте star_id або всі 16 ознак.")
        if self.star_id is None and len(provided) != len(FEATURE_COLUMNS):
            missing = [c for c in FEATURE_COLUMNS if getattr(self, c) is None]
            raise ValueError(f"Відсутні ознаки: {missing}")
        return self


class PredictResponse(BaseModel):
    prediction_id: int
    input_id: int
    predicted_type: int
    predicted_label: str
    source: str
    probability: float


class TrainingResponse(BaseModel):
    message: str
    n_train: int
    n_new_input: int
    train_accuracy: float
    new_input_accuracy: float
    new_input_f1_weighted: float
    model_path: str
    train_predictions_logged: int
