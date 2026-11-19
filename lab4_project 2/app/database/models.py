"""ORM-моделі таблиць, які створюються/використовуються API (ЛР4)."""
from sqlalchemy import Column, Integer, Float, String, text
from app.database.database import Base, engine

FEATURE_COLUMNS = [
    "Temperature_K", "Luminosity_L_Sun", "Radius_R_Sun", "Mass_M_Sun",
    "Distance_LY", "Apparent_Magnitude", "Absolute_Magnitude", "Spectral_Class",
    "Star_Color", "Age_Million_Years", "Metallicity", "Rotation_Speed_KM_S",
    "Constellation", "Galaxy_Type", "Variable_Star", "Surface_Gravity",
]


class Prediction(Base):
    """Таблиця predictions (структура збігається з тією, що створена в ЛР3)."""
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    Star_ID = Column(String, nullable=True)
    model_name = Column(String)
    true_label = Column(Integer, nullable=True)
    prediction = Column(Integer, nullable=False)
    source = Column(String, nullable=False)          # "train" | "inference"
    timestamp = Column(String)
    input_id = Column(Integer, nullable=True)        # зв'язок із inference_inputs.id


class InferenceInput(Base):
    """Усі вхідні параметри HTTP-запиту на передбачення."""
    __tablename__ = "inference_inputs"
    id = Column(Integer, primary_key=True, autoincrement=True)
    star_id = Column(String, nullable=True)
    Temperature_K = Column(Float)
    Luminosity_L_Sun = Column(Float)
    Radius_R_Sun = Column(Float)
    Mass_M_Sun = Column(Float)
    Distance_LY = Column(Float)
    Apparent_Magnitude = Column(Float)
    Absolute_Magnitude = Column(Float)
    Spectral_Class = Column(Float)
    Star_Color = Column(Float)
    Age_Million_Years = Column(Float)
    Metallicity = Column(Float)
    Rotation_Speed_KM_S = Column(Float)
    Constellation = Column(Float)
    Galaxy_Type = Column(Float)
    Variable_Star = Column(Float)
    Surface_Gravity = Column(Float)
    timestamp = Column(String)


def init_db():
    """Створює відсутні таблиці та додає колонку input_id у predictions."""
    Base.metadata.create_all(bind=engine)
    with engine.begin() as conn:
        cols = [r[1] for r in conn.execute(text("PRAGMA table_info(predictions)"))]
        if "input_id" not in cols:
            conn.execute(text("ALTER TABLE predictions ADD COLUMN input_id INTEGER"))
