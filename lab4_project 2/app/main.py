from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.database.models import init_db
from app.endpoints import inference, training


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()          # створює inference_inputs та додає input_id у predictions
    yield


app = FastAPI(title="Star Classifier API",
              description="Тренування та інференс моделі класифікації зір (astro_vision.db)",
              version="1.0.0", lifespan=lifespan)
app.include_router(training.router, tags=["Training"])
app.include_router(inference.router, tags=["Inference"])


@app.get("/", tags=["Root"])
def read_root():
    return {"message": "Star Classifier API працює з базою astro_vision.db", "docs": "/docs"}
