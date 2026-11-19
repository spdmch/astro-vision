from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from app.services.monitoring import REPORT_TYPES, generate

router = APIRouter()


@router.get("/monitor", response_class=FileResponse)
def monitor(report: str = Query("drift", description="drift | target | performance | quality"),
            run: str = Query("all", description="all | normal | drift (прогін із logs/traffic_runs.json)")):
    """Будує Evidently-дашборд reference (train) vs current (inference) і повертає HTML."""
    if report not in REPORT_TYPES:
        raise HTTPException(422, f"report має бути одним із {list(REPORT_TYPES)}")
    try:
        path = generate(report, run)
    except ValueError as e:                 # немає inference-даних / невідомий прогін
        raise HTTPException(409, str(e))
    except RuntimeError as e:               # Evidently не встановлено
        raise HTTPException(500, str(e))
    return FileResponse(path, media_type="text/html")
