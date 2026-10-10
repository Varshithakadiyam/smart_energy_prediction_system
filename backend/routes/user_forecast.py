"""
user_forecast.py
FastAPI router for user-specific CSV uploads, data quality validation,
and personalized energy forecasting.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, status, Response, Body
from typing import Optional
import io
import pandas as pd

from backend.services.user_data_service import UserDataService
from backend.services.univariate_model_service import UnivariateModelService
from backend.services.model_service import ModelService
from backend.services.prediction_service import PredictionService
from backend.utils.preprocessing import (
    UserCsvValidationReport,
    UserForecastRequest,
    UserForecastResponse,
    UserForecastItem,
    UserAnomalyReport,
    RecommendationItem
)

router = APIRouter(prefix="/user-forecast", tags=["User Energy Forecast"])

SAMPLE_CSV_TEMPLATE = """timestamp,energy_consumption_kwh
2026-09-01 00:00:00,0.320
2026-09-01 01:00:00,0.295
2026-09-01 02:00:00,0.280
2026-09-01 03:00:00,0.290
2026-09-01 04:00:00,0.310
2026-09-01 05:00:00,0.360
2026-09-01 06:00:00,0.580
2026-09-01 07:00:00,1.150
2026-09-01 08:00:00,1.420
2026-09-01 09:00:00,1.210
2026-09-01 10:00:00,0.850
2026-09-01 11:00:00,0.760
2026-09-01 12:00:00,1.350
2026-09-01 13:00:00,0.920
2026-09-01 14:00:00,0.780
2026-09-01 15:00:00,0.840
2026-09-01 16:00:00,0.950
2026-09-01 17:00:00,1.650
2026-09-01 18:00:00,2.450
2026-09-01 19:00:00,2.890
2026-09-01 20:00:00,2.720
2026-09-01 21:00:00,1.980
2026-09-01 22:00:00,1.120
2026-09-01 23:00:00,0.540
2026-09-02 00:00:00,0.340
"""

@router.get("/template")
def download_csv_template():
    """
    Supplies the standard 2-column CSV template with documented hourly timestamp
    and energy_consumption_kwh headers.
    """
    return Response(
        content=SAMPLE_CSV_TEMPLATE.strip(),
        media_type="text/csv",
        headers={
            "Content-Disposition": "attachment; filename=energy_consumption_template.csv"
        }
    )

@router.post("/upload", response_model=UserCsvValidationReport)
async def upload_user_csv(file: UploadFile = File(...)):
    """
    Validates user electricity consumption CSV via multipart/form-data.
    Performs integrity checks, detects sampling gaps, computes personal analytics,
    and caches cleaned records in-memory for the current session.
    """
    filename = file.filename or "upload.csv"
    if not filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Invalid file format. Only .csv files are supported."
        )

    content = await file.read()
    if len(content) == 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="The uploaded file is empty."
        )

    try:
        report = UserDataService.validate_and_process_csv(content, filename=filename)
        return UserCsvValidationReport(**report)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"CSV processing error: {str(e)}"
        )

@router.post("/predict", response_model=UserForecastResponse)
def generate_user_forecast(request: UserForecastRequest):
    """
    Generates single-step (1h) or 24-hour autoregressive forecasts using
    the uploaded user's validated historical sequence.
    """
    session_data = UserDataService.get_session(request.session_id)
    if not session_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session expired or not found. Please re-upload your CSV file."
        )

    df_user, metadata = session_data
    horizon = request.horizon.lower() if request.horizon else "24h"
    if horizon not in ["1h", "24h"]:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Invalid forecast horizon. Supported horizons: '1h' or '24h'."
        )

    uni_svc = UnivariateModelService.get_instance()
    if not uni_svc.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Univariate forecasting model unavailable: {uni_svc.load_error}"
        )

    try:
        if horizon == "1h":
            pred_res = uni_svc.predict_next_hour(df_user)
            forecast_items = [
                UserForecastItem(
                    step=1,
                    timestamp=pred_res["timestamp"],
                    hour_label=pd.to_datetime(pred_res["timestamp"]).strftime("%H:00"),
                    predicted_consumption=pred_res["predicted_consumption"],
                    predicted_wh=pred_res["predicted_wh"],
                    unit="kWh",
                    level=pred_res["level"]
                )
            ]
            total_kwh = pred_res["predicted_consumption"]
            peak_hour = forecast_items[0].hour_label
            peak_val = pred_res["predicted_consumption"]
            lowest_hour = forecast_items[0].hour_label
            lowest_val = pred_res["predicted_consumption"]
            model_used = pred_res["model_used"]
        else: # 24h
            pred_res = uni_svc.predict_next_24_hours(df_user)
            forecast_items = [
                UserForecastItem(
                    step=p["step"],
                    timestamp=p["timestamp"],
                    hour_label=p["hour_label"],
                    predicted_consumption=p["predicted_consumption"],
                    predicted_wh=p["predicted_wh"],
                    unit="kWh",
                    level=p["level"]
                )
                for p in pred_res["predictions"]
            ]
            total_kwh = pred_res["total_predicted_kwh"]
            peak_hour = pred_res["peak_hour"]
            peak_val = pred_res["peak_consumption"]
            lowest_hour = pred_res["lowest_hour"]
            lowest_val = pred_res["lowest_consumption"]
            model_used = pred_res["model_used"]

        # Run User-Specific Anomaly Detection
        anom_res = UserDataService.detect_user_anomalies(df_user, recent_hours=24)
        anom_report = UserAnomalyReport(**anom_res)

        # Generate User-Specific Recommendations
        recs_raw = UserDataService.generate_user_recommendations(
            [p.model_dump() for p in forecast_items],
            metadata["analytics"]
        )
        recommendations = [RecommendationItem(**r) for r in recs_raw]

        return UserForecastResponse(
            session_id=request.session_id,
            model_used=model_used,
            horizon=horizon,
            predictions=forecast_items,
            total_predicted_kwh=total_kwh,
            peak_hour=peak_hour,
            peak_consumption=peak_val,
            lowest_hour=lowest_hour,
            lowest_consumption=lowest_val,
            anomalies=anom_report,
            recommendations=recommendations,
            personal_analytics=metadata["analytics"],
            diurnal_profile=metadata.get("diurnal_profile"),
            historical_timeline=metadata.get("historical_timeline")
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"User forecasting error: {str(e)}"
        )

@router.delete("/session/{session_id}")
def reset_user_session(session_id: str):
    """
    Cleans up in-memory session data for the user.
    """
    success = UserDataService.clear_session(session_id)
    return {"session_id": session_id, "cleared": success}
