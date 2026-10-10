from fastapi import APIRouter, HTTPException
import pandas as pd
from backend.services.model_service import ModelService
from backend.services.anomaly_service import AnomalyService
from backend.utils.preprocessing import AnomalyRequest, AnomalyResponse

router = APIRouter(tags=["Anomaly Detection"])

@router.post("/anomaly-detection", response_model=AnomalyResponse)
def detect_anomalies(request: AnomalyRequest = None):
    svc = ModelService.get_instance()
    anomaly_svc = AnomalyService()
    
    if request and request.data and len(request.data) > 0:
        obs = request.data
    else:
        # Default: Evaluate latest 24 hours
        sample = svc.get_latest_sample(24)
        if sample is None:
            raise HTTPException(status_code=503, detail="Historical dataset unavailable")
        obs = sample.reset_index().rename(columns={'index': 'timestamp'}).to_dict(orient='records')
        
    try:
        report = anomaly_svc.detect_anomalies(obs)
        return AnomalyResponse(**report)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Anomaly evaluation failed: {str(e)}")
