from fastapi import APIRouter, HTTPException
from backend.services.model_service import ModelService
from backend.services.prediction_service import PredictionService
from backend.services.recommendation_service import RecommendationService
from backend.utils.preprocessing import RecommendationRequest, RecommendationResponse

router = APIRouter(tags=["Recommendations"])

@router.post("/recommendations", response_model=RecommendationResponse)
def get_energy_recommendations(request: RecommendationRequest = None):
    svc = ModelService.get_instance()
    
    if request and request.predictions and len(request.predictions) > 0:
        preds = request.predictions
    else:
        # Default: Generate next-24-hour forecast from reference data, then recommend
        sample_24 = svc.get_latest_sample(24)
        if sample_24 is None:
            raise HTTPException(status_code=503, detail="Historical dataset unavailable")
        pred_svc = PredictionService()
        preds = pred_svc.predict_next_24_hours(sample_24)
        
    try:
        recs = RecommendationService.generate_recommendations(preds)
        return RecommendationResponse(recommendations=recs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation engine error: {str(e)}")
