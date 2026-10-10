from fastapi import APIRouter
from backend.services.model_service import ModelService
from backend.utils.preprocessing import HealthResponse

router = APIRouter(tags=["Health"])

@router.get("/health", response_model=HealthResponse)
def health_check():
    svc = ModelService.get_instance()
    return HealthResponse(
        status="healthy",
        model_loaded=svc.is_loaded and svc.model is not None,
        model_type=svc.get_model_metadata()["model_type"]
    )
