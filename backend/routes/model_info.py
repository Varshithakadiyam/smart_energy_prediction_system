from fastapi import APIRouter
from backend.services.model_service import ModelService
from backend.utils.preprocessing import ModelInfoResponse

router = APIRouter(tags=["Model Info"])

@router.get("/model-info", response_model=ModelInfoResponse)
def get_model_info():
    svc = ModelService.get_instance()
    info = svc.get_model_metadata()
    return ModelInfoResponse(**info)
