import sys
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Ensure macOS ARM compatibility workaround is applied first
import src.compat

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.services.model_service import ModelService
from backend.services.univariate_model_service import UnivariateModelService
from backend.routes import (
    health,
    prediction,
    analytics,
    anomaly,
    recommendations,
    model_info,
    user_forecast
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Safe one-time artifact loading at startup
    svc = ModelService.get_instance()
    uni_svc = UnivariateModelService.get_instance()
    if svc.is_loaded:
        print("Backend startup: Multivariate GRU model & scaler artifacts initialized.")
    if uni_svc.is_loaded:
        print("Backend startup: Univariate GRU model & scaler artifacts initialized.")
    yield
    print("Backend shutdown: Resources cleaned up.")

app = FastAPI(
    title="Smart Energy Consumption Prediction and Management System API",
    description="Modular REST API for LSTM/GRU short-term load forecasting, user CSV forecasting, anomaly detection, and data-driven recommendations.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for React frontend (Vite default: http://localhost:5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register route modules
app.include_router(health.router)
app.include_router(prediction.router)
app.include_router(analytics.router)
app.include_router(anomaly.router)
app.include_router(recommendations.router)
app.include_router(model_info.router)
app.include_router(user_forecast.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=False)
