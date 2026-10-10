"""
preprocessing.py
Pydantic schemas and data validation models for FastAPI request and response payloads.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class HealthResponse(BaseModel):
    status: str = "healthy"
    model_loaded: bool = True
    model_type: Optional[str] = "GRU"

class EnergyDataPoint(BaseModel):
    timestamp: str
    energy_consumption: Optional[float] = None
    actual_consumption: Optional[float] = None
    consumption: Optional[float] = None
    hour: Optional[int] = None
    day_of_week: Optional[int] = None
    is_weekend: Optional[int] = None
    sin_hour: Optional[float] = None
    cos_hour: Optional[float] = None
    sin_day_of_week: Optional[float] = None
    cos_day_of_week: Optional[float] = None
    lag_1: Optional[float] = None
    lag_24: Optional[float] = None
    lag_168: Optional[float] = None
    rolling_mean_24h: Optional[float] = None
    rolling_std_24h: Optional[float] = None
    average_voltage: Optional[float] = None
    average_global_reactive_power: Optional[float] = None
    average_global_intensity: Optional[float] = None
    sub_metering_1_wh: Optional[float] = None
    sub_metering_2_wh: Optional[float] = None
    sub_metering_3_wh: Optional[float] = None

class PredictionRequest(BaseModel):
    data: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="List of at least 24 hourly observations. If omitted, uses latest 24 hours from reference dataset."
    )

class PredictionResponse(BaseModel):
    predicted_consumption: float
    unit: str = "kWh"
    level: str
    timestamp: str
    predicted_wh: Optional[float] = None
    reference_timestamp: Optional[str] = None

class HourlyForecastItem(BaseModel):
    step: int
    timestamp: str
    hour_label: str
    predicted_consumption: float
    predicted_wh: float
    unit: str = "kWh"
    level: str

class Prediction24Response(BaseModel):
    predictions: List[HourlyForecastItem]
    total_predicted_kwh: float
    peak_hour: str
    peak_consumption: float
    lowest_hour: str
    lowest_consumption: float

class AnalyticsRequest(BaseModel):
    data: Optional[List[Dict[str, Any]]] = None
    range_type: Optional[str] = "all" # '24h', '7d', 'all'

class AnalyticsResponse(BaseModel):
    total_consumption: float
    average_consumption: float
    min_consumption: float
    max_consumption: float
    peak_hour: str
    lowest_consumption_hour: str
    daily_average: float
    weekly_average: float
    unit: str = "kWh"

class AnomalyRequest(BaseModel):
    data: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="List of observations with 'timestamp' and 'actual_consumption' or 'energy_consumption'."
    )

class AnomalyItem(BaseModel):
    timestamp: str
    actual_consumption: float
    expected_consumption: float
    deviation: float
    z_score: float
    is_anomaly: bool
    anomaly_status: str

class AnomalyResponse(BaseModel):
    total_evaluated: int
    anomalies_detected: int
    has_anomalies: bool
    summary: str
    results: List[AnomalyItem]

class RecommendationRequest(BaseModel):
    predictions: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="List of forecast items with 'timestamp' and 'predicted_consumption'."
    )
    current_hour: Optional[int] = None

class RecommendationItem(BaseModel):
    category: str
    priority: str
    title: str
    recommendation: str
    explanation: Optional[str] = None
    trigger_pattern: Optional[str] = None
    action: Optional[str] = None
    estimated_savings: Optional[str] = None

class RecommendationResponse(BaseModel):
    recommendations: List[RecommendationItem]

class BaselineMetricItem(BaseModel):
    model: str
    mae_wh: float
    rmse_wh: float
    mape_pct: float
    r2: float

class ModelMetrics(BaseModel):
    MAE_Wh: float
    RMSE_Wh: float
    MAPE_pct: float
    R2_score: float
    MAE_kWh: float
    RMSE_kWh: float

class ModelInfoResponse(BaseModel):
    model_name: str
    model_type: str
    input_window: str
    prediction_horizon: str
    features_used: List[str]
    total_features: int
    loss_function: str
    optimizer: str
    metrics: ModelMetrics
    baseline_comparison: List[BaselineMetricItem]

class UserPersonalAnalytics(BaseModel):
    total_consumption_kwh: float
    average_hourly_kwh: float
    min_hourly_kwh: float
    max_hourly_kwh: float
    peak_historical_hour: str
    lowest_historical_hour: str
    daily_average_kwh: float
    total_days_coverage: float

class UserCsvPreviewItem(BaseModel):
    timestamp: str
    hour_label: str
    energy_consumption_kwh: float

class UserCsvValidationReport(BaseModel):
    session_id: str
    filename: str
    is_valid: bool
    total_rows: int
    accepted_rows: int
    start_timestamp: str
    end_timestamp: str
    sampling_frequency: str
    duplicate_count: int
    missing_count: int
    imputed_count: int
    compatible_model: str
    preview_data: List[UserCsvPreviewItem]
    recent_history: Optional[List[UserCsvPreviewItem]] = None
    diurnal_profile: Optional[List[Dict[str, Any]]] = None
    historical_timeline: Optional[List[Dict[str, Any]]] = None
    personal_analytics: UserPersonalAnalytics

class UserForecastRequest(BaseModel):
    session_id: str
    horizon: str = Field(default="24h", description="'1h' or '24h'")

class UserForecastItem(BaseModel):
    step: int
    timestamp: str
    hour_label: str
    predicted_consumption: float
    predicted_wh: float
    unit: str = "kWh"
    level: str

class UserAnomalyItem(BaseModel):
    timestamp: str
    hour_label: str
    actual_consumption_kwh: float
    expected_kwh: float
    deviation_kwh: float
    z_score: float
    is_anomaly: bool
    status: str
    severity: Optional[str] = "Normal"
    explanation: Optional[str] = None

class UserAnomalyReport(BaseModel):
    total_evaluated: int
    anomalies_detected: int
    has_sufficient_history: bool
    method: str
    results: List[UserAnomalyItem]

class UserForecastResponse(BaseModel):
    session_id: str
    model_used: str
    horizon: str
    predictions: List[UserForecastItem]
    total_predicted_kwh: float
    peak_hour: str
    peak_consumption: float
    lowest_hour: str
    lowest_consumption: float
    anomalies: UserAnomalyReport
    recommendations: List[RecommendationItem]
    personal_analytics: UserPersonalAnalytics
    diurnal_profile: Optional[List[Dict[str, Any]]] = None
    historical_timeline: Optional[List[Dict[str, Any]]] = None
