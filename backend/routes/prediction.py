from fastapi import APIRouter, HTTPException, status
import pandas as pd
from backend.services.model_service import ModelService
from backend.services.prediction_service import PredictionService
from backend.utils.preprocessing import (
    PredictionRequest,
    PredictionResponse,
    Prediction24Response
)

router = APIRouter(tags=["Prediction"])

def parse_input_dataframe(data_list):
    svc = ModelService.get_instance()
    if not svc.is_loaded or svc.model is None or svc.feature_scaler is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model service is not loaded: {svc.load_error or 'Artifacts unavailable'}"
        )
    if data_list is None or len(data_list) == 0:
        # Default to latest 24 hours from reference dataset
        df = svc.get_latest_sample(24)
        if df is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Reference dataset not available and no data provided in request."
            )
        return df
        
    if len(data_list) < 24:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"At least 24 hourly time steps required. Received {len(data_list)}."
        )
        
    df = pd.DataFrame(data_list)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
        
    # Check for missing feature columns and fill defaults if necessary
    for col in svc.feature_cols:
        if col not in df.columns:
            # If energy_consumption is provided in kWh, convert to Wh
            if col == 'energy_consumption' and 'consumption' in df.columns:
                df['energy_consumption'] = df['consumption'] * 1000.0
            elif col in ['hour', 'day_of_week', 'is_weekend']:
                if isinstance(df.index, pd.DatetimeIndex):
                    if col == 'hour': df['hour'] = df.index.hour
                    elif col == 'day_of_week': df['day_of_week'] = df.index.dayofweek
                    elif col == 'is_weekend': df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
            else:
                # Fallback to zero
                df[col] = 0.0
                
    return df

@router.post("/predict", response_model=PredictionResponse)
def predict_next_hour(request: PredictionRequest = None):
    try:
        data_list = request.data if request else None
        df = parse_input_dataframe(data_list)
        pred_svc = PredictionService()
        result = pred_svc.predict_next_hour(df)
        return PredictionResponse(**result)
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Prediction error: {str(e)}")

@router.post("/predict-next-24-hours", response_model=Prediction24Response)
def predict_next_24_hours(request: PredictionRequest = None):
    try:
        data_list = request.data if request else None
        df = parse_input_dataframe(data_list)
        pred_svc = PredictionService()
        predictions = pred_svc.predict_next_24_hours(df)
        
        total_kwh = round(sum(p['predicted_consumption'] for p in predictions), 2)
        peak_item = max(predictions, key=lambda x: x['predicted_consumption'])
        lowest_item = min(predictions, key=lambda x: x['predicted_consumption'])
        
        return Prediction24Response(
            predictions=predictions,
            total_predicted_kwh=total_kwh,
            peak_hour=peak_item['hour_label'],
            peak_consumption=peak_item['predicted_consumption'],
            lowest_hour=lowest_item['hour_label'],
            lowest_consumption=lowest_item['predicted_consumption']
        )
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Multi-step forecast error: {str(e)}")
