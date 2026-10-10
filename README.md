# Smart Energy Consumption Prediction and Management System Using LSTM & GRU

An end-to-end Machine Learning and Full-Stack Engineering system for short-term load forecasting (STLF), statistical diurnal anomaly detection, and demand-side energy optimization.

---

## 1. Project Overview & Objectives

Accurate short-term load forecasting is essential for modern smart grid stability, peak-demand surcharge reduction, dynamic time-of-use (TOU) tariff optimization, and battery storage management. In individual residential households, electrical load exhibits deterministic diurnal and weekly seasonality alongside volatile, non-linear surges caused by appliance usage.

This project delivers:
1. **Recurrent Deep Learning Models:** Multi-layer LSTM and GRU architectures trained to predict next-hour electrical demand ($t+1$) from past 24-hour observation sequences ($t-23 \dots t$).
2. **Dedicated User-Uploaded Data Forecasting (Strategy B):** Standalone univariate GRU forecasting engine allowing users to upload arbitrary electricity consumption CSVs (`timestamp,energy_consumption_kwh`) with zero invented features and zero scaler refitting.
3. **Strict Chronological Data Integrity:** Zero future information leakage, with MinMax normalization frozen strictly to the training splits.
4. **Modular FastAPI Backend:** High-performance REST API with single-step inference, 24-hour recursive rolling forecasting, statistical diurnal anomaly detection, data-driven recommendations, and in-memory privacy-preserving session caching.
5. **Interactive React/Vite Dashboard:** Real-time energy analytics interface featuring KPI cards, forecast charts, actual vs. predicted timelines, anomaly indicators, model benchmarking, and personalized CSV upload forecasting with preview and historical timeline visuals.
6. **macOS ARM Compatibility Bridge:** Integrated runtime workaround resolving PyArrow/TensorFlow mutex lock conflicts on Apple Silicon.

---

## 2. System Architecture

```
[ UCI Household Benchmark Dataset ]               [ User-Uploaded Electricity CSV ]
              │                                                 │
              ▼                                                 ▼
    60-Minute Resampling & Aggregation               Temporal & Numeric Validation
              │                                      (Non-negative, Gaps <= 3h, 1h freq)
              ▼                                                 │
    19 Multivariate Features                                    ▼
              │                                      In-Memory Privacy Session Cache
              ▼                                      (Zero disk writes, 2h auto-TTL)
    Chronological Split (70% / 15% / 15%)                       │
              │                                                 ▼
        ┌─────┴─────┐                                Frozen Univariate MinMaxScaler
        ▼           ▼                                           │
    LSTM Model  GRU Model (Champion: R²=0.65)                   ▼
        └─────┬─────┘                                Univariate GRU Predictor (Strategy B)
              │                                      (Next-Hour & 24h Rolling Forecasts)
              │                                                 │
              └────────────────────────┬────────────────────────┘
                                       │
                                       ▼
                       FastAPI Modular Backend (Port 8000)
                       ├─ GET    /health
                       ├─ GET    /model-info
                       ├─ POST   /predict
                       ├─ POST   /predict-next-24-hours
                       ├─ POST   /analytics
                       ├─ GET    /analytics/historical
                       ├─ POST   /anomaly-detection
                       ├─ POST   /recommendations
                       ├─ GET    /user-forecast/template
                       ├─ POST   /user-forecast/upload
                       ├─ POST   /user-forecast/predict
                       └─ DELETE /user-forecast/session/{session_id}
                                       │
                                       ▼
                       React + Vite Frontend (Port 5173)
                       ├─ Overview Dashboard & KPI Cards
                       ├─ My Energy Forecast (Upload, Validation, Historical Chart & Forecast)
                       ├─ 24-Hour Day-Ahead Forecast
                       ├─ Actual vs Predicted Timeline & Diurnal Heatmaps
                       ├─ Diurnal Anomaly Detection Table & Controls
                       ├─ Personalized Energy Recommendations
                       └─ Model Benchmarks & Metrics
```

---

## 3. Dataset & Preprocessing Pipeline

- **Source:** Cleaned hourly dataset derived from the UCI Individual Household Electric Power Consumption benchmark.
- **Observations:** 8,164 hourly records across 42 engineered channels (reduced to 19 optimal predictors for tensor generation).
- **Date Range:** `2006-12-23 18:00:00` to `2007-11-28 21:00:00`.
- **Target Formulations & Units:**
  $$\text{energy\_consumption (Wh)} = \text{average\_global\_active\_power (kW)} \times 1000$$
  $$\text{energy\_consumption\_kwh (kWh)} = \frac{\text{energy\_consumption (Wh)}}{1000}$$
- **Data Partitions:**
  - **Train (70%):** 5,714 hourly observations ($5,690$ sequences of shape `(24, 19)`)
  - **Validation (15%):** 1,225 hourly observations ($1,201$ sequences of shape `(24, 19)`)
  - **Test (15%):** 1,225 hourly observations ($1,201$ unseen sequences)

### Selected Predictor Channels ($D = 19$):
1. `energy_consumption` (Wh target and autoregressive history)
2. `hour` (0–23)
3. `day_of_week` (0–6)
4. `is_weekend` (0/1 binary indicator)
5. `sin_hour`, `cos_hour` ($2\pi h / 24$ trigonometric encoding)
6. `sin_day_of_week`, `cos_day_of_week` ($2\pi \text{dow} / 7$)
7. `lag_1` (Immediate persistence from previous hour)
8. `lag_24` (Same-hour consumption yesterday)
9. `lag_168` (Same-hour consumption same day last week)
10. `rolling_mean_24h` (Smoothed 24-hour baseline demand)
11. `rolling_std_24h` (24-hour demand volatility and appliance switching)
12. `average_voltage` (Grid voltage in Volts)
13. `average_global_reactive_power` (Inductive loads)
14. `average_global_intensity` (Circuit current intensity in Amperes)
15. `sub_metering_1_wh` (Kitchen zone: dishwasher, microwave)
16. `sub_metering_2_wh` (Laundry zone: washing machine, dryer, fridge)
17. `sub_metering_3_wh` (Climate control: water heater & AC)

---

## 4. Deep Learning Model Architectures

| Architecture Parameter | LSTM Baseline | GRU Comparative (Champion) |
| :--- | :--- | :--- |
| **Layer 1** | `LSTM(units=64, return_sequences=True)` | `GRU(units=64, return_sequences=True)` |
| **Regularization 1** | `Dropout(0.2)` | `Dropout(0.2)` |
| **Layer 2** | `LSTM(units=32, return_sequences=False)` | `GRU(units=32, return_sequences=False)` |
| **Regularization 2** | `Dropout(0.2)` | `Dropout(0.2)` |
| **Dense Intermediate**| `Dense(units=16, activation='relu')` | `Dense(units=16, activation='relu')` |
| **Output Layer** | `Dense(units=1, activation='linear')` | `Dense(units=1, activation='linear')` |
| **Total Parameters** | **34,465** | **26,273** (~24% more efficient) |
| **Loss & Optimizer** | MSE Loss, Adam Optimizer ($\alpha=0.001$) | MSE Loss, Adam Optimizer ($\alpha=0.001$) |

---

## 5. Model Evaluation & Baseline Comparison

All metrics were computed strictly on the **unseen test partition** ($1,201$ sequences) in original unscaled energy units:

| Model Category | Model Name | MAE (Wh) | RMSE (Wh) | MAPE (%) | $R^2$ Score | MAE (kWh) | RMSE (kWh) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Statistical Baseline** | Previous Hour ($y_t$) | 496.30 | 766.10 | 48.11% | 0.4411 | 0.496 | 0.766 |
| **Statistical Baseline** | Previous Day ($y_{t-23}$) | 732.42 | 1086.26 | 92.06% | -0.1236 | 0.732 | 1.086 |
| **Statistical Baseline** | Moving Average (24h) | 785.88 | 1013.10 | 125.25% | 0.0227 | 0.786 | 1.013 |
| **Deep Learning** | **LSTM (Baseline)** | 407.03 | 614.46 | 44.96% | 0.6405 | 0.407 | 0.614 |
| **Deep Learning** | **GRU (Selected Best)** | **406.86** | **607.26** | **44.80%** | **0.6489** | **0.407** | **0.607** |

**Key Findings:**
1. Deep learning achieves an **18.0% reduction in MAE** and **20.7% reduction in RMSE** over the best persistence baseline.
2. GRU attained superior generalizability over LSTM with 24% fewer trainable weights.
3. The model achieves an $R^2$ score of **0.6489** on individual residential load (a notoriously noisy STLF domain).

---

## 6. Statistical Diurnal Anomaly Detection

To prevent false alarms arising from legitimate differences between daytime standby (~0.3 kW) and evening cooking peaks (~2.5 kW), anomaly detection dynamically references hour-of-day distributions:

$$\text{Threshold}(h) = \mu_h \pm 2.5 \times \sigma_h$$
$$Z = \frac{x_t - \mu_{\text{hour}(t)}}{\sigma_{\text{hour}(t)}}$$

- Observations with $|Z| > 2.5$ are flagged as anomalous.
- Missing values in input feeds trigger graceful imputation using time-windowed median profiles.

---

## 7. macOS ARM Compatibility Workaround

On Apple Silicon (M1/M2/M3/M4) running Python 3.13, TensorFlow/Keras and Scikit-learn/PyArrow can trigger a C-level mutex lock collision (`[mutex.cc : 452] RAW: Lock blocking`).

The module `src/compat.py` resolves this issue by routing `KERAS_HOME` and `MPLCONFIGDIR` to writable locations and providing a lightweight dummy module for PyArrow version checks before TensorFlow initializes. **Always keep `import src.compat` as the first import in any backend or CLI execution entrypoint.**

---

## 8. User-Uploaded Energy Data Forecasting Pipeline (Strategy B)

To allow real-world residential users to upload their own smart meter data without requiring unavailable electrical channels (e.g., sub-metered kitchen/laundry circuits, grid reactive power, line voltage), the system implements a **dedicated Univariate GRU forecasting pipeline**:

### CSV Schema Requirements
- **Required Columns:** `timestamp,energy_consumption_kwh`
  - Flexible header aliases supported: `timestamp`, `datetime`, `time` and `energy_consumption_kwh`, `consumption_kwh`, `kwh`, `consumption`.
- **Sampling Frequency:** Regular hourly intervals ($\Delta t \approx 3600\,\text{s}$).
- **Minimum Sequence:** At least 24 consecutive hourly observations.
- **Maximum File Size:** 10 MB.

### Validation & Cleaning Engine (`backend/services/user_data_service.py`)
1. **Physical Bounds:** Rejects negative readings and extreme spikes ($> 50.0\,\text{kWh/h}$).
2. **Auto Unit Scaling:** Detects raw Watt-hour input (mean $> 50$) and scales automatically to kWh ($\div 1000$).
3. **Temporal Gap Interpolation:** Automatically identifies missing hourly slots. Permitted gaps of contiguous length $\le 3\,\text{hours}$ and $\le 15\%$ missing volume are imputed via time-weighted linear interpolation. Datasets exceeding these thresholds are rejected with clear actionable error messages.
4. **Duplicate & Timezone Handling:** Deduplicates identical timestamps and normalizes timezones.

### Dedicated Univariate Model Architecture & Frozen Normalization
- **Model Artifact:** `models/univariate_energy_model.keras`
- **Scaler Artifact:** `models/univariate_scaler.pkl` (MinMaxScaler fitted strictly on the 70% training split of `energy_consumption_kwh`).
- **Layers:** `Input(shape=(24, 1))` $\rightarrow$ `GRU(32, return_sequences=True)` $\rightarrow$ `Dropout(0.2)` $\rightarrow$ `GRU(16)` $\rightarrow$ `Dropout(0.2)` $\rightarrow$ `Dense(8, activation='relu')` $\rightarrow$ `Dense(1, activation='linear')`.
- **Zero Future Leakage:** The scaler is **never refitted** on uploaded data; uploaded sequences are transformed strictly using the frozen training bounds.

### Privacy & In-Memory Session Management
- **Zero Disk Persistence:** Uploaded files and parsed DataFrames are never written to permanent disk storage.
- **Sliding In-Memory Session Cache:** Stored in memory with UUID session identifiers and a 2-hour TTL (`SESSION_TTL_SECONDS = 7200`). Sessions can be explicitly purged via `DELETE /user-forecast/session/{session_id}`.

---

## 9. Installation & Setup

### Prerequisites
- Python 3.10+ (tested on Python 3.13.2 on macOS ARM)
- Node.js v18+ and npm v9+

### Backend Setup
```bash
# Navigate to project directory
cd "smart energy prediction system"

# Verify Python dependencies
pip install -r backend/requirements.txt
```

### Frontend Setup
```bash
# Navigate to frontend directory
cd frontend

# Install npm packages
npm install
```

---

## 10. Running the Application

### 1. Start the FastAPI Backend
```bash
# From the project root directory
python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```
- API Base URL: `http://localhost:8000`
- Interactive Swagger Documentation: `http://localhost:8000/docs`
- ReDoc Documentation: `http://localhost:8000/redoc`

### 2. Start the React Frontend
```bash
# In a separate terminal, from the frontend directory
cd frontend
npm run dev
```
- Dashboard URL: `http://localhost:5173`

---

## 11. API Specification Summary

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Service health status and loaded model confirmation |
| `GET` | `/model-info` | Architectural metadata, feature names, and benchmark metrics |
| `POST` | `/predict` | Predicts next-hour consumption ($t+1$) in kWh and Wh (Reference dataset) |
| `POST` | `/predict-next-24-hours` | Generates 24-hour autoregressive roll-forward forecast (Reference dataset) |
| `POST` | `/analytics` | Summary statistics (total, average, min, max, peak hour) |
| `GET` | `/analytics/historical` | 24h & 7d actual vs. predicted sequences and diurnal curves |
| `POST` | `/anomaly-detection` | Hourly z-score evaluation against diurnal baselines |
| `POST` | `/recommendations` | Data-driven load-shifting and standby conservation guidance |
| `GET` | `/user-forecast/template` | Downloads the standard 2-column CSV template with sample data |
| `POST` | `/user-forecast/upload` | Validates uploaded CSV, extracts analytics, preview records & recent history |
| `POST` | `/user-forecast/predict` | Generates 1h or 24h forecasts, anomaly alerts & personalized advice for user CSV |
| `DELETE` | `/user-forecast/session/{session_id}` | Immediately invalidates and frees in-memory session dataset |

---

## 12. Testing & Verification

### Running the Complete Test Suite
Run the 22 automated test suites covering all reference endpoints, user CSV validation, interpolation, forecasting, error states, and regression checks:
```bash
python3 -m unittest tests/test_api.py tests/test_user_forecast.py
```
Or run full test discovery:
```bash
python3 -m unittest discover tests
```

### Running CLI Inference
Verify standalone model inference against the reference test dataset:
```bash
python3 predict.py
```

### Building Frontend for Production
```bash
cd frontend
npm run build
```

---

## 13. Known Limitations & Future Work

1. **Unpredictable Extreme Spikes:** Instantaneous demand spikes above 6 kW (e.g., simultaneous water heater + oven + dryer usage) cannot be anticipated without occupancy sensors.
2. **Exogenous Meteorological Data:** Ambient outdoor temperature and solar irradiance were not present in the UCI dataset; incorporating ERA5 weather features is recommended for future space conditioning modeling.
3. **Multi-Horizon Architectures:** Extending from autoregressive rolling inference to simultaneous multi-horizon Seq2Seq or Temporal Fusion Transformer (TFT) architectures.
4. **Autoregressive Univariate Horizon Ceiling (24h vs. 7d):** Autoregressively rolling single-step univariate load forecasts beyond 24 hours (e.g., 168 hours / 7 days) without exogenous weather and calendar signals experiences variance dampening and compounding error. The system deliberately bounds user forecasting to Next-Hour ($1\text{h}$) and Day-Ahead ($24\text{h}$) to guarantee reliable load trajectories.

