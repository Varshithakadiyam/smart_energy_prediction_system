# Smart Energy Consumption Prediction Using LSTM
## Comprehensive Data Inspection, Cleaning & Preprocessing Report

---

### 1. Executive Summary & Dataset Overview
- **Project Name:** Smart Energy Consumption Prediction Using LSTM
- **Raw Dataset Source:** `household_power_consumption_part_1.csv`
- **Initial Dataset Dimensions:** 500,000 rows $\times$ 9 columns
- **Cleaned Hourly Dataset Dimensions:** 8,164 rows $\times$ 42 columns
- **Sampling Frequency Transformation:** 1-minute interval $\longrightarrow$ 1-hour interval
- **Date Range:** `2006-12-16 17:24:00` to `2007-11-28 22:43:00`
- **Chronological Sequence:** Verified strictly monotonic increasing (no timestamp disorder)
- **Duplicate Observations:** 0 duplicate rows, 0 duplicate timestamps

---

### 2. Column Inventory & Data Schema

| Original Column | Initial Data Type | Cleaned Minute Type | Description |
| :--- | :--- | :--- | :--- |
| `Date` | `object` (`dd/mm/yyyy`) | Parsed to `timestamp` index | Calendar date of observation |
| `Time` | `object` (`hh:mm:ss`) | Parsed to `timestamp` index | 24-hour time of observation |
| `Global_active_power` | `object` | `float64` | Household global active power (in kW) |
| `Global_reactive_power` | `object` | `float64` | Household global reactive power (in kVAR) |
| `Voltage` | `object` | `float64` | Minute-averaged voltage (in Volts) |
| `Global_intensity` | `object` | `float64` | Minute-averaged current intensity (in Amperes) |
| `Sub_metering_1` | `object` | `float64` | Energy sub-metering No. 1 (kitchen: dishwasher, microwave) in Wh |
| `Sub_metering_2` | `object` | `float64` | Energy sub-metering No. 2 (laundry room: washing machine, dryer, fridge) in Wh |
| `Sub_metering_3` | `float64` | `float64` | Energy sub-metering No. 3 (climate control: electric water heater & AC) in Wh |

---

### 3. Missing Value Audit & Imputation Strategy

The raw dataset encodes missing observations as `?` in columns 3–8 and as `NaN` in column 9.

| Column | Missing Indicators Before Cleaning | Missing % Before | Missing Values After Cleaning | Imputation Methodology |
| :--- | :---: | :---: | :---: | :--- |
| `Date` | 0 | 0.000% | 0 | N/A (Complete) |
| `Time` | 0 | 0.000% | 0 | N/A (Complete) |
| `Global_active_power` | 3,933 | 0.787% | **0** | Time-based Linear Interpolation |
| `Global_reactive_power` | 3,933 | 0.787% | **0** | Time-based Linear Interpolation |
| `Voltage` | 3,933 | 0.787% | **0** | Time-based Linear Interpolation |
| `Global_intensity` | 3,933 | 0.787% | **0** | Time-based Linear Interpolation |
| `Sub_metering_1` | 3,933 | 0.787% | **0** | Time-based Linear Interpolation |
| `Sub_metering_2` | 3,933 | 0.787% | **0** | Time-based Linear Interpolation |
| `Sub_metering_3` | 3,933 | 0.787% | **0** | Time-based Linear Interpolation |

#### Justification for Time-Based Linear Interpolation:
1. Power consumption exhibits strong temporal continuity; discarding rows would introduce temporal discontinuity and corrupt subsequent lag/rolling operations.
2. The missing data is organized into 23 discrete continuous chunks. The longest chunk spans ~2.5 days (April 28 to April 30, 2007), while the remaining 22 chunks are small transient sensor dropouts (1 to 83 minutes).
3. Boundary fallback (`bfill` and `ffill`) was provided to guarantee 0 missing values across all channels.

---

### 4. Data Validation, Range Audits & Outlier Analysis

| Feature | Min Observed | Median | Mean | Max Observed | Negatives Count | Physical Validity Assessment |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `Global_active_power` (kW) | 0.082 | 0.482 | 1.104 | 10.670 | 0 | Valid (Represents high-draw appliance loads) |
| `Global_reactive_power` (kVAR) | 0.000 | 0.100 | 0.119 | 1.148 | 0 | Valid (Typical inductive loads) |
| `Voltage` (V) | 223.490 | 239.630 | 239.241 | 251.700 | 0 | Valid (Standard 230V $\pm 10\%$ grid variance) |
| `Global_intensity` (A) | 0.400 | 2.200 | 4.718 | 46.400 | 0 | Valid ($I \approx P / V$, matches 10 kW spikes) |
| `Sub_metering_1` (Wh/min) | 0.000 | 0.000 | 1.197 | 78.000 | 0 | Valid (Intermittent kitchen loads) |
| `Sub_metering_2` (Wh/min) | 0.000 | 0.000 | 1.637 | 78.000 | 0 | Valid (Intermittent laundry heating cycles) |
| `Sub_metering_3` (Wh/min) | 0.000 | 0.000 | 5.646 | 20.000 | 0 | Valid (Water heater / AC duty cycles) |

**Outlier Treatment Rationale:**
No legitimate high-power spikes were clipped or discarded. High active power readings up to ~10 kW correspond to realistic concurrent appliance usage (e.g., cooking + water heating + laundry). Clipping these would suppress the true variance the LSTM needs to learn for peak demand forecasting.

---

### 5. Minute-to-Hour Aggregation & Target Formulation

- **Minute Records:** 500,000 raw minutes.
- **Hourly Records:** 8,332 complete 60-minute hours (boundary partial hours: 36 min at start, 44 min at end were trimmed to ensure exact 60-minute equivalence for all hourly rows).
- **Primary Forecast Target:** `average_global_active_power` (kW) and `energy_consumption` (Wh).
- **Units & Mathematical Relations:**
  $$\text{Hourly Average Power (kW)} = \frac{1}{60} \sum_{m=1}^{60} P_m$$
  $$\text{Energy Consumption (Wh)} = \sum_{m=1}^{60} \left(P_m \times 1000 \times \frac{1}{60}\right) = \text{Hourly Average Power (kW)} \times 1000$$
  $$\text{Energy Consumption (kWh)} = \text{Hourly Average Power (kW)} \times 1.0\,\text{hour}$$
  $$\text{Sub-metering Total (Wh)} = \sum_{m=1}^{60} \text{Sub\_metering\_X}_m$$
  $$\text{Sub-metering Remainder (Wh)} = \max(0,\, \text{Energy Consumption (Wh)} - [\text{Sub\_1} + \text{Sub\_2} + \text{Sub\_3}])$$

---

### 6. Feature Engineering Matrix

Total Engineered Features: **42**

1. **Temporal & Calendar Features:**
   - `hour` (0–23), `day` (1–31), `day_of_week` (0–6), `day_of_month` (1–31), `month` (1–12), `week_of_year` (1–52), `is_weekend` (0/1).
2. **Behavioral Indicators:**
   - `is_morning` (06:00–11:59), `is_evening` (17:00–21:59), `is_peak_hour` (18:00–21:00).
3. **Cyclical Sine/Cosine Encodings:**
   - `sin_hour`, `cos_hour` ($2\pi \times \text{hour} / 24$)
   - `sin_day_of_week`, `cos_day_of_week` ($2\pi \times \text{day\_of\_week} / 7$)
   - `sin_month`, `cos_month` ($2\pi \times (\text{month} - 1) / 12$)
4. **Historical Lag Features (Past Only):**
   - `lag_1`, `lag_2`, `lag_3`, `lag_6`, `lag_12`, `lag_24`, `lag_48`, `lag_168`.
   - `lag_24`: Power consumption 24 hours prior (daily recurrence).
   - `lag_168`: Power consumption 168 hours prior (7 days $\times$ 24 hours, weekly recurrence).
5. **Leak-Free Rolling Statistics:**
   - `rolling_mean_6h`, `rolling_mean_12h`, `rolling_mean_24h`, `rolling_std_24h`, `rolling_mean_168h`.
   - Computed on `.shift(1)` to ensure the rolling window at step $t$ utilizes strictly observations up to $t-1$.
6. **Warm-up Drop:** The first 168 hourly rows were removed to eliminate NaNs introduced by the 168-hour lookback, yielding 8,164 continuous, fully populated hourly samples.

---

### 7. Chronological Data Partitioning & Leakage Prevention

| Partition | Row Count | Percentage | Start Timestamp | End Timestamp |
| :--- | :---: | :---: | :--- | :--- |
| **Training** | 5,714 | 70.0% | `2006-12-23 18:00:00` | `2007-08-18 19:00:00` |
| **Validation** | 1,225 | 15.0% | `2007-08-18 20:00:00` | `2007-10-08 20:00:00` |
| **Testing** | 1,225 | 15.0% | `2007-10-08 21:00:00` | `2007-11-28 21:00:00` |

**Leakage Prevention Protocol:**
- No random shuffling was applied; temporal sequence is strictly maintained ($T_{\text{train}} < T_{\text{val}} < T_{\text{test}}$).
- `MinMaxScaler(feature_range=(0, 1))` was fitted **exclusively on the Training partition**.
- Validation and Testing datasets were transformed using the frozen Training scaler parameters (min and max).
- Rolling features and lag features strictly exclude the current or future target observations.

---

### 8. LSTM Sequence Tensor Formulation

- **Input Sequence Length ($L$):** 24 hours ($[t-23, t-22, \dots, t]$)
- **Forecast Horizon ($H$):** 1 hour ($t+1$) (extensible to 24-hour horizon)
- **Feature Dimension ($D$):** 42 features

| Tensor Split | Input Tensor ($X$) Shape | Target Tensor ($y$) Shape | Description |
| :--- | :---: | :---: | :--- |
| **Train Tensors** | `(5690, 24, 42)` | `(5690, 1)` | 5,690 historical 24-hour sliding windows |
| **Validation Tensors** | `(1201, 24, 42)` | `(1201, 1)` | 1,201 historical 24-hour sliding windows |
| **Test Tensors** | `(1201, 24, 42)` | `(1201, 1)` | 1,201 historical 24-hour sliding windows |

---

### 9. Output Artifact Directory Structure

```
├── data/
│   └── processed/
│       ├── hourly_energy_consumption.csv   # Complete 8,164 hourly dataset with all features
│       ├── train.csv                       # Unscaled training split with timestamps
│       ├── validation.csv                  # Unscaled validation split with timestamps
│       ├── test.csv                        # Unscaled test split with timestamps
│       ├── lstm_sequences.npz              # Scaled 3D tensors: X_train, y_train, X_val, y_val, X_test, y_test
│       └── data_cleaning_report.md         # This comprehensive cleaning audit
├── models/
│   └── scaler.pkl                          # Serialized MinMaxScaler bundle (feature & target scalers)
└── figures/
    ├── 01_energy_consumption_over_time.png
    ├── 02_daily_average_consumption.png
    ├── 03_weekly_consumption.png
    ├── 04_monthly_consumption.png
    ├── 05_hourly_consumption_profile.png
    ├── 06_day_of_week_consumption.png
    ├── 07_heatmap_hour_vs_dayofweek.png
    ├── 08_distribution_energy_consumption.png
    ├── 09_missing_value_analysis.png
    └── 10_correlation_heatmap.png
```

---

### 10. EDA Findings & Energy Consumption Insights

1. **Diurnal Seasonality:**
   - Base consumption occurs between 02:00 and 05:00 (~0.45 kW) representing background vampire loads (refrigerator, standby electronics).
   - Morning peak occurs between 07:00 and 09:00 (~1.4 kW) as occupants wake and prepare for work/school.
   - Significant evening peak occurs between 18:00 and 21:30 (~2.3 to 2.8 kW) when cooking, lighting, heating/AC, and home entertainment are concurrently active.
2. **Weekly Seasonality:**
   - Weekend days (Saturday and Sunday) exhibit substantially higher midday consumption (~1.3–1.6 kW) compared to weekdays (~0.9–1.1 kW), due to daytime occupancy.
3. **Macro-Annual Trends:**
   - Winter months (Dec–Feb) exhibit the highest total load due to electric space heating and early sunsets requiring lighting.
   - Summer months (Jul–Aug) exhibit the lowest energy consumption, reflecting mild Parisian weather and occupant vacation periods.
4. **Correlation Insights:**
   - `Global_intensity` has a near-perfect linear correlation with `Global_active_power` ($r = 0.999$).
   - `lag_1` has the strongest single-step autocorrelation ($r = 0.81$).
   - `lag_24` retains substantial 24-hour diurnal correlation ($r = 0.58$).
   - `rolling_mean_24h` exhibits high correlation ($r = 0.67$), providing strong smoothed baseline context to the LSTM.

---

### 11. Assumptions & Modeling Recommendations

1. **Grid Standard:** Voltage and current align with standard 230V residential European standards.
2. **Missing Chunk Continuity:** The 2.5-day sensor dropout in April 2007 was linearly interpolated smoothly across time; if desired in downstream ablation studies, sequence generation across that boundary can be masked or analyzed.
3. **Model Ready Status:**
   - All input tensors are bounded in $[0, 1]$.
   - Zero missing values remain.
   - Tensors are saved in high-performance `.npz` binary format ready for immediate loading via `np.load('data/processed/lstm_sequences.npz')`.
