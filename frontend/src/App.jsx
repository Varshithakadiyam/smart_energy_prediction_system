import React, { useState, useEffect, useCallback } from 'react';
import Header from './components/Header';
import ActiveDatasetBar from './components/ActiveDatasetBar';
import Navigation from './components/Navigation';
import KpiCards from './components/KpiCards';
import Forecast24hChart from './components/Forecast24hChart';
import ActualVsPredictedChart from './components/ActualVsPredictedChart';
import DiurnalPatternChart from './components/DiurnalPatternChart';
import AnomalyView from './components/AnomalyView';
import RecommendationsView from './components/RecommendationsView';
import ModelBenchmarksView from './components/ModelBenchmarksView';
import UserForecastView from './components/UserForecastView';
import { api } from './api';
import { AlertCircle, RefreshCw, Zap } from 'lucide-react';

export default function App() {
  const [activeTab, setActiveTab] = useState('overview');
  const [loading, setLoading] = useState(true);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [error, setError] = useState(null);

  // Global Active Dataset State: 'reference' vs 'user'
  const [datasetMode, setDatasetMode] = useState('reference');
  const [userValidationReport, setUserValidationReport] = useState(null);
  const [userForecastResult, setUserForecastResult] = useState(null);

  // Reference Benchmark API State
  const [health, setHealth] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [forecast24, setForecast24] = useState(null);
  const [historicalData, setHistoricalData] = useState(null);
  const [anomalies, setAnomalies] = useState(null);
  const [recommendations, setRecommendations] = useState(null);

  const fetchDashboardData = useCallback(async () => {
    try {
      setError(null);
      const [
        healthRes,
        infoRes,
        predRes,
        f24Res,
        histRes,
        anomRes,
        recRes
      ] = await Promise.all([
        api.getHealth().catch(e => ({ status: 'error', model_loaded: false, error: e.message })),
        api.getModelInfo().catch(() => null),
        api.predictNextHour().catch(() => null),
        api.predictNext24Hours().catch(() => null),
        api.getHistoricalData().catch(() => null),
        api.detectAnomalies().catch(() => null),
        api.getRecommendations().catch(() => null)
      ]);

      setHealth(healthRes);
      setModelInfo(infoRes);
      setPrediction(predRes);
      setForecast24(f24Res);
      setHistoricalData(histRes);
      setAnomalies(anomRes);
      setRecommendations(recRes);
    } catch (err) {
      console.error("Dashboard fetch error:", err);
      setError(err.message || "Failed to load dashboard data.");
    } finally {
      setLoading(false);
      setIsRefreshing(false);
    }
  }, []);

  useEffect(() => {
    fetchDashboardData();
  }, [fetchDashboardData]);

  const handleRefresh = () => {
    setIsRefreshing(true);
    fetchDashboardData();
  };

  // Callback when customer uploads and generates forecast
  const handleUserUploadSuccess = (report, forecast) => {
    setUserValidationReport(report);
    if (forecast) {
      setUserForecastResult(forecast);
    }
    setDatasetMode('user');
  };

  // Callback to reset customer session cleanly
  const handleResetCustomerSession = async () => {
    if (userValidationReport?.session_id) {
      try {
        await api.resetUserSession(userValidationReport.session_id);
      } catch (e) {
        // ignore
      }
    }
    setUserValidationReport(null);
    setUserForecastResult(null);
    setDatasetMode('reference');
  };

  const handleNavigateToUpload = () => {
    setActiveTab('user_forecast');
  };

  // Determine active data objects based on datasetMode
  const isCustomerActive = datasetMode === 'user' && Boolean(userForecastResult);

  const activePrediction = isCustomerActive ? {
    predicted_consumption: userForecastResult.predictions?.[0]?.predicted_consumption ?? 0,
    predicted_wh: userForecastResult.predictions?.[0]?.predicted_wh ?? 0,
    level: userForecastResult.predictions?.[0]?.level ?? 'Normal',
    unit: 'kWh'
  } : prediction;

  const activeForecast24 = isCustomerActive ? {
    horizon: userForecastResult.horizon,
    predictions: userForecastResult.predictions,
    total_predicted_kwh: userForecastResult.total_predicted_kwh,
    peak_hour: userForecastResult.peak_hour,
    peak_consumption: userForecastResult.peak_consumption,
    lowest_hour: userForecastResult.lowest_hour,
    lowest_consumption: userForecastResult.lowest_consumption,
    model_used: userForecastResult.model_used
  } : forecast24;

  const activeAnomalies = isCustomerActive ? userForecastResult.anomalies : anomalies;

  const activeRecommendations = isCustomerActive ? {
    recommendations: userForecastResult.recommendations
  } : recommendations;

  if (loading) {
    return (
      <div className="min-h-screen bg-slate-950 flex flex-col items-center justify-center p-4">
        <div className="relative">
          <div className="h-16 w-16 rounded-2xl bg-emerald-500/20 border border-emerald-500/40 flex items-center justify-center animate-pulse">
            <Zap className="h-8 w-8 text-emerald-400" />
          </div>
          <div className="absolute inset-0 rounded-2xl border-2 border-emerald-400 border-t-transparent animate-spin" />
        </div>
        <h2 className="text-white font-bold text-lg mt-6 tracking-tight">
          Connecting to Smart Energy Forecasting Engine...
        </h2>
        <p className="text-xs text-slate-400 font-mono mt-2">
          Initializing PyTorch/TensorFlow GRU model weights & test dataset
        </p>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-[#0b0f19] text-slate-100 flex flex-col">
      {/* Top Navbar */}
      <Header
        health={health}
        modelInfo={modelInfo}
        isRefreshing={isRefreshing}
        onRefresh={handleRefresh}
        datasetMode={datasetMode}
      />

      {/* Persistent Active Dataset Bar with Mandatory CSV Template Download */}
      <ActiveDatasetBar
        datasetMode={datasetMode}
        setDatasetMode={setDatasetMode}
        userValidationReport={userValidationReport}
        userForecastResult={userForecastResult}
        onResetCustomerSession={handleResetCustomerSession}
        onNavigateToUpload={handleNavigateToUpload}
      />

      {/* Main Tab Navigation */}
      <Navigation
        activeTab={activeTab}
        setActiveTab={setActiveTab}
      />

      {/* Content Container */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-4 lg:p-8 space-y-6">
        {/* Error Alert if API unreachable */}
        {(!health || health.status !== 'healthy') && (
          <div className="bg-rose-500/10 border border-rose-500/30 rounded-xl p-4 flex items-start gap-3">
            <AlertCircle className="h-5 w-5 text-rose-400 shrink-0 mt-0.5" />
            <div>
              <h3 className="text-sm font-bold text-rose-400">FastAPI Backend Connection Unavailable</h3>
              <p className="text-xs text-slate-300 mt-1">
                Unable to reach the backend at <code className="bg-slate-900 px-1 py-0.5 rounded text-rose-300">http://localhost:8000</code>. Ensure the server is running with:
              </p>
              <div className="mt-2 font-mono text-xs bg-slate-950 p-2 rounded border border-slate-800 text-slate-200">
                python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
              </div>
            </div>
          </div>
        )}

        {/* Tab 1: Overview Dashboard */}
        {activeTab === 'overview' && (
          <div className="space-y-6">
            <KpiCards
              prediction={activePrediction}
              forecast24={activeForecast24}
              anomalies={activeAnomalies}
              modelInfo={modelInfo}
              datasetMode={datasetMode}
              userValidationReport={userValidationReport}
            />

            {/* Main Forecast Visual */}
            {activeForecast24 && (
              <Forecast24hChart 
                forecast24={activeForecast24} 
                datasetMode={datasetMode}
                userValidationReport={userValidationReport}
              />
            )}

            {/* Diurnal and Day of Week Patterns */}
            <DiurnalPatternChart 
              historicalData={historicalData} 
              datasetMode={datasetMode}
              userValidationReport={userValidationReport}
              userForecastResult={userForecastResult}
            />
          </div>
        )}

        {/* Tab: User-Specific Energy Forecast from CSV Upload */}
        {activeTab === 'user_forecast' && (
          <UserForecastView 
            onUserUploadSuccess={handleUserUploadSuccess}
            onResetCustomerSession={handleResetCustomerSession}
            activeValidationReport={userValidationReport}
            activeForecastResult={userForecastResult}
            datasetMode={datasetMode}
            onSwitchTab={setActiveTab}
          />
        )}

        {/* Tab 2: 24-Hour Forecast */}
        {activeTab === 'forecast24' && (
          <div className="space-y-6">
            <Forecast24hChart 
              forecast24={activeForecast24} 
              datasetMode={datasetMode}
              userValidationReport={userValidationReport}
            />
          </div>
        )}

        {/* Tab 3: Actual vs Predicted & Diurnal */}
        {activeTab === 'historical' && (
          <div className="space-y-6">
            <ActualVsPredictedChart 
              historicalData={historicalData} 
              datasetMode={datasetMode}
              userValidationReport={userValidationReport}
              userForecastResult={userForecastResult}
            />
            <DiurnalPatternChart 
              historicalData={historicalData} 
              datasetMode={datasetMode}
              userValidationReport={userValidationReport}
              userForecastResult={userForecastResult}
            />
          </div>
        )}

        {/* Tab 4: Anomaly Detection */}
        {activeTab === 'anomalies' && (
          <div className="space-y-6">
            <AnomalyView 
              anomalies={activeAnomalies} 
              datasetMode={datasetMode}
              userValidationReport={userValidationReport}
            />
          </div>
        )}

        {/* Tab 5: Energy Recommendations */}
        {activeTab === 'recommendations' && (
          <div className="space-y-6">
            <RecommendationsView 
              recommendations={activeRecommendations} 
              datasetMode={datasetMode}
              userValidationReport={userValidationReport}
            />
          </div>
        )}

        {/* Tab 6: Model Benchmarks */}
        {activeTab === 'benchmarks' && (
          <div className="space-y-6">
            <ModelBenchmarksView modelInfo={modelInfo} />
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-800 bg-slate-900/60 py-4 px-4 text-center text-xs text-slate-500">
        <div className="max-w-7xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-2">
          <span>Smart Energy Consumption Prediction and Management System &bull; B.Tech Data Science</span>
          <span className="font-mono text-slate-400">
            {datasetMode === 'user' 
              ? `Personalized Mode (${userValidationReport?.filename || 'user_data.csv'}) • Univariate GRU` 
              : 'UCI Household Power Dataset • Multivariate GRU/LSTM Neural Network'}
          </span>
        </div>
      </footer>
    </div>
  );
}
