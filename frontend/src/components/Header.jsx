import React from 'react';
import { Zap, Activity, RefreshCw, CheckCircle2, AlertCircle, Cpu } from 'lucide-react';

export default function Header({ health, modelInfo, isRefreshing, onRefresh, datasetMode }) {
  const isHealthy = health?.status === 'healthy' && health?.model_loaded;
  const isCustomerMode = datasetMode === 'user';
  const modelName = isCustomerMode 
    ? 'Univariate GRU (Customer)' 
    : (modelInfo?.model_type || (health?.model_type ? `${health.model_type} Model` : 'Deep Learning Model'));

  return (
    <header className="border-b border-slate-800 bg-slate-900/80 backdrop-blur sticky top-0 z-50 px-4 lg:px-8 py-3.5">
      <div className="max-w-7xl mx-auto flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        {/* Title & Brand */}
        <div className="flex items-center gap-3">
          <div className="h-10 w-10 rounded-xl bg-gradient-to-br from-emerald-500 to-teal-700 flex items-center justify-center shadow-lg shadow-emerald-500/20 text-white font-bold">
            <Zap className="h-5 w-5 fill-current" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-lg font-bold text-white tracking-tight">Smart Energy Forecast</h1>
              <span className="text-xs px-2 py-0.5 rounded-full font-mono bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                v1.0.0
              </span>
            </div>
            <p className="text-xs text-slate-400">
              LSTM/GRU Short-Term Load Forecasting & Statistical Anomaly System
            </p>
          </div>
        </div>

        {/* Status Indicators & Controls */}
        <div className="flex items-center flex-wrap gap-2.5 sm:gap-3">
          {/* Health Status */}
          <div className={`flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-medium border ${
            isHealthy 
              ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/25' 
              : 'bg-rose-500/10 text-rose-400 border-rose-500/25'
          }`}>
            {isHealthy ? (
              <>
                <span className="h-2 w-2 rounded-full bg-emerald-400 animate-pulse" />
                <span>API Connected</span>
              </>
            ) : (
              <>
                <AlertCircle className="h-3.5 w-3.5" />
                <span>API Disconnected</span>
              </>
            )}
          </div>

          {/* Model Status */}
          <div className={`flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-mono border ${
            isCustomerMode
              ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20'
              : 'bg-blue-500/10 text-blue-400 border-blue-500/20'
          }`}>
            <Cpu className="h-3.5 w-3.5" />
            <span>{modelName}</span>
          </div>

          {/* Horizon */}
          <div className="hidden md:flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-mono bg-slate-800 text-slate-300 border border-slate-700">
            <Activity className="h-3.5 w-3.5 text-amber-400" />
            <span>{isCustomerMode ? 'Horizon: 24h Autoregressive' : 'Horizon: 1h Ahead (24h Window)'}</span>
          </div>

          {/* Refresh Button */}
          <button
            onClick={onRefresh}
            disabled={isRefreshing}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 transition disabled:opacity-50"
            title="Refresh dashboard data"
          >
            <RefreshCw className={`h-3.5 w-3.5 ${isRefreshing ? 'animate-spin text-emerald-400' : ''}`} />
            <span>{isRefreshing ? 'Updating...' : 'Refresh'}</span>
          </button>
        </div>
      </div>
    </header>
  );
}
