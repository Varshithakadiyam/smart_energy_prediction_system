import React from 'react';
import { Zap, TrendingUp, AlertTriangle, ShieldCheck, Flame, BarChart3 } from 'lucide-react';

export default function KpiCards({ prediction, forecast24, anomalies, modelInfo, datasetMode }) {
  const isCustomerMode = datasetMode === 'user';
  // Classification badge color
  const getLevelBadge = (level) => {
    switch (level?.toLowerCase()) {
      case 'low':
        return 'bg-blue-500/15 text-blue-400 border-blue-500/30';
      case 'normal':
        return 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30';
      case 'high':
        return 'bg-amber-500/15 text-amber-400 border-amber-500/30';
      case 'unusually high':
        return 'bg-rose-500/15 text-rose-400 border-rose-500/30';
      default:
        return 'bg-slate-500/15 text-slate-400 border-slate-500/30';
    }
  };

  const nextKwh = prediction?.predicted_consumption ?? 0;
  const nextWh = prediction?.predicted_wh ?? (nextKwh * 1000);
  const total24Kwh = forecast24?.total_predicted_kwh ?? 0;
  const peakHour = forecast24?.peak_hour ?? '--:--';
  const peakKwh = forecast24?.peak_consumption ?? 0;
  const anomalyCount = anomalies?.anomalies_detected ?? 0;
  const r2Score = modelInfo?.metrics?.R2_score ?? 0.6489;
  const maeWh = modelInfo?.metrics?.MAE_Wh ?? 406.86;

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
      {/* 1. Next-Hour Forecast */}
      <div className="glass-panel p-4 flex flex-col justify-between relative overflow-hidden group">
        <div className="absolute top-0 right-0 w-24 h-24 bg-emerald-500/10 rounded-full blur-2xl group-hover:bg-emerald-500/15 transition" />
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">Next-Hour Forecast</span>
          <div className="p-1.5 rounded-lg bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
            <Zap className="h-4 w-4" />
          </div>
        </div>
        <div>
          <div className="flex items-baseline gap-2">
            <span className="text-2xl lg:text-3xl font-extrabold text-white tracking-tight">
              {nextKwh.toFixed(2)}
            </span>
            <span className="text-sm font-semibold text-emerald-400">kWh</span>
          </div>
          <div className="text-xs text-slate-400 font-mono mt-0.5">
            {nextWh.toFixed(1)} Wh
          </div>
        </div>
        <div className="mt-3 pt-3 border-t border-slate-800/80 flex items-center justify-between">
          <span className="text-xs text-slate-400">Demand Tier</span>
          <span className={`text-xs px-2 py-0.5 rounded-full font-semibold border ${getLevelBadge(prediction?.level)}`}>
            {prediction?.level || 'Normal'}
          </span>
        </div>
      </div>

      {/* 2. 24-Hour Projected Total */}
      <div className="glass-panel p-4 flex flex-col justify-between relative overflow-hidden group">
        <div className="absolute top-0 right-0 w-24 h-24 bg-blue-500/10 rounded-full blur-2xl group-hover:bg-blue-500/15 transition" />
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">24h Total Projected</span>
          <div className="p-1.5 rounded-lg bg-blue-500/10 text-blue-400 border border-blue-500/20">
            <TrendingUp className="h-4 w-4" />
          </div>
        </div>
        <div>
          <div className="flex items-baseline gap-2">
            <span className="text-2xl lg:text-3xl font-extrabold text-white tracking-tight">
              {total24Kwh.toFixed(2)}
            </span>
            <span className="text-sm font-semibold text-blue-400">kWh</span>
          </div>
          <div className="text-xs text-slate-400 mt-0.5">
            Avg: {(total24Kwh / 24).toFixed(2)} kWh/hour
          </div>
        </div>
        <div className="mt-3 pt-3 border-t border-slate-800/80 flex items-center justify-between">
          <span className="text-xs text-slate-400">Horizon</span>
          <span className="text-xs font-mono text-slate-300">24-Step Recursive</span>
        </div>
      </div>

      {/* 3. Projected Peak Hour */}
      <div className="glass-panel p-4 flex flex-col justify-between relative overflow-hidden group">
        <div className="absolute top-0 right-0 w-24 h-24 bg-amber-500/10 rounded-full blur-2xl group-hover:bg-amber-500/15 transition" />
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">Projected Peak Load</span>
          <div className="p-1.5 rounded-lg bg-amber-500/10 text-amber-400 border border-amber-500/20">
            <Flame className="h-4 w-4" />
          </div>
        </div>
        <div>
          <div className="flex items-baseline gap-2">
            <span className="text-2xl lg:text-3xl font-extrabold text-white tracking-tight">
              {peakHour}
            </span>
            <span className="text-xs font-mono text-amber-400 font-semibold">({peakKwh.toFixed(2)} kWh)</span>
          </div>
          <div className="text-xs text-slate-400 mt-0.5">
            {isCustomerMode ? 'Personal Peak Window' : 'Diurnal Evening Ramp Window'}
          </div>
        </div>
        <div className="mt-3 pt-3 border-t border-slate-800/80 flex items-center justify-between">
          <span className="text-xs text-slate-400">Lowest Window</span>
          <span className="text-xs font-mono text-slate-300">
            {forecast24?.lowest_hour || '03:00'} ({(forecast24?.lowest_consumption ?? 0.3).toFixed(2)} kWh)
          </span>
        </div>
      </div>

      {/* 4. Anomaly Alert Status */}
      <div className="glass-panel p-4 flex flex-col justify-between relative overflow-hidden group">
        <div className="absolute top-0 right-0 w-24 h-24 bg-rose-500/10 rounded-full blur-2xl group-hover:bg-rose-500/15 transition" />
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">Anomaly Monitor</span>
          <div className={`p-1.5 rounded-lg border ${
            anomalyCount > 0 
              ? 'bg-rose-500/10 text-rose-400 border-rose-500/20' 
              : 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20'
          }`}>
            {anomalyCount > 0 ? <AlertTriangle className="h-4 w-4" /> : <ShieldCheck className="h-4 w-4" />}
          </div>
        </div>
        <div>
          <div className="flex items-baseline gap-2">
            <span className="text-2xl lg:text-3xl font-extrabold text-white tracking-tight">
              {anomalyCount}
            </span>
            <span className="text-xs font-semibold text-slate-300">Detected ({anomalyCount > 0 ? 'Surges' : 'Zero'})</span>
          </div>
          <div className="text-xs text-slate-400 mt-0.5">
            Threshold: ±2.5σ by Hour-of-Day
          </div>
        </div>
        <div className="mt-3 pt-3 border-t border-slate-800/80 flex items-center justify-between">
          <span className="text-xs text-slate-400">{isCustomerMode ? 'Active Model' : 'Model Accuracy'}</span>
          <span className="text-xs font-mono text-emerald-400 font-semibold">
            {isCustomerMode ? 'Univariate GRU' : `R² = ${r2Score.toFixed(4)}`}
          </span>
        </div>
      </div>
    </div>
  );
}
