import React from 'react';
import { Award, Cpu, BarChart2, CheckCircle2, Layers, Zap } from 'lucide-react';

export default function ModelBenchmarksView({ modelInfo }) {
  const baselines = modelInfo?.baseline_comparison || [
    { model: "Previous Hour (Persistence)", mae_wh: 496.30, rmse_wh: 766.10, mape_pct: 48.11, r2: 0.4411 },
    { model: "Previous Day (24h Persistence)", mae_wh: 732.42, rmse_wh: 1086.26, mape_pct: 92.06, r2: -0.1236 },
    { model: "Moving Average (24h)", mae_wh: 785.88, rmse_wh: 1013.10, mape_pct: 125.25, r2: 0.0227 },
    { model: "LSTM (Baseline)", mae_wh: 407.03, rmse_wh: 614.46, mape_pct: 44.96, r2: 0.6405 },
    { model: "GRU (Best Selected)", mae_wh: 406.86, rmse_wh: 607.26, mape_pct: 44.80, r2: 0.6489 }
  ];

  const features = modelInfo?.features_used || [
    'energy_consumption', 'hour', 'day_of_week', 'is_weekend', 'sin_hour', 'cos_hour',
    'sin_day_of_week', 'cos_day_of_week', 'lag_1', 'lag_24', 'lag_168', 'rolling_mean_24h',
    'rolling_std_24h', 'average_voltage', 'average_global_reactive_power',
    'average_global_intensity', 'sub_metering_1_wh', 'sub_metering_2_wh', 'sub_metering_3_wh'
  ];

  return (
    <div className="space-y-6">
      {/* Benchmark Summary Header */}
      <div className="glass-panel p-5">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2">
              <Award className="h-5 w-5 text-amber-400" />
              <h2 className="text-base font-bold text-white">Empirical Model Benchmarking & Baseline Evaluation</h2>
            </div>
            <p className="text-xs text-slate-300 mt-1 max-w-3xl leading-relaxed">
              Models evaluated strictly on the 1,201 unseen test sequences (October–November 2007) in original unscaled energy units (Wh and kWh). All data partitions were chronologically split with zero future leakage.
            </p>
          </div>
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 text-xs font-mono">
            <CheckCircle2 className="h-4 w-4" />
            <span>Best Model: GRU (Lowest RMSE & Max R²)</span>
          </div>
        </div>
      </div>

      {/* Model Comparison Table */}
      <div className="glass-panel p-5">
        <h3 className="text-sm font-bold text-white mb-3">Unseen Test Set Performance Comparison</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs border-collapse">
            <thead>
              <tr className="border-b border-slate-800 text-slate-400 uppercase tracking-wider font-semibold">
                <th className="py-2.5 px-3">Model Architecture</th>
                <th className="py-2.5 px-3">Category</th>
                <th className="py-2.5 px-3">MAE (Wh)</th>
                <th className="py-2.5 px-3">RMSE (Wh)</th>
                <th className="py-2.5 px-3">MAPE (%)</th>
                <th className="py-2.5 px-3">R² Score</th>
                <th className="py-2.5 px-3">MAE (kWh)</th>
                <th className="py-2.5 px-3">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 font-mono">
              {baselines.map((m, idx) => {
                const isSelected = m.model.includes('Best') || m.model.includes('GRU');
                const isDl = m.model.includes('LSTM') || m.model.includes('GRU');

                return (
                  <tr 
                    key={idx} 
                    className={`hover:bg-slate-800/40 transition-colors ${
                      isSelected ? 'bg-emerald-500/10 font-semibold' : ''
                    }`}
                  >
                    <td className="py-2.5 px-3 font-sans text-white font-medium">
                      {m.model}
                    </td>
                    <td className="py-2.5 px-3 font-sans text-slate-400">
                      {isDl ? 'Deep Learning RNN' : 'Statistical Baseline'}
                    </td>
                    <td className="py-2.5 px-3 text-slate-200">
                      {m.mae_wh.toFixed(2)}
                    </td>
                    <td className="py-2.5 px-3 text-slate-200">
                      {m.rmse_wh.toFixed(2)}
                    </td>
                    <td className="py-2.5 px-3 text-slate-300">
                      {m.mape_pct.toFixed(2)}%
                    </td>
                    <td className="py-2.5 px-3 font-bold">
                      <span className={m.r2 > 0.6 ? 'text-emerald-400' : m.r2 < 0 ? 'text-rose-400' : 'text-slate-400'}>
                        {m.r2.toFixed(4)}
                      </span>
                    </td>
                    <td className="py-2.5 px-3 text-slate-300">
                      {(m.mae_wh / 1000.0).toFixed(3)}
                    </td>
                    <td className="py-2.5 px-3 font-sans">
                      {isSelected ? (
                        <span className="px-2 py-0.5 rounded-full text-[11px] font-semibold bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
                          Active Model
                        </span>
                      ) : (
                        <span className="text-slate-500 text-[11px]">Evaluated</span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Visual Error Bar Comparison */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* MAE Comparison Chart */}
        <div className="glass-panel p-5">
          <div className="flex items-center gap-2 mb-3">
            <BarChart2 className="h-4 w-4 text-emerald-400" />
            <h4 className="text-sm font-bold text-white">Mean Absolute Error (Wh) — Lower is Better</h4>
          </div>
          <div className="space-y-3 font-mono text-xs">
            {baselines.map((m, idx) => {
              const maxMae = 850;
              const pct = Math.min(100, (m.mae_wh / maxMae) * 100);
              const isBest = m.model.includes('GRU');

              return (
                <div key={idx} className="space-y-1">
                  <div className="flex justify-between text-slate-300 font-sans">
                    <span>{m.model}</span>
                    <span className="font-mono font-bold text-white">{m.mae_wh.toFixed(1)} Wh</span>
                  </div>
                  <div className="w-full bg-slate-950 h-2.5 rounded-full overflow-hidden border border-slate-800">
                    <div 
                      className={`h-full rounded-full transition-all duration-500 ${
                        isBest ? 'bg-gradient-to-r from-emerald-500 to-teal-400' : 'bg-slate-600'
                      }`}
                      style={{ width: `${pct}%` }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* RMSE Comparison Chart */}
        <div className="glass-panel p-5">
          <div className="flex items-center gap-2 mb-3">
            <BarChart2 className="h-4 w-4 text-cyan-400" />
            <h4 className="text-sm font-bold text-white">Root Mean Squared Error (Wh) — Lower is Better</h4>
          </div>
          <div className="space-y-3 font-mono text-xs">
            {baselines.map((m, idx) => {
              const maxRmse = 1150;
              const pct = Math.min(100, (m.rmse_wh / maxRmse) * 100);
              const isBest = m.model.includes('GRU');

              return (
                <div key={idx} className="space-y-1">
                  <div className="flex justify-between text-slate-300 font-sans">
                    <span>{m.model}</span>
                    <span className="font-mono font-bold text-white">{m.rmse_wh.toFixed(1)} Wh</span>
                  </div>
                  <div className="w-full bg-slate-950 h-2.5 rounded-full overflow-hidden border border-slate-800">
                    <div 
                      className={`h-full rounded-full transition-all duration-500 ${
                        isBest ? 'bg-gradient-to-r from-cyan-500 to-blue-400' : 'bg-slate-600'
                      }`}
                      style={{ width: `${pct}%` }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* Feature Engineering Inventory */}
      <div className="glass-panel p-5">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <Layers className="h-4 w-4 text-emerald-400" />
            <h4 className="text-sm font-bold text-white">Multivariate Feature Engineering Matrix ({features.length} Channels)</h4>
          </div>
          <span className="text-xs text-slate-400 font-mono">Input Shape: (24, {features.length})</span>
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-2">
          {features.map((feat, idx) => (
            <div key={idx} className="bg-slate-950/70 p-2 rounded border border-slate-800 text-xs font-mono text-slate-300 flex items-center justify-between">
              <span className="truncate">{feat}</span>
              <span className="text-[10px] text-slate-400 font-sans ml-1">#{idx + 1}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
