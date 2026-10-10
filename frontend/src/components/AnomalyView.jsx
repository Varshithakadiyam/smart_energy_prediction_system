import React, { useState } from 'react';
import { 
  AlertTriangle, 
  ShieldCheck, 
  HelpCircle, 
  Filter, 
  ArrowUp, 
  ArrowDown, 
  Info, 
  ChevronDown, 
  ChevronUp 
} from 'lucide-react';

export default function AnomalyView({ anomalies, datasetMode, userValidationReport }) {
  const [filter, setFilter] = useState('all'); // 'all', 'anomalies', 'normal'
  const [expandedRow, setExpandedRow] = useState(null);
  const isCustomerMode = datasetMode === 'user';
  const results = anomalies?.results || [];

  const filteredResults = results.filter(item => {
    if (filter === 'anomalies') return item.is_anomaly;
    if (filter === 'normal') return !item.is_anomaly;
    return true;
  });

  const anomalyCount = anomalies?.anomalies_detected ?? 0;
  const totalEvaluated = anomalies?.total_evaluated ?? 0;
  const hasSufficientHistory = anomalies?.has_sufficient_history ?? true;

  const getSeverityBadge = (severity) => {
    switch (severity?.toLowerCase()) {
      case 'high':
        return 'bg-rose-500/20 text-rose-300 border-rose-500/40';
      case 'moderate':
        return 'bg-amber-500/20 text-amber-300 border-amber-500/40';
      case 'normal':
      default:
        return 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20';
    }
  };

  return (
    <div className="space-y-6">
      {/* Overview & Methodology Panel */}
      <div className="glass-panel p-5">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2">
              <AlertTriangle className="h-5 w-5 text-amber-400" />
              <h2 className="text-base font-bold text-white">
                {isCustomerMode 
                  ? 'Personalized Diurnal Anomaly Detection' 
                  : 'Statistical Diurnal Anomaly Detection'}
              </h2>
            </div>
            <p className="text-xs text-slate-300 mt-1 max-w-3xl leading-relaxed">
              Standard static thresholding fails on electrical demand because nighttime standby (~0.3 kW) differs vastly from legitimate evening cooking/heating loads (~2.5 kW). Our system computes dynamic diurnal thresholds using hour-of-day distributions:
            </p>
            <div className="mt-2.5 inline-block bg-slate-950/80 px-3.5 py-1.5 rounded-lg border border-slate-800 font-mono text-xs text-emerald-400">
              Threshold(h) = μ_hour ± 2.5 × σ_hour &nbsp;|&nbsp; Z = (Actual - μ_hour) / σ_hour
            </div>
          </div>

          <div className="flex items-center gap-3 shrink-0">
            <div className="bg-slate-900/80 p-3 rounded-xl border border-slate-800 text-center min-w-[110px]">
              <div className="text-2xl font-extrabold text-white font-mono">{totalEvaluated}</div>
              <div className="text-[11px] text-slate-400 uppercase tracking-wider">Evaluated</div>
            </div>
            <div className={`p-3 rounded-xl border text-center min-w-[110px] ${
              anomalyCount > 0 
                ? 'bg-rose-500/10 border-rose-500/30 text-rose-400' 
                : 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400'
            }`}>
              <div className="text-2xl font-extrabold font-mono">{anomalyCount}</div>
              <div className="text-[11px] uppercase tracking-wider">Anomalies</div>
            </div>
          </div>
        </div>

        {/* Insufficient History Advisory Banner */}
        {isCustomerMode && !hasSufficientHistory && (
          <div className="mt-4 p-3 bg-amber-500/10 border border-amber-500/30 rounded-xl flex items-start gap-2.5 text-xs text-amber-300">
            <Info className="h-4 w-4 shrink-0 mt-0.5 text-amber-400" />
            <div>
              <strong className="text-amber-200">Baseline Advisory:</strong> Your uploaded dataset contains fewer than 168 hours (7 days) of historical observations ({userValidationReport?.accepted_rows || 0} hours loaded). The diurnal baseline distribution is estimated across available data. For highest seasonal precision, 7+ days of continuous hourly data is recommended.
            </div>
          </div>
        )}
      </div>

      {/* Anomaly Table and Filtering */}
      <div className="glass-panel p-5">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4">
          <div>
            <h3 className="text-sm font-bold text-white">Observation Analysis & Z-Score Residuals</h3>
            <p className="text-xs text-slate-400 mt-0.5">
              {isCustomerMode 
                ? `Derived from active upload: ${userValidationReport?.filename || 'customer CSV'}` 
                : 'Derived from UCI household benchmark unseen test set'}
            </p>
          </div>

          {/* Filters */}
          <div className="flex items-center gap-1.5 bg-slate-950/60 p-1 rounded-lg border border-slate-800 text-xs">
            <button
              onClick={() => setFilter('all')}
              className={`px-2.5 py-1 rounded font-medium transition ${
                filter === 'all' ? 'bg-slate-800 text-white' : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              All ({totalEvaluated})
            </button>
            <button
              onClick={() => setFilter('anomalies')}
              className={`px-2.5 py-1 rounded font-medium transition ${
                filter === 'anomalies' ? 'bg-rose-500/20 text-rose-400 border border-rose-500/30' : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Anomalies ({anomalyCount})
            </button>
            <button
              onClick={() => setFilter('normal')}
              className={`px-2.5 py-1 rounded font-medium transition ${
                filter === 'normal' ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30' : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Normal ({totalEvaluated - anomalyCount})
            </button>
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs border-collapse">
            <thead>
              <tr className="border-b border-slate-800 text-slate-400 uppercase tracking-wider font-semibold">
                <th className="py-2.5 px-3">Timestamp</th>
                <th className="py-2.5 px-3">Actual Load</th>
                <th className="py-2.5 px-3">Hourly Baseline (μ)</th>
                <th className="py-2.5 px-3">Deviation (Δ)</th>
                <th className="py-2.5 px-3">Z-Score</th>
                <th className="py-2.5 px-3">Severity</th>
                <th className="py-2.5 px-3">Assessment</th>
                <th className="py-2.5 px-3 text-right">Details</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 font-mono">
              {filteredResults.map((item, idx) => {
                const isAnomaly = item.is_anomaly;
                const actual = item.actual_consumption_kwh ?? item.actual_consumption ?? 0;
                const expected = item.expected_kwh ?? item.expected_consumption ?? 0;
                const deviation = item.deviation_kwh ?? item.deviation ?? 0;
                const status = item.status ?? item.anomaly_status ?? (isAnomaly ? 'Anomalous Spike' : 'Normal');
                const severity = item.severity || (isAnomaly ? 'High' : 'Normal');
                const explanation = item.explanation;
                const isExpanded = expandedRow === idx;

                return (
                  <React.Fragment key={idx}>
                    <tr 
                      className={`hover:bg-slate-800/40 transition-colors ${
                        isAnomaly ? 'bg-rose-500/10' : ''
                      }`}
                    >
                      <td className="py-2.5 px-3 text-slate-300">
                        {item.timestamp?.replace('T', ' ')}
                      </td>
                      <td className="py-2.5 px-3 font-bold text-white">
                        {actual.toFixed(2)} kWh
                      </td>
                      <td className="py-2.5 px-3 text-slate-400">
                        {expected.toFixed(2)} kWh
                      </td>
                      <td className="py-2.5 px-3">
                        <span className={`flex items-center gap-1 ${
                          deviation > 0 ? 'text-amber-400' : 'text-blue-400'
                        }`}>
                          {deviation > 0 ? <ArrowUp className="h-3 w-3" /> : <ArrowDown className="h-3 w-3" />}
                          {Math.abs(deviation).toFixed(2)} kWh
                        </span>
                      </td>
                      <td className="py-2.5 px-3 font-bold">
                        <span className={isAnomaly ? 'text-rose-400' : 'text-slate-400'}>
                          {item.z_score > 0 ? `+${item.z_score.toFixed(2)}σ` : `${item.z_score.toFixed(2)}σ`}
                        </span>
                      </td>
                      <td className="py-2.5 px-3 font-sans">
                        <span className={`px-2 py-0.5 rounded-full text-[11px] font-semibold border ${getSeverityBadge(severity)}`}>
                          {severity}
                        </span>
                      </td>
                      <td className="py-2.5 px-3 font-sans">
                        {isAnomaly ? (
                          <span className="inline-flex items-center gap-1 text-xs font-semibold text-rose-300">
                            <AlertTriangle className="h-3.5 w-3.5 shrink-0 text-rose-400" />
                            {status}
                          </span>
                        ) : (
                          <span className="inline-flex items-center gap-1 text-xs font-medium text-emerald-400">
                            <ShieldCheck className="h-3.5 w-3.5 shrink-0" /> Normal Diurnal Pattern
                          </span>
                        )}
                      </td>
                      <td className="py-2.5 px-3 text-right">
                        {explanation ? (
                          <button
                            onClick={() => setExpandedRow(isExpanded ? null : idx)}
                            className="text-slate-400 hover:text-white p-1 rounded hover:bg-slate-800 transition"
                            title="Toggle explanation"
                          >
                            {isExpanded ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
                          </button>
                        ) : (
                          <span className="text-slate-600 font-sans text-[11px]">--</span>
                        )}
                      </td>
                    </tr>
                    {isExpanded && explanation && (
                      <tr className="bg-slate-900/90 font-sans">
                        <td colSpan={8} className="py-2.5 px-4 text-xs text-slate-300 border-l-2 border-amber-500">
                          <span className="text-amber-400 font-semibold">Diagnosis: </span>
                          {explanation}
                        </td>
                      </tr>
                    )}
                  </React.Fragment>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
