import React from 'react';
import { 
  Lightbulb, 
  AlertCircle, 
  Clock, 
  Shield, 
  Sliders, 
  CheckCircle2, 
  Sparkles, 
  ArrowRight,
  TrendingDown
} from 'lucide-react';

export default function RecommendationsView({ recommendations, datasetMode, userValidationReport }) {
  const isCustomerMode = datasetMode === 'user';
  const recList = recommendations?.recommendations || [];

  const getPriorityBadge = (priority) => {
    switch (priority?.toLowerCase()) {
      case 'high':
        return 'bg-rose-500/15 text-rose-400 border-rose-500/30';
      case 'medium':
        return 'bg-amber-500/15 text-amber-400 border-amber-500/30';
      case 'low':
        return 'bg-blue-500/15 text-blue-400 border-blue-500/30';
      default:
        return 'bg-slate-500/15 text-slate-400 border-slate-500/30';
    }
  };

  const getCategoryIcon = (category) => {
    switch (category?.toLowerCase()) {
      case 'personalized peak shifting':
      case 'peak load shifting':
      case 'peak management':
        return <Clock className="h-5 w-5 text-amber-400" />;
      case 'baseload optimization':
      case 'vampire power':
      case 'base load':
        return <Shield className="h-5 w-5 text-blue-400" />;
      case 'high demand alert':
      case 'anomaly mitigation':
        return <AlertCircle className="h-5 w-5 text-rose-400" />;
      default:
        return <Sliders className="h-5 w-5 text-emerald-400" />;
    }
  };

  return (
    <div className="space-y-6">
      {/* Header and Disclaimer */}
      <div className="glass-panel p-5">
        <div className="flex items-center gap-2 mb-2">
          <Lightbulb className="h-5 w-5 text-emerald-400" />
          <h2 className="text-base font-bold text-white">
            {isCustomerMode 
              ? 'Personalized Energy Optimization Recommendations' 
              : 'Demand-Side Energy Optimization Recommendations'}
          </h2>
        </div>
        <p className="text-xs text-slate-300 leading-relaxed max-w-4xl">
          {isCustomerMode 
            ? `Actionable demand-shifting, baseload mitigation, and tariff efficiency guidance tailored specifically to ${userValidationReport?.filename || 'your uploaded energy dataset'}.`
            : 'Actionable demand-shifting and efficiency guidance derived from upcoming 24-hour neural network forecasts and diurnal load curves.'}
        </p>

        {/* Scientific Disclaimer */}
        <div className="mt-3 bg-slate-900/80 p-3 rounded-lg border border-slate-800 text-xs text-slate-400 flex items-start gap-2.5">
          <CheckCircle2 className="h-4 w-4 text-emerald-400 shrink-0 mt-0.5" />
          <div>
            <strong className="text-slate-200">Scientific Integrity Notice:</strong> Recommendations are derived from aggregate household power trajectories, diurnal profiles, and model forecasts. Estimates represent potential reductions based on typical electrical appliance duty cycles and do not constitute guaranteed utility billing credits.
          </div>
        </div>
      </div>

      {/* Recommendations Cards Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {recList.map((rec, idx) => (
          <div key={idx} className="glass-panel p-5 flex flex-col justify-between space-y-4 hover:border-slate-700 transition">
            <div className="space-y-3">
              <div className="flex items-center justify-between gap-3">
                <div className="flex items-center gap-2">
                  <div className="p-2 rounded-lg bg-slate-800/80 border border-slate-700">
                    {getCategoryIcon(rec.category)}
                  </div>
                  <div>
                    <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-400">
                      {rec.category}
                    </span>
                    <h3 className="text-sm font-bold text-white">
                      {rec.title || 'Optimization Strategy'}
                    </h3>
                  </div>
                </div>
                <span className={`px-2.5 py-0.5 rounded-full text-xs font-semibold border ${getPriorityBadge(rec.priority)}`}>
                  {rec.priority} Priority
                </span>
              </div>

              {/* Core Recommendation Statement */}
              <p className="text-xs text-slate-200 leading-relaxed font-medium">
                {rec.recommendation}
              </p>

              {/* Contextual Explanation if available */}
              {rec.explanation && (
                <p className="text-xs text-slate-400 leading-relaxed pl-2 border-l-2 border-slate-700">
                  {rec.explanation}
                </p>
              )}

              {/* Trigger Pattern Box */}
              {rec.trigger_pattern && (
                <div className="bg-slate-950/70 p-2.5 rounded-lg border border-slate-800 text-xs text-slate-300">
                  <span className="text-amber-400 font-semibold text-[10px] uppercase tracking-wider block mb-0.5">
                    Observation Trigger
                  </span>
                  <span>{rec.trigger_pattern}</span>
                </div>
              )}

              {/* Actionable Step Box */}
              {rec.action && (
                <div className="bg-emerald-950/30 p-2.5 rounded-lg border border-emerald-500/30 text-xs text-emerald-200">
                  <span className="text-emerald-400 font-semibold text-[10px] uppercase tracking-wider block mb-0.5 flex items-center gap-1">
                    <ArrowRight className="h-3 w-3" /> Recommended Action
                  </span>
                  <span>{rec.action}</span>
                </div>
              )}

              {/* Estimated Savings Pill */}
              {rec.estimated_savings && (
                <div className="flex items-center gap-2 text-xs font-mono text-emerald-300 bg-emerald-500/10 px-3 py-1.5 rounded-lg border border-emerald-500/25">
                  <TrendingDown className="h-3.5 w-3.5 text-emerald-400 shrink-0" />
                  <span>{rec.estimated_savings}</span>
                </div>
              )}
            </div>

            <div className="pt-3 border-t border-slate-800/80 flex items-center justify-between text-[11px] text-slate-400">
              <span>{isCustomerMode ? 'Targeted to Active Upload' : 'Grounding: 24h Forward Horizon'}</span>
              <span className="font-mono text-emerald-400 font-semibold">
                {isCustomerMode ? 'Customer-Specific' : 'Model-Driven'}
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

