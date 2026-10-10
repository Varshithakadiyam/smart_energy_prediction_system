import React from 'react';
import { Sun, Calendar, Activity } from 'lucide-react';

export default function DiurnalPatternChart({ 
  historicalData, 
  datasetMode, 
  userValidationReport, 
  userForecastResult 
}) {
  const isCustomerMode = datasetMode === 'user';
  const customerDiurnal = userForecastResult?.diurnal_profile || userValidationReport?.diurnal_profile;
  const rawTimeline = userForecastResult?.historical_timeline || userValidationReport?.historical_timeline || [];

  const diurnal = isCustomerMode && customerDiurnal && customerDiurnal.length > 0
    ? customerDiurnal.map(d => ({
        ...d,
        mean_kwh: d.average_kwh ?? d.mean_kwh ?? 0,
        std_kwh: d.std_kwh ?? 0
      }))
    : (historicalData?.diurnal_pattern || []);

  let dow = historicalData?.day_of_week_pattern || [];
  if (isCustomerMode && rawTimeline.length > 0) {
    const dayBuckets = [[], [], [], [], [], [], []];
    rawTimeline.forEach(item => {
      const day = new Date(item.timestamp).getDay();
      dayBuckets[day].push(item.energy_consumption_kwh);
    });
    const reorderedDays = [1, 2, 3, 4, 5, 6, 0];
    const dowNamesMonFirst = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
    dow = reorderedDays.map((dIdx, i) => {
      const vals = dayBuckets[dIdx];
      const mean = vals.length > 0 ? vals.reduce((a, b) => a + b, 0) / vals.length : 0;
      return {
        day_of_week: i,
        day_name: dowNamesMonFirst[i],
        mean_kwh: mean
      };
    });
  }

  if (diurnal.length === 0) return null;

  const maxDiurnal = Math.max(...diurnal.map(d => d.mean_kwh + (d.std_kwh || 0)), 2.5);
  const svgW = 600;
  const svgH = 180;
  const padX = 35;
  const padY = 25;
  const chartW = svgW - padX * 2;
  const chartH = svgH - padY * 2;

  // Diurnal points
  const points = diurnal.map((d, i) => {
    const x = padX + (i / (diurnal.length - 1)) * chartW;
    const y = svgH - padY - (d.mean_kwh / maxDiurnal) * chartH;
    return { x, y, ...d };
  });

  const pathLine = points.reduce((acc, pt, i) => (
    i === 0 ? `M ${pt.x} ${pt.y}` : `${acc} L ${pt.x} ${pt.y}`
  ), '');

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
      {/* 24-Hour Diurnal Load Profile */}
      <div className="glass-panel p-5">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <Sun className="h-4 w-4 text-amber-400" />
            <h3 className="text-sm font-bold text-white">
              {isCustomerMode ? 'Customer Diurnal 24-Hour Load Profile' : 'Diurnal 24-Hour Average Load Curve'}
            </h3>
          </div>
          <span className="text-[11px] font-mono text-slate-400">
            {isCustomerMode ? 'Personal Hourly Mean (kWh)' : 'Mean Hourly Profile (kWh)'}
          </span>
        </div>

        <div className="relative w-full overflow-hidden bg-slate-950/40 rounded-xl p-2 border border-slate-800/60">
          <svg viewBox={`0 0 ${svgW} ${svgH}`} className="w-full h-auto select-none">
            {/* Grid */}
            {[0, 1, 2].map((g) => {
              if (g > maxDiurnal) return null;
              const y = svgH - padY - (g / maxDiurnal) * chartH;
              return (
                <g key={g}>
                  <line x1={padX} y1={y} x2={svgW - padX} y2={y} stroke="#1e293b" strokeDasharray="3 3" />
                  <text x={padX - 6} y={y + 3} fill="#64748b" fontSize="8" textAnchor="end">{g}k</text>
                </g>
              );
            })}

            {/* Line */}
            <path d={pathLine} fill="none" stroke="#f59e0b" strokeWidth="2.5" strokeLinecap="round" />

            {/* Nodes */}
            {points.map((pt, idx) => (
              <g key={idx}>
                <circle cx={pt.x} cy={pt.y} r="2.5" fill="#f59e0b" />
                {idx % 4 === 0 && (
                  <text x={pt.x} y={svgH - 8} fill="#64748b" fontSize="8" textAnchor="middle">
                    {pt.hour_label}
                  </text>
                )}
              </g>
            ))}
          </svg>
        </div>
      </div>

      {/* Day of Week Profile */}
      <div className="glass-panel p-5">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <Calendar className="h-4 w-4 text-emerald-400" />
            <h3 className="text-sm font-bold text-white">Day-of-Week Average Consumption</h3>
          </div>
          <span className="text-[11px] font-mono text-slate-400">Weekly Dynamics (kWh)</span>
        </div>

        <div className="grid grid-cols-7 gap-2 pt-2">
          {dow.map((d, idx) => {
            const isWeekend = idx >= 5;
            const maxDow = 1.6;
            const heightPct = Math.min(100, (d.mean_kwh / maxDow) * 100);

            return (
              <div key={idx} className="flex flex-col items-center gap-2">
                <div className="h-28 w-full bg-slate-950/60 rounded-lg p-1 flex flex-col justify-end items-center border border-slate-800">
                  <div
                    className={`w-full rounded-md transition-all ${
                      isWeekend 
                        ? 'bg-gradient-to-t from-emerald-600 to-teal-400' 
                        : 'bg-gradient-to-t from-blue-600 to-cyan-400'
                    }`}
                    style={{ height: `${heightPct}%` }}
                  />
                </div>
                <div className="text-center">
                  <div className={`text-xs font-semibold ${isWeekend ? 'text-emerald-400' : 'text-slate-200'}`}>
                    {d.day_name}
                  </div>
                  <div className="text-[10px] font-mono text-slate-400">
                    {d.mean_kwh.toFixed(2)}k
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
