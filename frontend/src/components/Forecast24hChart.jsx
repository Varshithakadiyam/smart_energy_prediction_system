import React, { useState } from 'react';
import { Clock, TrendingUp, ArrowDown, ArrowUp, Table as TableIcon, Download } from 'lucide-react';

export default function Forecast24hChart({ forecast24, datasetMode, userValidationReport }) {
  const [hoveredIdx, setHoveredIdx] = useState(null);
  const predictions = forecast24?.predictions || [];
  const isCustomerMode = datasetMode === 'user';

  const handleExportForecastCsv = () => {
    if (!predictions || predictions.length === 0) return;
    const header = "step,timestamp,hour_label,predicted_consumption_kwh,predicted_wh,level\n";
    const rows = predictions.map(p => 
      `${p.step},"${p.timestamp || ''}","${p.hour_label || ''}",${p.predicted_consumption},${(p.predicted_wh ?? (p.predicted_consumption * 1000)).toFixed(1)},"${p.level || 'Normal'}"`
    ).join("\n");
    const blob = new Blob([header + rows], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `forecast_24h_${isCustomerMode ? 'customer' : 'benchmark'}_${new Date().toISOString().slice(0, 10)}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  if (predictions.length === 0) {
    return (
      <div className="glass-panel p-6 text-center text-slate-400">
        No forecast data available. Click Refresh or verify backend connectivity.
      </div>
    );
  }

  // Calculate SVG bounds & scaling
  const maxVal = Math.max(...predictions.map(p => p.predicted_consumption), 4.0) * 1.15;
  const minVal = 0;
  const svgWidth = 800;
  const svgHeight = 240;
  const paddingX = 40;
  const paddingY = 30;
  const chartW = svgWidth - paddingX * 2;
  const chartH = svgHeight - paddingY * 2;

  const points = predictions.map((p, i) => {
    const x = paddingX + (i / (predictions.length - 1)) * chartW;
    const y = svgHeight - paddingY - ((p.predicted_consumption - minVal) / (maxVal - minVal)) * chartH;
    return { x, y, ...p };
  });

  const pathD = points.reduce((acc, pt, i) => {
    return i === 0 ? `M ${pt.x} ${pt.y}` : `${acc} L ${pt.x} ${pt.y}`;
  }, '');

  const areaD = `${pathD} L ${points[points.length - 1].x} ${svgHeight - paddingY} L ${points[0].x} ${svgHeight - paddingY} Z`;

  const hoveredPt = hoveredIdx !== null ? points[hoveredIdx] : null;

  const getLevelColor = (level) => {
    switch (level?.toLowerCase()) {
      case 'low': return 'text-blue-400 bg-blue-500/10 border-blue-500/30';
      case 'normal': return 'text-emerald-400 bg-emerald-500/10 border-emerald-500/30';
      case 'high': return 'text-amber-400 bg-amber-500/10 border-amber-500/30';
      case 'unusually high': return 'text-rose-400 bg-rose-500/10 border-rose-500/30';
      default: return 'text-slate-400 bg-slate-500/10 border-slate-500/30';
    }
  };

  return (
    <div className="space-y-6">
      {/* Chart Card */}
      <div className="glass-panel p-5">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4">
          <div>
            <div className="flex items-center gap-2">
              <Clock className="h-4 w-4 text-emerald-400" />
              <h2 className="text-base font-bold text-white">
                {isCustomerMode ? 'Personalized 24-Hour Demand Forecast' : '24-Hour Ahead Hourly Demand Forecast'}
              </h2>
            </div>
            <p className="text-xs text-slate-400 mt-0.5">
              {isCustomerMode 
                ? `Generated from ${userValidationReport?.filename || 'customer CSV'} using Univariate GRU` 
                : 'Autoregressive recursive roll-forward using frozen historical feature scaler parameters'}
            </p>
          </div>
          <div className="flex items-center flex-wrap gap-3 text-xs">
            <span className="flex items-center gap-1.5 text-slate-300">
              <span className="h-2.5 w-2.5 rounded-full bg-emerald-500" />
              <span>Projected Load (kWh)</span>
            </span>
            <span className="text-slate-500">|</span>
            <span className="font-mono text-emerald-400 font-semibold">
              Peak: {forecast24.peak_hour} ({forecast24.peak_consumption.toFixed(2)} kWh)
            </span>
            <button
              onClick={handleExportForecastCsv}
              className="flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-semibold bg-emerald-500/15 hover:bg-emerald-500/25 text-emerald-300 border border-emerald-500/30 transition shadow-sm"
              title="Download 24-hour forecast as CSV"
            >
              <Download className="h-3.5 w-3.5" />
              <span>Download Forecast CSV</span>
            </button>
          </div>
        </div>

        {/* Interactive SVG Chart */}
        <div className="relative w-full overflow-hidden bg-slate-950/40 rounded-xl p-2 border border-slate-800/60">
          <svg
            viewBox={`0 0 ${svgWidth} ${svgHeight}`}
            className="w-full h-auto overflow-visible select-none"
          >
            <defs>
              <linearGradient id="forecastAreaGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#10b981" stopOpacity="0.4" />
                <stop offset="85%" stopColor="#10b981" stopOpacity="0.0" />
              </linearGradient>
            </defs>

            {/* Horizontal Grid lines */}
            {[0, 1, 2, 3, 4].map((gridVal) => {
              if (gridVal > maxVal) return null;
              const y = svgHeight - paddingY - ((gridVal - minVal) / (maxVal - minVal)) * chartH;
              return (
                <g key={gridVal}>
                  <line
                    x1={paddingX}
                    y1={y}
                    x2={svgWidth - paddingX}
                    y2={y}
                    stroke="#1e293b"
                    strokeDasharray="4 4"
                    strokeWidth="1"
                  />
                  <text
                    x={paddingX - 8}
                    y={y + 3}
                    fill="#64748b"
                    fontSize="9"
                    fontFamily="monospace"
                    textAnchor="end"
                  >
                    {gridVal}k
                  </text>
                </g>
              );
            })}

            {/* Gradient Area */}
            <path d={areaD} fill="url(#forecastAreaGrad)" />

            {/* Foreground Line */}
            <path
              d={pathD}
              fill="none"
              stroke="#10b981"
              strokeWidth="2.5"
              strokeLinecap="round"
              strokeLinejoin="round"
            />

            {/* Points & Hover Target */}
            {points.map((pt, idx) => {
              const isPeak = pt.hour_label === forecast24.peak_hour;
              const isLowest = pt.hour_label === forecast24.lowest_hour;
              const isHovered = hoveredIdx === idx;

              return (
                <g
                  key={idx}
                  className="cursor-pointer"
                  onMouseEnter={() => setHoveredIdx(idx)}
                  onMouseLeave={() => setHoveredIdx(null)}
                >
                  {/* Invisible broad hitbox */}
                  <circle cx={pt.x} cy={pt.y} r="14" fill="transparent" />

                  {/* Visual Node */}
                  <circle
                    cx={pt.x}
                    cy={pt.y}
                    r={isHovered ? 6 : isPeak ? 4.5 : 3}
                    fill={isPeak ? '#f59e0b' : isLowest ? '#3b82f6' : '#10b981'}
                    stroke="#0b0f19"
                    strokeWidth="2"
                    className="transition-all"
                  />

                  {/* X-axis labels (every 3 hours) */}
                  {idx % 3 === 0 && (
                    <text
                      x={pt.x}
                      y={svgHeight - 10}
                      fill="#64748b"
                      fontSize="9"
                      fontFamily="monospace"
                      textAnchor="middle"
                    >
                      {pt.hour_label}
                    </text>
                  )}
                </g>
              );
            })}

            {/* Active Hover Guideline & Pin */}
            {hoveredPt && (
              <g pointerEvents="none">
                <line
                  x1={hoveredPt.x}
                  y1={paddingY}
                  x2={hoveredPt.x}
                  y2={svgHeight - paddingY}
                  stroke="#34d399"
                  strokeWidth="1"
                  strokeDasharray="3 3"
                />
                <circle
                  cx={hoveredPt.x}
                  cy={hoveredPt.y}
                  r="7"
                  fill="#34d399"
                  fillOpacity="0.3"
                />
              </g>
            )}
          </svg>

          {/* Interactive Hover Tooltip Box */}
          {hoveredPt && (
            <div
              className="absolute top-3 left-1/2 -translate-x-1/2 bg-slate-900/95 border border-emerald-500/40 rounded-lg px-3 py-1.5 shadow-xl backdrop-blur pointer-events-none flex items-center gap-4 text-xs font-mono"
            >
              <div>
                <span className="text-slate-400">Step {hoveredPt.step}: </span>
                <span className="text-white font-semibold">{hoveredPt.hour_label}</span>
              </div>
              <div className="text-emerald-400 font-bold">
                {hoveredPt.predicted_consumption.toFixed(2)} kWh
              </div>
              <div className="text-slate-300">
                ({hoveredPt.predicted_wh.toFixed(1)} Wh)
              </div>
              <span className={`px-2 py-0.5 rounded text-[10px] font-semibold border ${getLevelColor(hoveredPt.level)}`}>
                {hoveredPt.level}
              </span>
            </div>
          )}
        </div>
      </div>

      {/* 24-Hour Forecast Table */}
      <div className="glass-panel p-5">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <TableIcon className="h-4 w-4 text-emerald-400" />
            <h3 className="text-sm font-bold text-white">Next 24-Hours Granular Forecast Table</h3>
          </div>
          <span className="text-xs text-slate-400">
            Total Day-Ahead Load: <strong className="text-white font-mono">{forecast24.total_predicted_kwh.toFixed(2)} kWh</strong>
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs border-collapse">
            <thead>
              <tr className="border-b border-slate-800 text-slate-400 uppercase tracking-wider font-semibold">
                <th className="py-2.5 px-3">Step</th>
                <th className="py-2.5 px-3">Hour</th>
                <th className="py-2.5 px-3">Predicted Demand (kWh)</th>
                <th className="py-2.5 px-3">Energy (Wh)</th>
                <th className="py-2.5 px-3">Tier</th>
                <th className="py-2.5 px-3">Profile Notes</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 font-mono">
              {predictions.map((p) => {
                const isPeak = p.hour_label === forecast24.peak_hour;
                const isLowest = p.hour_label === forecast24.lowest_hour;

                return (
                  <tr 
                    key={p.step} 
                    className={`hover:bg-slate-800/40 transition-colors ${
                      isPeak ? 'bg-amber-500/5' : isLowest ? 'bg-blue-500/5' : ''
                    }`}
                  >
                    <td className="py-2.5 px-3 text-slate-400">+{p.step}h</td>
                    <td className="py-2.5 px-3 font-semibold text-white">
                      {p.hour_label}
                    </td>
                    <td className="py-2.5 px-3 font-bold text-emerald-400">
                      {p.predicted_consumption.toFixed(2)} kWh
                    </td>
                    <td className="py-2.5 px-3 text-slate-300">
                      {p.predicted_wh.toFixed(1)} Wh
                    </td>
                    <td className="py-2.5 px-3">
                      <span className={`px-2 py-0.5 rounded-full text-[11px] font-sans font-semibold border ${getLevelColor(p.level)}`}>
                        {p.level}
                      </span>
                    </td>
                    <td className="py-2.5 px-3 font-sans text-slate-400">
                      {isPeak && (
                        <span className="flex items-center gap-1 text-amber-400 font-medium">
                          <ArrowUp className="h-3 w-3" /> Projected Peak Load
                        </span>
                      )}
                      {isLowest && (
                        <span className="flex items-center gap-1 text-blue-400 font-medium">
                          <ArrowDown className="h-3 w-3" /> Minimum Base Load
                        </span>
                      )}
                      {!isPeak && !isLowest && 'Standard forecast trajectory'}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
