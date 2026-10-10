import React, { useState, useRef } from 'react';
import { 
  UploadCloud, 
  FileText, 
  Download, 
  CheckCircle2, 
  AlertTriangle, 
  Clock, 
  TrendingUp, 
  Flame, 
  ShieldCheck, 
  RefreshCw, 
  ArrowRight, 
  Trash2, 
  Table as TableIcon,
  Lightbulb,
  Zap,
  ChevronDown,
  ChevronUp,
  Eye,
  BarChart3,
  LineChart
} from 'lucide-react';
import { api } from '../api';

export default function UserForecastView({ 
  onUserUploadSuccess, 
  onResetCustomerSession, 
  activeValidationReport, 
  activeForecastResult,
  datasetMode,
  onSwitchTab
}) {
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [validationReport, setValidationReport] = useState(activeValidationReport || null);
  const [uploadError, setUploadError] = useState(null);

  // Historical preview toggles
  const [showPreviewTable, setShowPreviewTable] = useState(true);
  const [showHistoricalChart, setShowHistoricalChart] = useState(true);

  // Forecasting state
  const [horizon, setHorizon] = useState('24h');
  const [generating, setGenerating] = useState(false);
  const [forecastResult, setForecastResult] = useState(activeForecastResult || null);
  const [forecastError, setForecastError] = useState(null);

  const fileInputRef = useRef(null);

  // Keep synced with parent active state
  React.useEffect(() => {
    if (activeValidationReport) setValidationReport(activeValidationReport);
    if (activeForecastResult) setForecastResult(activeForecastResult);
  }, [activeValidationReport, activeForecastResult]);

  // Drag and drop handlers
  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const droppedFile = e.dataTransfer.files[0];
      if (droppedFile.name.toLowerCase().endsWith('.csv')) {
        setFile(droppedFile);
        setUploadError(null);
      } else {
        setUploadError("Invalid file type. Please upload a .csv file.");
      }
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      setFile(e.target.files[0]);
      setUploadError(null);
    }
  };

  // Upload, Validate, and Auto-hydrate Forecast
  const handleUpload = async () => {
    if (!file) return;
    setUploading(true);
    setUploadError(null);
    setForecastResult(null);
    setForecastError(null);

    try {
      const report = await api.uploadUserCsv(file);
      setValidationReport(report);

      // Immediately generate 24h forecast to hydrate the full dashboard
      setGenerating(true);
      try {
        const initialForecast = await api.predictUserForecast(report.session_id, '24h');
        setForecastResult(initialForecast);
        if (onUserUploadSuccess) {
          onUserUploadSuccess(report, initialForecast);
        }
      } catch (fErr) {
        console.warn("Initial forecast warning:", fErr);
        if (onUserUploadSuccess) {
          onUserUploadSuccess(report, null);
        }
      } finally {
        setGenerating(false);
      }
    } catch (err) {
      console.error("Upload error:", err);
      setUploadError(err.message || "Failed to process and validate CSV file.");
    } finally {
      setUploading(false);
    }
  };

  // Generate Forecast on button click
  const handleGenerateForecast = async () => {
    if (!validationReport?.session_id) return;
    setGenerating(true);
    setForecastError(null);

    try {
      const res = await api.predictUserForecast(validationReport.session_id, horizon);
      setForecastResult(res);
      if (onUserUploadSuccess) {
        onUserUploadSuccess(validationReport, res);
      }
    } catch (err) {
      console.error("Forecast error:", err);
      setForecastError(err.message || "Failed to generate forecast.");
    } finally {
      setGenerating(false);
    }
  };

  // Download forecast CSV
  const handleDownloadForecastCsv = () => {
    if (!forecastResult?.predictions) return;
    const header = "step,timestamp,hour_label,predicted_consumption_kwh,predicted_wh,level\n";
    const rows = forecastResult.predictions.map(p => 
      `${p.step},"${p.timestamp}","${p.hour_label}",${p.predicted_consumption},${p.predicted_wh},"${p.level}"`
    ).join("\n");
    const blob = new Blob([header + rows], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `forecast_${horizon}_${validationReport.filename || 'user_data'}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  // Reset session
  const handleReset = async () => {
    if (validationReport?.session_id) {
      try {
        await api.resetUserSession(validationReport.session_id);
      } catch (e) {
        // ignore reset failure
      }
    }
    setFile(null);
    setValidationReport(null);
    setForecastResult(null);
    setUploadError(null);
    setForecastError(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
    if (onResetCustomerSession) {
      onResetCustomerSession();
    }
  };

  // Level badge styling
  const getLevelBadge = (level) => {
    switch (level?.toLowerCase()) {
      case 'low': return 'text-blue-400 bg-blue-500/10 border-blue-500/30';
      case 'normal': return 'text-emerald-400 bg-emerald-500/10 border-emerald-500/30';
      case 'high': return 'text-amber-400 bg-amber-500/10 border-amber-500/30';
      case 'unusually high': return 'text-rose-400 bg-rose-500/10 border-rose-500/30';
      default: return 'text-slate-400 bg-slate-500/10 border-slate-500/30';
    }
  };

  // SVG Chart rendering for User Forecast
  const predictions = forecastResult?.predictions || [];
  const maxVal = predictions.length > 0 ? Math.max(...predictions.map(p => p.predicted_consumption), 3.0) * 1.2 : 4.0;
  const minVal = 0;
  const svgWidth = 800;
  const svgHeight = 220;
  const paddingX = 40;
  const paddingY = 25;
  const chartW = svgWidth - paddingX * 2;
  const chartH = svgHeight - paddingY * 2;

  const points = predictions.map((p, i) => {
    const x = paddingX + (i / Math.max(1, predictions.length - 1)) * chartW;
    const y = svgHeight - paddingY - ((p.predicted_consumption - minVal) / (maxVal - minVal)) * chartH;
    return { x, y, ...p };
  });

  const pathD = points.reduce((acc, pt, i) => (
    i === 0 ? `M ${pt.x} ${pt.y}` : `${acc} L ${pt.x} ${pt.y}`
  ), '');

  const areaD = points.length > 1
    ? `${pathD} L ${points[points.length - 1].x} ${svgHeight - paddingY} L ${points[0].x} ${svgHeight - paddingY} Z`
    : '';

  // SVG Chart rendering for Uploaded Historical Sequence
  const recentHistory = validationReport?.recent_history || validationReport?.preview_data || [];
  const histMaxVal = recentHistory.length > 0 ? Math.max(...recentHistory.map(h => h.energy_consumption_kwh), 2.5) * 1.2 : 4.0;
  const histPoints = recentHistory.map((h, i) => {
    const x = paddingX + (i / Math.max(1, recentHistory.length - 1)) * chartW;
    const y = svgHeight - paddingY - ((h.energy_consumption_kwh - minVal) / (histMaxVal - minVal)) * chartH;
    return { x, y, ...h };
  });

  const histPathD = histPoints.reduce((acc, pt, i) => (
    i === 0 ? `M ${pt.x} ${pt.y}` : `${acc} L ${pt.x} ${pt.y}`
  ), '');

  const histAreaD = histPoints.length > 1
    ? `${histPathD} L ${histPoints[histPoints.length - 1].x} ${svgHeight - paddingY} L ${histPoints[0].x} ${svgHeight - paddingY} Z`
    : '';

  return (
    <div className="space-y-6">
      {/* Header Banner */}
      <div className="glass-panel p-5">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2">
              <UploadCloud className="h-5 w-5 text-emerald-400" />
              <h2 className="text-base font-bold text-white">My Energy Forecast — User CSV Forecasting</h2>
            </div>
            <p className="text-xs text-slate-300 mt-1 max-w-3xl leading-relaxed">
              Upload your own historical electricity consumption data to generate personalized next-hour and 24-hour neural forecasts. Evaluated using a dedicated Univariate GRU model with zero simulated data.
            </p>
          </div>
          <div className="flex items-center gap-3 shrink-0">
            <a
              href={api.getTemplateUrl()}
              download="energy_consumption_template.csv"
              className="flex items-center gap-2 px-3 py-2 rounded-lg text-xs font-medium bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 transition"
            >
              <Download className="h-4 w-4 text-emerald-400" />
              <span>Download CSV Template</span>
            </a>
          </div>
        </div>
      </div>

      {/* Step 1: Upload Zone (if no validation report yet) */}
      {!validationReport && (
        <div className="glass-panel p-6">
          <div
            onDragOver={handleDragOver}
            onDrop={handleDrop}
            className="border-2 border-dashed border-slate-700 hover:border-emerald-500/60 rounded-2xl p-8 text-center transition-all bg-slate-950/40 flex flex-col items-center justify-center cursor-pointer"
            onClick={() => fileInputRef.current?.click()}
          >
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileChange}
              accept=".csv"
              className="hidden"
            />
            <div className="h-14 w-14 rounded-2xl bg-emerald-500/10 border border-emerald-500/30 flex items-center justify-center text-emerald-400 mb-3 shadow-lg shadow-emerald-500/10">
              <UploadCloud className="h-7 w-7" />
            </div>
            <h3 className="text-sm font-bold text-white">
              {file ? file.name : "Drag & Drop your electricity consumption CSV here"}
            </h3>
            <p className="text-xs text-slate-400 mt-1 max-w-md">
              {file 
                ? `Size: ${(file.size / 1024).toFixed(1)} KB — Click "Process & Validate CSV" below`
                : "Supports hourly timestamp and energy_consumption_kwh columns (minimum 24 consecutive hours required)"}
            </p>
            <button
              type="button"
              className="mt-4 px-4 py-2 rounded-lg text-xs font-semibold bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 transition"
              onClick={(e) => {
                e.stopPropagation();
                fileInputRef.current?.click();
              }}
            >
              Browse Files
            </button>
          </div>

          {uploadError && (
            <div className="mt-4 bg-rose-500/10 border border-rose-500/30 rounded-xl p-3 text-xs text-rose-300 flex items-center gap-2">
              <AlertTriangle className="h-4 w-4 shrink-0 text-rose-400" />
              <span>{uploadError}</span>
            </div>
          )}

          {file && (
            <div className="mt-4 flex items-center justify-between pt-2 border-t border-slate-800">
              <div className="flex items-center gap-2 text-xs text-slate-300">
                <FileText className="h-4 w-4 text-emerald-400" />
                <span className="font-semibold">{file.name}</span>
                <span className="text-slate-500">({(file.size / 1024).toFixed(1)} KB)</span>
              </div>
              <button
                onClick={handleUpload}
                disabled={uploading}
                className="flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-bold bg-emerald-600 hover:bg-emerald-500 text-white transition disabled:opacity-50"
              >
                {uploading ? (
                  <>
                    <RefreshCw className="h-3.5 w-3.5 animate-spin" />
                    <span>Validating CSV...</span>
                  </>
                ) : (
                  <>
                    <span>Process & Validate CSV</span>
                    <ArrowRight className="h-3.5 w-3.5" />
                  </>
                )}
              </button>
            </div>
          )}
        </div>
      )}

      {/* Step 2: Validated Data Overview & Personal Analytics */}
      {validationReport && (
        <div className="space-y-6">
          {/* Validation Status Card */}
          <div className="glass-panel p-5">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-4 border-b border-slate-800">
              <div className="flex items-center gap-3">
                <div className="h-10 w-10 rounded-xl bg-emerald-500/15 border border-emerald-500/30 flex items-center justify-center text-emerald-400">
                  <CheckCircle2 className="h-5 w-5" />
                </div>
                <div>
                  <div className="flex items-center gap-2">
                    <h3 className="text-sm font-bold text-white">{validationReport.filename}</h3>
                    <span className="px-2 py-0.5 rounded-full text-[10px] font-semibold bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
                      Verified Valid
                    </span>
                  </div>
                  <p className="text-xs text-slate-400 font-mono mt-0.5">
                    Timeline: {validationReport.start_timestamp.replace('T', ' ')} &rarr; {validationReport.end_timestamp.replace('T', ' ')}
                  </p>
                </div>
              </div>

              <button
                onClick={handleReset}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 transition"
              >
                <Trash2 className="h-3.5 w-3.5 text-rose-400" />
                <span>Replace Dataset</span>
              </button>
            </div>

            {/* Quality Summary Grid */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-4 text-xs font-mono">
              <div className="bg-slate-950/60 p-2.5 rounded-lg border border-slate-800">
                <span className="text-slate-400 text-[11px] font-sans">Accepted Rows</span>
                <div className="text-base font-bold text-white mt-0.5">{validationReport.accepted_rows}</div>
              </div>
              <div className="bg-slate-950/60 p-2.5 rounded-lg border border-slate-800">
                <span className="text-slate-400 text-[11px] font-sans">Sampling Rate</span>
                <div className="text-base font-bold text-white mt-0.5">{validationReport.sampling_frequency}</div>
              </div>
              <div className="bg-slate-950/60 p-2.5 rounded-lg border border-slate-800">
                <span className="text-slate-400 text-[11px] font-sans">Imputed Hours</span>
                <div className="text-base font-bold text-emerald-400 mt-0.5">{validationReport.imputed_count}</div>
              </div>
              <div className="bg-slate-950/60 p-2.5 rounded-lg border border-slate-800">
                <span className="text-slate-400 text-[11px] font-sans">Routing Target</span>
                <div className="text-xs font-bold text-blue-400 truncate mt-1">{validationReport.compatible_model}</div>
              </div>
            </div>
          </div>

          {/* Personal Analytics KPI Cards */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="glass-panel p-4">
              <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Total Recorded</span>
              <div className="text-2xl font-extrabold text-white mt-1">
                {validationReport.personal_analytics.total_consumption_kwh.toFixed(2)}{" "}
                <span className="text-xs font-medium text-emerald-400">kWh</span>
              </div>
              <div className="text-xs text-slate-500 mt-1">Across {validationReport.personal_analytics.total_days_coverage} days</div>
            </div>

            <div className="glass-panel p-4">
              <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Hourly Average</span>
              <div className="text-2xl font-extrabold text-white mt-1">
                {validationReport.personal_analytics.average_hourly_kwh.toFixed(3)}{" "}
                <span className="text-xs font-medium text-blue-400">kWh/h</span>
              </div>
              <div className="text-xs text-slate-500 mt-1">Daily Avg: {validationReport.personal_analytics.daily_average_kwh.toFixed(2)} kWh</div>
            </div>

            <div className="glass-panel p-4">
              <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Historical Peak</span>
              <div className="text-2xl font-extrabold text-amber-400 mt-1">
                {validationReport.personal_analytics.peak_historical_hour}
              </div>
              <div className="text-xs text-slate-500 mt-1">Lowest: {validationReport.personal_analytics.lowest_historical_hour}</div>
            </div>

            <div className="glass-panel p-4">
              <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Demand Range</span>
              <div className="text-lg font-bold text-white mt-1">
                {validationReport.personal_analytics.min_hourly_kwh.toFixed(2)} &ndash; {validationReport.personal_analytics.max_hourly_kwh.toFixed(2)}{" "}
                <span className="text-xs text-slate-400">kWh</span>
              </div>
              <div className="text-xs text-slate-500 mt-1">Dynamic load variation</div>
            </div>
          </div>

          {/* Historical Data Preview Table */}
          <div className="glass-panel p-5">
            <div className="flex items-center justify-between cursor-pointer select-none" onClick={() => setShowPreviewTable(!showPreviewTable)}>
              <div className="flex items-center gap-2">
                <TableIcon className="h-4 w-4 text-emerald-400" />
                <h3 className="text-sm font-bold text-white">Uploaded Data Preview</h3>
                <span className="ml-2 px-2 py-0.5 rounded text-[10px] font-semibold bg-slate-800 text-slate-300 border border-slate-700">
                  {validationReport.preview_data?.length || 0} Sampled Records ({validationReport.accepted_rows} Total)
                </span>
              </div>
              <button
                type="button"
                className="flex items-center gap-1 text-xs text-slate-400 hover:text-slate-200 transition"
              >
                <span>{showPreviewTable ? "Hide Preview" : "Show Preview"}</span>
                {showPreviewTable ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
              </button>
            </div>

            {showPreviewTable && (
              <div className="mt-3 overflow-x-auto">
                <table className="w-full text-left text-xs border-collapse">
                  <thead>
                    <tr className="border-b border-slate-800 text-slate-400 uppercase tracking-wider font-semibold">
                      <th className="py-2.5 px-3">Position</th>
                      <th className="py-2.5 px-3">Timestamp</th>
                      <th className="py-2.5 px-3">Hour Label</th>
                      <th className="py-2.5 px-3">Energy (kWh)</th>
                      <th className="py-2.5 px-3">Energy (Wh)</th>
                      <th className="py-2.5 px-3">Data Status</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/60 font-mono">
                    {validationReport.preview_data.map((row, idx) => {
                      const isHead = idx < 5;
                      const posLabel = isHead ? `Start #${idx + 1}` : `End #${validationReport.preview_data.length - idx}`;
                      const whVal = (row.energy_consumption_kwh * 1000).toFixed(1);
                      return (
                        <tr key={idx} className="hover:bg-slate-800/40 transition">
                          <td className="py-2 px-3 text-slate-400 font-sans">
                            <span className={`px-2 py-0.5 rounded text-[10px] font-semibold border ${
                              isHead 
                                ? 'bg-blue-500/10 text-blue-400 border-blue-500/20' 
                                : 'bg-purple-500/10 text-purple-400 border-purple-500/20'
                            }`}>
                              {posLabel}
                            </span>
                          </td>
                          <td className="py-2 px-3 text-white">{row.timestamp.replace('T', ' ')}</td>
                          <td className="py-2 px-3 text-slate-300 font-sans">{row.hour_label}</td>
                          <td className="py-2 px-3 font-bold text-emerald-400">{row.energy_consumption_kwh.toFixed(3)}</td>
                          <td className="py-2 px-3 text-slate-400">{whVal}</td>
                          <td className="py-2 px-3 font-sans">
                            <span className="text-emerald-400 text-[11px] font-medium flex items-center gap-1">
                              <CheckCircle2 className="h-3 w-3" /> Validated
                            </span>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </div>

          {/* Historical Consumption Sequence Chart */}
          {recentHistory.length > 0 && (
            <div className="glass-panel p-5">
              <div className="flex items-center justify-between cursor-pointer select-none" onClick={() => setShowHistoricalChart(!showHistoricalChart)}>
                <div className="flex items-center gap-2">
                  <LineChart className="h-4 w-4 text-sky-400" />
                  <h3 className="text-sm font-bold text-white">Historical Consumption Timeline (Recent History)</h3>
                  <span className="ml-2 px-2 py-0.5 rounded text-[10px] font-semibold bg-sky-500/10 text-sky-300 border border-sky-500/20">
                    Past {recentHistory.length} Hours
                  </span>
                </div>
                <button
                  type="button"
                  className="flex items-center gap-1 text-xs text-slate-400 hover:text-slate-200 transition"
                >
                  <span>{showHistoricalChart ? "Hide Chart" : "Show Chart"}</span>
                  {showHistoricalChart ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
                </button>
              </div>

              {showHistoricalChart && (
                <div className="mt-4 space-y-4">
                  <p className="text-xs text-slate-400">
                    Chronological consumption curve feeding the sliding 24-hour neural forecast sequence.
                  </p>

                  <div className="relative w-full overflow-hidden bg-slate-950/40 rounded-xl p-2 border border-slate-800/60">
                    <svg viewBox={`0 0 ${svgWidth} ${svgHeight}`} className="w-full h-auto select-none">
                      <defs>
                        <linearGradient id="userHistGrad" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="0%" stopColor="#0284c7" stopOpacity="0.4" />
                          <stop offset="85%" stopColor="#0284c7" stopOpacity="0.0" />
                        </linearGradient>
                      </defs>

                      {/* Horizontal Grid */}
                      {[0, 1, 2, 3].map((g) => {
                        if (g > histMaxVal) return null;
                        const y = svgHeight - paddingY - ((g - minVal) / (histMaxVal - minVal)) * chartH;
                        return (
                          <g key={g}>
                            <line x1={paddingX} y1={y} x2={svgWidth - paddingX} y2={y} stroke="#1e293b" strokeDasharray="3 3" />
                            <text x={paddingX - 8} y={y + 3} fill="#64748b" fontSize="9" fontFamily="monospace" textAnchor="end">{g}k</text>
                          </g>
                        );
                      })}

                      {/* Area fill */}
                      {histAreaD && <path d={histAreaD} fill="url(#userHistGrad)" />}

                      {/* Line */}
                      {histPathD && <path d={histPathD} fill="none" stroke="#38bdf8" strokeWidth="2.5" strokeLinecap="round" />}

                      {/* Points */}
                      {histPoints.map((pt, idx) => {
                        const isMax = pt.energy_consumption_kwh === Math.max(...recentHistory.map(h => h.energy_consumption_kwh));
                        return (
                          <g key={idx}>
                            <circle
                              cx={pt.x}
                              cy={pt.y}
                              r={isMax ? 5 : 3}
                              fill={isMax ? '#f59e0b' : '#38bdf8'}
                              stroke="#0b0f19"
                              strokeWidth="1.5"
                            />
                            {idx % 3 === 0 && (
                              <text
                                x={pt.x}
                                y={svgHeight - 8}
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
                    </svg>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Forecast Controls Bar */}
          <div className="glass-panel p-5 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
            <div>
              <h3 className="text-sm font-bold text-white">Generate User-Specific Forecast</h3>
              <p className="text-xs text-slate-400 mt-0.5">
                Uses the latest 24 hours of your uploaded dataset as sliding inference input
              </p>
            </div>

            <div className="flex items-center gap-3">
              {/* Horizon Selector */}
              <div className="flex items-center bg-slate-950/60 p-1 rounded-lg border border-slate-800 text-xs">
                <button
                  onClick={() => setHorizon('1h')}
                  className={`px-3 py-1.5 rounded-md font-medium transition ${
                    horizon === '1h'
                      ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                      : 'text-slate-400 hover:text-slate-200'
                  }`}
                >
                  Next Hour (1h)
                </button>
                <button
                  onClick={() => setHorizon('24h')}
                  className={`px-3 py-1.5 rounded-md font-medium transition ${
                    horizon === '24h'
                      ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                      : 'text-slate-400 hover:text-slate-200'
                  }`}
                >
                  Next 24 Hours (Day-Ahead)
                </button>
              </div>

              <button
                onClick={handleGenerateForecast}
                disabled={generating}
                className="flex items-center gap-2 px-5 py-2 rounded-lg text-xs font-bold bg-emerald-600 hover:bg-emerald-500 text-white transition disabled:opacity-50"
              >
                {generating ? (
                  <>
                    <RefreshCw className="h-3.5 w-3.5 animate-spin" />
                    <span>Predicting...</span>
                  </>
                ) : (
                  <>
                    <Zap className="h-3.5 w-3.5 fill-current" />
                    <span>Generate Forecast</span>
                  </>
                )}
              </button>
            </div>
          </div>

          {forecastError && (
            <div className="bg-rose-500/10 border border-rose-500/30 rounded-xl p-3 text-xs text-rose-300 flex items-center gap-2">
              <AlertTriangle className="h-4 w-4 text-rose-400" />
              <span>{forecastError}</span>
            </div>
          )}

          {/* Forecast Results View */}
          {forecastResult && (
            <div className="space-y-6">
              {/* Notification Banner that Personalized Mode is Active Across Tabs */}
              <div className="bg-emerald-500/10 border border-emerald-500/30 rounded-xl p-4 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                <div className="flex items-center gap-3">
                  <div className="p-2 rounded-lg bg-emerald-500/20 text-emerald-400">
                    <CheckCircle2 className="h-5 w-5" />
                  </div>
                  <div>
                    <h4 className="text-xs font-bold text-white uppercase tracking-wider">
                      Personalized Mode Active Across Entire Dashboard
                    </h4>
                    <p className="text-xs text-slate-300 mt-0.5">
                      Overview KPIs, 24h Forecast, Historical Trends, Anomaly Monitoring, and Recommendations are now powered by {validationReport?.filename || 'your uploaded CSV'}.
                    </p>
                  </div>
                </div>
                {onSwitchTab && (
                  <div className="flex items-center flex-wrap gap-2">
                    <button
                      onClick={() => onSwitchTab('overview')}
                      className="px-3 py-1.5 rounded-lg text-xs font-semibold bg-emerald-600 hover:bg-emerald-500 text-white transition flex items-center gap-1 shadow-sm"
                    >
                      <span>Overview</span>
                      <ArrowRight className="h-3 w-3" />
                    </button>
                    <button
                      onClick={() => onSwitchTab('forecast24')}
                      className="px-3 py-1.5 rounded-lg text-xs font-medium bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 transition"
                    >
                      24h Forecast
                    </button>
                    <button
                      onClick={() => onSwitchTab('anomalies')}
                      className="px-3 py-1.5 rounded-lg text-xs font-medium bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 transition"
                    >
                      Anomalies
                    </button>
                    <button
                      onClick={() => onSwitchTab('recommendations')}
                      className="px-3 py-1.5 rounded-lg text-xs font-medium bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 transition"
                    >
                      Tips
                    </button>
                  </div>
                )}
              </div>

              {/* Forecast SVG Chart Card */}
              <div className="glass-panel p-5">
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4">
                  <div>
                    <h3 className="text-base font-bold text-white">
                      Predicted Consumption Trajectory ({forecastResult.horizon.toUpperCase()})
                    </h3>
                    <p className="text-xs text-slate-400 mt-0.5">
                      Model: <span className="text-blue-400 font-mono">{forecastResult.model_used}</span>
                    </p>
                  </div>
                  <div className="flex items-center gap-3">
                    <button
                      onClick={handleDownloadForecastCsv}
                      className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 transition"
                    >
                      <Download className="h-3.5 w-3.5 text-emerald-400" />
                      <span>Download Forecast CSV</span>
                    </button>
                  </div>
                </div>

                {/* SVG Area Chart */}
                <div className="relative w-full overflow-hidden bg-slate-950/40 rounded-xl p-2 border border-slate-800/60">
                  <svg viewBox={`0 0 ${svgWidth} ${svgHeight}`} className="w-full h-auto select-none">
                    <defs>
                      <linearGradient id="userForecastGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor="#10b981" stopOpacity="0.4" />
                        <stop offset="85%" stopColor="#10b981" stopOpacity="0.0" />
                      </linearGradient>
                    </defs>

                    {/* Horizontal Grid */}
                    {[0, 1, 2, 3].map((g) => {
                      if (g > maxVal) return null;
                      const y = svgHeight - paddingY - ((g - minVal) / (maxVal - minVal)) * chartH;
                      return (
                        <g key={g}>
                          <line x1={paddingX} y1={y} x2={svgWidth - paddingX} y2={y} stroke="#1e293b" strokeDasharray="3 3" />
                          <text x={paddingX - 8} y={y + 3} fill="#64748b" fontSize="9" fontFamily="monospace" textAnchor="end">{g}k</text>
                        </g>
                      );
                    })}

                    {/* Area fill */}
                    {areaD && <path d={areaD} fill="url(#userForecastGrad)" />}

                    {/* Line */}
                    {pathD && <path d={pathD} fill="none" stroke="#10b981" strokeWidth="2.5" strokeLinecap="round" />}

                    {/* Points */}
                    {points.map((pt, idx) => {
                      const isPeak = pt.hour_label === forecastResult.peak_hour;
                      return (
                        <g key={idx}>
                          <circle
                            cx={pt.x}
                            cy={pt.y}
                            r={isPeak ? 5 : 3.5}
                            fill={isPeak ? '#f59e0b' : '#10b981'}
                            stroke="#0b0f19"
                            strokeWidth="1.5"
                          />
                          {idx % 3 === 0 && (
                            <text
                              x={pt.x}
                              y={svgHeight - 8}
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
                  </svg>
                </div>

                {/* Summary Highlight Cards */}
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-4 text-xs">
                  <div className="bg-slate-950/60 p-3 rounded-lg border border-slate-800">
                    <span className="text-slate-400">Total Forecasted</span>
                    <div className="text-lg font-bold text-emerald-400 mt-0.5">
                      {forecastResult.total_predicted_kwh.toFixed(2)} kWh
                    </div>
                  </div>
                  <div className="bg-slate-950/60 p-3 rounded-lg border border-slate-800">
                    <span className="text-slate-400">Projected Peak</span>
                    <div className="text-lg font-bold text-amber-400 mt-0.5">
                      {forecastResult.peak_hour} ({forecastResult.peak_consumption.toFixed(2)} kWh)
                    </div>
                  </div>
                  <div className="bg-slate-950/60 p-3 rounded-lg border border-slate-800">
                    <span className="text-slate-400">Lowest Load</span>
                    <div className="text-lg font-bold text-blue-400 mt-0.5">
                      {forecastResult.lowest_hour} ({forecastResult.lowest_consumption.toFixed(2)} kWh)
                    </div>
                  </div>
                  <div className="bg-slate-950/60 p-3 rounded-lg border border-slate-800">
                    <span className="text-slate-400">Forecast Steps</span>
                    <div className="text-lg font-bold text-white mt-0.5">
                      {forecastResult.predictions.length} Hours
                    </div>
                  </div>
                </div>
              </div>

              {/* Forecast Table */}
              <div className="glass-panel p-5">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <TableIcon className="h-4 w-4 text-emerald-400" />
                    <h3 className="text-sm font-bold text-white">Detailed Hourly Forecast Breakdown</h3>
                  </div>
                  <span className="text-xs text-slate-400 font-mono">
                    Session ID: {forecastResult.session_id.slice(0, 8)}...
                  </span>
                </div>
                <div className="overflow-x-auto">
                  <table className="w-full text-left text-xs border-collapse">
                    <thead>
                      <tr className="border-b border-slate-800 text-slate-400 uppercase tracking-wider font-semibold">
                        <th className="py-2.5 px-3">Step</th>
                        <th className="py-2.5 px-3">Timestamp</th>
                        <th className="py-2.5 px-3">Forecast (kWh)</th>
                        <th className="py-2.5 px-3">Energy (Wh)</th>
                        <th className="py-2.5 px-3">Demand Tier</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800/60 font-mono">
                      {forecastResult.predictions.map((p) => (
                        <tr key={p.step} className="hover:bg-slate-800/40 transition">
                          <td className="py-2 px-3 text-slate-400">+{p.step}h</td>
                          <td className="py-2 px-3 text-white">{p.timestamp.replace('T', ' ')}</td>
                          <td className="py-2 px-3 font-bold text-emerald-400">{p.predicted_consumption.toFixed(3)}</td>
                          <td className="py-2 px-3 text-slate-300">{p.predicted_wh.toFixed(1)}</td>
                          <td className="py-2 px-3 font-sans">
                            <span className={`px-2 py-0.5 rounded-full text-[11px] font-semibold border ${getLevelBadge(p.level)}`}>
                              {p.level}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* User Anomaly Report Panel */}
              {forecastResult.anomalies && (
                <div className="glass-panel p-5">
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-2">
                      <AlertTriangle className="h-4 w-4 text-amber-400" />
                      <h3 className="text-sm font-bold text-white">Recent Observation Anomaly Evaluation</h3>
                    </div>
                    <span className="text-xs text-slate-400">
                      {forecastResult.anomalies.anomalies_detected} Anomalies in Recent 24h
                    </span>
                  </div>
                  <p className="text-xs text-slate-400 mb-3">
                    Calculated using personal diurnal baseline (mean_hour ± 2.5 × std_hour) across your uploaded data.
                  </p>
                  <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2">
                    {forecastResult.anomalies.results.slice(-6).map((item, idx) => (
                      <div key={idx} className={`p-2.5 rounded-lg border text-xs font-mono ${
                        item.is_anomaly 
                          ? 'bg-rose-500/10 border-rose-500/30 text-rose-300' 
                          : 'bg-slate-950/60 border-slate-800 text-slate-300'
                      }`}>
                        <div className="flex justify-between font-sans text-slate-400">
                          <span>{item.hour_label}</span>
                          <span>Z = {item.z_score > 0 ? `+${item.z_score}σ` : `${item.z_score}σ`}</span>
                        </div>
                        <div className="font-bold text-white mt-1">
                          Actual: {item.actual_consumption_kwh.toFixed(2)} kWh &nbsp;|&nbsp; Exp: {item.expected_kwh.toFixed(2)} kWh
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Personalized Recommendations Panel */}
              {forecastResult.recommendations && (
                <div className="glass-panel p-5">
                  <div className="flex items-center gap-2 mb-3">
                    <Lightbulb className="h-4 w-4 text-emerald-400" />
                    <h3 className="text-sm font-bold text-white">Personalized Demand-Side Recommendations</h3>
                  </div>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                    {forecastResult.recommendations.map((rec, idx) => (
                      <div key={idx} className="bg-slate-950/60 p-3.5 rounded-xl border border-slate-800 space-y-1.5">
                        <div className="flex items-center justify-between">
                          <span className="text-[11px] font-semibold text-emerald-400 uppercase tracking-wider">{rec.category}</span>
                          <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-amber-500/20 text-amber-300 border border-amber-500/30">
                            {rec.priority}
                          </span>
                        </div>
                        <h4 className="text-xs font-bold text-white">{rec.title}</h4>
                        <p className="text-xs text-slate-300 leading-relaxed">{rec.recommendation}</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
