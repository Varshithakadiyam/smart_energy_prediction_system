import React from 'react';
import { 
  Database, 
  UserCheck, 
  Download, 
  UploadCloud, 
  RotateCcw, 
  CheckCircle2, 
  Info, 
  Trash2,
  FileSpreadsheet
} from 'lucide-react';
import { api } from '../api';

export default function ActiveDatasetBar({ 
  datasetMode, 
  setDatasetMode, 
  userValidationReport, 
  userForecastResult, 
  onResetCustomerSession, 
  onNavigateToUpload 
}) {
  const isCustomerMode = datasetMode === 'user';
  const hasCustomerData = Boolean(userValidationReport && userForecastResult);
  const templateUrl = api.getTemplateUrl();

  const handleDownloadTemplate = () => {
    // Direct browser navigation to download endpoint
    const link = document.createElement('a');
    link.href = templateUrl;
    link.setAttribute('download', 'smart_energy_template.csv');
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="bg-slate-900/90 border-b border-slate-800 backdrop-blur px-4 lg:px-8 py-2.5">
      <div className="max-w-7xl mx-auto flex flex-col md:flex-row md:items-center justify-between gap-3">
        {/* Left: Current Active Dataset Badge & Context */}
        <div className="flex items-center gap-3">
          {isCustomerMode ? (
            <div className="flex items-center gap-2">
              <span className="flex h-3 w-3 relative">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
                <span className="relative inline-flex rounded-full h-3 w-3 bg-emerald-500" />
              </span>
              <div className="flex items-center gap-2">
                <span className="text-xs font-bold uppercase tracking-wider text-emerald-400 bg-emerald-500/10 px-2.5 py-1 rounded-md border border-emerald-500/30 flex items-center gap-1.5">
                  <UserCheck className="h-3.5 w-3.5" />
                  Personalized Customer Mode
                </span>
                <span className="text-xs text-slate-300 font-medium truncate max-w-[200px] sm:max-w-xs" title={userValidationReport?.filename}>
                  {userValidationReport?.filename || 'user_data.csv'}
                </span>
                <span className="hidden sm:inline-block text-[11px] font-mono text-slate-400 bg-slate-800/80 px-2 py-0.5 rounded border border-slate-700">
                  {userValidationReport?.accepted_rows || 0} hrs &bull; Univariate GRU
                </span>
              </div>
            </div>
          ) : (
            <div className="flex items-center gap-2">
              <span className="h-2.5 w-2.5 rounded-full bg-blue-400" />
              <div className="flex items-center gap-2">
                <span className="text-xs font-bold uppercase tracking-wider text-blue-400 bg-blue-500/10 px-2.5 py-1 rounded-md border border-blue-500/30 flex items-center gap-1.5">
                  <Database className="h-3.5 w-3.5" />
                  Demo Mode: Reference Dataset
                </span>
                <span className="hidden sm:inline-block text-xs text-slate-400">
                  UCI Individual Household Power (19-Feature Multivariate GRU)
                </span>
              </div>
            </div>
          )}
        </div>

        {/* Right: Actions, Mandatory CSV Template Download, and Mode Toggles */}
        <div className="flex items-center flex-wrap gap-2">
          {/* MANDATORY DOWNLOAD CSV TEMPLATE BUTTON */}
          <button
            onClick={handleDownloadTemplate}
            id="download-csv-template-btn"
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold bg-emerald-600 hover:bg-emerald-500 text-white shadow-sm shadow-emerald-950 transition border border-emerald-400/40"
            title="Download verified customer CSV template (timestamp, energy_consumption_kwh)"
          >
            <Download className="h-3.5 w-3.5" />
            <span>Download CSV Template</span>
          </button>

          {/* Toggle between Customer Data and Reference Benchmark */}
          {hasCustomerData && (
            isCustomerMode ? (
              <button
                onClick={() => setDatasetMode('reference')}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 transition"
                title="Switch back to UCI reference benchmark dataset"
              >
                <Database className="h-3.5 w-3.5 text-blue-400" />
                <span>Switch to Demo Mode</span>
              </button>
            ) : (
              <button
                onClick={() => setDatasetMode('user')}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-emerald-950/80 hover:bg-emerald-900/80 text-emerald-300 border border-emerald-600/50 transition"
                title="Switch to your uploaded customer dataset"
              >
                <UserCheck className="h-3.5 w-3.5 text-emerald-400" />
                <span>Switch to My Customer Data</span>
              </button>
            )
          )}

          {/* Upload New File Action */}
          <button
            onClick={onNavigateToUpload}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 transition"
            title="Upload customer energy consumption CSV file"
          >
            <UploadCloud className="h-3.5 w-3.5 text-cyan-400" />
            <span>{hasCustomerData ? 'Replace CSV' : 'Upload My CSV'}</span>
          </button>

          {/* Purge / Clear Session Button */}
          {hasCustomerData && (
            <button
              onClick={onResetCustomerSession}
              className="flex items-center gap-1 px-2.5 py-1.5 rounded-lg text-xs font-medium text-rose-400 hover:bg-rose-500/10 border border-transparent hover:border-rose-500/20 transition"
              title="Clear customer data from memory and return to demo mode"
            >
              <Trash2 className="h-3.5 w-3.5" />
              <span className="hidden md:inline">Purge</span>
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
