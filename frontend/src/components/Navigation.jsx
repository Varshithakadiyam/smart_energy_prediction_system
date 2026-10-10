import React from 'react';
import { 
  LayoutDashboard, 
  UploadCloud,
  Clock, 
  LineChart, 
  AlertTriangle, 
  Lightbulb, 
  Award 
} from 'lucide-react';

const TABS = [
  { id: 'overview', label: 'Overview Dashboard', icon: LayoutDashboard },
  { id: 'user_forecast', label: 'My Energy Forecast', icon: UploadCloud },
  { id: 'forecast24', label: '24-Hour Forecast', icon: Clock },
  { id: 'historical', label: 'Actual vs Predicted', icon: LineChart },
  { id: 'anomalies', label: 'Anomaly Detection', icon: AlertTriangle },
  { id: 'recommendations', label: 'Recommendations', icon: Lightbulb },
  { id: 'benchmarks', label: 'Model Benchmarks', icon: Award }
];

export default function Navigation({ activeTab, setActiveTab }) {
  return (
    <nav className="border-b border-slate-800 bg-slate-900/40 px-4 lg:px-8">
      <div className="max-w-7xl mx-auto flex items-center overflow-x-auto no-scrollbar gap-1 py-2">
        {TABS.map((tab) => {
          const Icon = tab.icon;
          const isActive = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex items-center gap-2 px-3.5 py-2 rounded-lg text-xs font-medium whitespace-nowrap transition-all ${
                isActive
                  ? 'bg-emerald-500/15 text-emerald-400 border border-emerald-500/30 shadow-sm'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60 border border-transparent'
              }`}
            >
              <Icon className={`h-4 w-4 ${isActive ? 'text-emerald-400' : 'text-slate-500'}`} />
              <span>{tab.label}</span>
            </button>
          );
        })}
      </div>
    </nav>
  );
}
