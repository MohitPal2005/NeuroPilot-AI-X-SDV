import React, { useState, useEffect } from 'react';
import LiveFeed from './components/LiveFeed';
import CockpitDash from './components/CockpitDash';
import AnalyticsView from './components/AnalyticsView';

export default function App() {
  const [currentTab, setCurrentTab] = useState('dashboard');
  const [telemetry, setTelemetry] = useState({});

  useEffect(() => {
    const handleFetch = () => {
      fetch('http://localhost:5000/driver/status')
        .then(res => res.json())
        .then(data => setTelemetry(data))
        .catch(err => console.error("Backend offline: ", err));
    };
    const pollInterval = setInterval(handleFetch, 100);
    return () => clearInterval(pollInterval);
  }, []);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans">
      <header className="border-b border-slate-800 bg-slate-900/60 backdrop-blur px-8 py-4 flex justify-between items-center">
        <div className="flex items-center gap-3">
          <div className="h-3 w-3 rounded-full bg-cyan-500 animate-pulse" />
          <h1 className="text-xl font-black tracking-widest bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">
            NEUROPILOT AI-X <span className="text-xs border border-cyan-500/40 px-1.5 py-0.5 rounded text-cyan-400">SDV</span>
          </h1>
        </div>
        
        <div className="flex gap-4 bg-slate-950 p-1.5 rounded-lg border border-slate-800">
          <button onClick={() => setCurrentTab('dashboard')} className={`px-6 py-2 rounded-md font-semibold tracking-wide text-sm transition-all duration-300 ${currentTab === 'dashboard' ? 'bg-cyan-600 text-white shadow-lg' : 'text-slate-400 hover:text-slate-200'}`}>
            INTELLIGENT COCKPIT
          </button>
          <button onClick={() => setCurrentTab('analytics')} className={`px-6 py-2 rounded-md font-semibold tracking-wide text-sm transition-all duration-300 ${currentTab === 'analytics' ? 'bg-cyan-600 text-white shadow-lg' : 'text-slate-400 hover:text-slate-200'}`}>
            REAL-TIME ANALYTICS
          </button>
        </div>
      </header>

      <main className="flex-1 p-6 grid grid-cols-12 gap-6 max-w-[1600px] w-full mx-auto">
        <div className="col-span-3 flex flex-col gap-6">
          <LiveFeed telemetry={telemetry} />
        </div>
        <div className="col-span-9">
          {currentTab === 'dashboard' ? <CockpitDash telemetry={telemetry} /> : <AnalyticsView />}
        </div>
      </main>
    </div>
  );
}