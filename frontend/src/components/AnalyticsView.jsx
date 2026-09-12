import React, { useEffect, useState } from 'react';
import { ResponsiveContainer, AreaChart, Area, XAxis, YAxis, Tooltip, CartesianGrid } from 'recharts';

export default function AnalyticsView() {
  const [history, setHistory] = useState([]);

  useEffect(() => {
    const fetchAnalytics = () => {
      fetch('http://localhost:5000/analytics')
        .then(res => res.json())
        .then(data => {
          // Format the data array for the chart
          const structuredData = data.map((item, idx) => ({
            id: idx,
            csi: item.csi
          }));
          setHistory(structuredData);
        })
        .catch(err => console.error("History engine lagging: ", err));
    };

    fetchAnalytics();
    const loop = setInterval(fetchAnalytics, 1000); // Fetch new chart data every second
    return () => clearInterval(loop);
  }, []);

  return (
    <div className="bg-slate-900/40 border border-slate-800 rounded-2xl p-6 flex flex-col gap-6 backdrop-blur h-full">
      <div>
        <h2 className="text-lg font-bold tracking-wider text-slate-200">Edge Analytics & Signal Streams</h2>
        <p className="text-xs text-slate-500 font-mono mt-0.5">Real-time dynamic monitoring window</p>
      </div>

      <div className="flex-1 bg-slate-950 p-4 rounded-xl border border-slate-800/80 min-h-[400px]">
        <span className="text-xs uppercase tracking-wider font-bold text-cyan-400 mb-4 block font-mono">
          Stream 01: Cognitive State Index (CSI) Tracker
        </span>
        <div className="h-[300px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={history}>
              <defs>
                <linearGradient id="colorCsi" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#06b6d4" stopOpacity={0.3}/>
                  <stop offset="95%" stopColor="#06b6d4" stopOpacity={0}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="id" stroke="#475569" fontSize={10} />
              <YAxis domain={[0, 100]} stroke="#475569" fontSize={10} />
              <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', color: '#f8fafc' }} />
              <Area type="monotone" dataKey="csi" stroke="#06b6d4" strokeWidth={2} fillOpacity={1} fill="url(#colorCsi)" />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}