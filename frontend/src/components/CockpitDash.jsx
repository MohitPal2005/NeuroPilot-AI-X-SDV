import React from 'react';

export default function CockpitDash({ telemetry }) {
  const csi = telemetry.csi || 0;
  const isAdaptiveMode = csi > 40; 
  const isEmergencyMode = csi > 80;

  return (
    <div className="h-full flex flex-col gap-6 select-none">
      {csi > 60 && (
        <div className="bg-rose-950/60 border-2 border-rose-600 text-rose-200 px-6 py-4 rounded-xl flex items-center justify-between animate-bounce">
          <div className="flex items-center gap-4">
            <div className="h-4 w-4 bg-rose-600 rounded-full animate-ping" />
            <div>
              <h4 className="font-black tracking-widest text-sm uppercase">CRITICAL COGNITIVE LOAD WARNING</h4>
              <p className="text-xs text-rose-300 mt-0.5">DRIVER DISTRACTION DETECTED</p>
            </div>
          </div>
        </div>
      )}

      <div className="grid grid-cols-3 gap-6">
        <div className={`p-6 rounded-2xl border transition-all duration-500 flex flex-col justify-between aspect-square bg-gradient-to-b ${csi > 80 ? 'from-rose-950/20 to-slate-900/40 border-rose-500/40 shadow-lg' : 'from-slate-900/80 to-slate-950/40 border-slate-800'}`}>
          <div className="flex justify-between items-start text-xs font-bold text-slate-500 uppercase tracking-widest">
            <span>Velocity telemetry</span><span className="font-mono text-cyan-400">km/h</span>
          </div>
          <div className="text-center my-auto">
            <h2 className="text-7xl font-black font-mono tracking-tight text-white">{isEmergencyMode ? "72" : "104"}</h2>
          </div>
        </div>

        <div className={`p-6 rounded-2xl border transition-all duration-500 flex flex-col justify-between aspect-square ${isEmergencyMode ? 'col-span-2 border-rose-500/40 bg-rose-950/5' : 'border-slate-800 bg-slate-900/40'} ${isAdaptiveMode && !isEmergencyMode ? 'col-span-2 border-cyan-500/40' : ''}`}>
          <div className="my-auto flex flex-col justify-center">
            <p className={`font-black tracking-wide text-white transition-all ${isAdaptiveMode ? 'text-3xl text-center text-cyan-300' : 'text-xl'}`}>
              {isAdaptiveMode ? "FOCUS ROAD: EXIT 14 AHEAD" : "Turn left onto NH-44 Expressway"}
            </p>
          </div>
        </div>

        {!isAdaptiveMode && (
          <div className="p-6 rounded-2xl border border-slate-800 bg-slate-900/40 flex flex-col justify-between aspect-square transition-all duration-500">
            <div className="text-xs font-bold text-slate-500 uppercase tracking-widest">EV Battery Array</div>
            <div className="text-center"><span className="text-5xl font-black font-mono text-emerald-400">79%</span></div>
          </div>
        )}
      </div>
    </div>
  );
}