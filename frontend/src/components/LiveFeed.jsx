import React from 'react';

export default function LiveFeed({ telemetry }) {
  const getRiskColor = (zone) => {
    if (zone === 'Safe') return 'text-emerald-400 border-emerald-500/30 bg-emerald-500/5';
    if (zone === 'Attention Required' || zone === 'Cognitive Overload') return 'text-amber-400 border-amber-500/30 bg-amber-500/5';
    if (zone === 'No Data (Calibrating)') return 'text-slate-400 border-slate-500/30 bg-slate-500/5';
    return 'text-rose-400 border-rose-500/30 bg-rose-500/5';
  };

  const sensorStatus = telemetry.sensor_status || { visual: 'active', acoustic: 'active', kinematic: 'active' };
  const activeCount = Object.values(sensorStatus).filter(s => s === 'active').length;
  const totalCount = Object.keys(sensorStatus).length;

  return (
    <div className="bg-slate-900/40 border border-slate-800/80 rounded-2xl p-4 flex flex-col gap-4 backdrop-blur-md">
      <div className="border-b border-slate-800 pb-2 flex flex-col gap-1.5">
        <h3 className="text-xs font-bold uppercase tracking-widest text-slate-400">
          In-Cabin Sensor Matrix
        </h3>
        <div className="flex justify-between text-[9px] uppercase tracking-wider font-mono">
          <span className={sensorStatus.visual === 'active' ? 'text-emerald-400' : 'text-rose-400'}>Vis: {sensorStatus.visual}</span>
          <span className={sensorStatus.acoustic === 'active' ? 'text-emerald-400' : 'text-rose-400'}>Aud: {sensorStatus.acoustic}</span>
          <span className={sensorStatus.kinematic === 'active' ? 'text-emerald-400' : (sensorStatus.kinematic === 'buffering' ? 'text-amber-400' : 'text-rose-400')}>Kin: {sensorStatus.kinematic}</span>
        </div>
      </div>
      
      {/* --- UPGRADED LIVE VIDEO FEED --- */}
      <div className="relative aspect-video bg-slate-950 rounded-xl overflow-hidden border border-slate-800 flex items-center justify-center">
        {/* Pulls the live JPEG stream directly from the Flask server */}
        <img 
          src="http://localhost:5000/video_feed" 
          alt="Live Camera Feed" 
          className="w-full h-full object-cover"
        />
        
        {telemetry.face_detected && (
          <div className="absolute top-3 left-3 bg-cyan-950/80 border border-cyan-500/40 text-[10px] text-cyan-400 px-2 py-1 rounded font-mono uppercase tracking-wider animate-pulse shadow-lg">
            Tracking Active
          </div>
        )}
      </div>
      {/* -------------------------------- */}

      <div className="grid grid-cols-2 gap-2 font-mono text-xs text-slate-300">
        <div className="bg-slate-950 p-2.5 rounded-lg border border-slate-800/60">
          <span className="text-[10px] uppercase text-slate-500 block mb-0.5">Gaze Orientation</span>
          <span className="font-bold text-cyan-400 text-sm">{telemetry.gaze_direction || 'N/A'}</span>
        </div>
        <div className="bg-slate-950 p-2.5 rounded-lg border border-slate-800/60">
          <span className="text-[10px] uppercase text-slate-500 block mb-0.5">Eye State (EAR)</span>
          <span className="font-bold text-cyan-400 text-sm">{telemetry.ear || '0.00'}</span>
        </div>
      </div>

      <div className={`mt-2 border rounded-xl p-3.5 text-center transition-all duration-300 ${getRiskColor(telemetry.zone)}`}>
        <div className="text-[10px] tracking-widest uppercase font-bold opacity-70 mb-1">Cognitive Evaluation</div>
        <div className="text-3xl font-black font-mono">{telemetry.csi || 0}</div>
        <div className="text-xs font-bold tracking-wide mt-1 uppercase">{telemetry.zone || 'Calibrating'}</div>
        <div className="text-[9px] font-mono mt-2 opacity-60">Fused from {activeCount}/{totalCount} sensors</div>
      </div>
    </div>
  );
}