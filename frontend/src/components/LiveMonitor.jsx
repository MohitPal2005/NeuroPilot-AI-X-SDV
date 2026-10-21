import React, { useState, useEffect, useRef, useCallback } from 'react';

// ─── Threshold constants sourced directly from the Python backend ─────────────
// config.py  line 1:  EAR_THRESHOLD = 0.18
const EAR_THRESHOLD = 0.18;
// recovery_engine.py  line 123: "EYE": vis_valid and eye_closed >= 5.0
const EYE_CLOSED_TRIGGER = 5.0;
// config.py  line 18: CSI_YAW_THRESHOLD = 20
const YAW_MARKER = 20;
// config.py  line 20: CSI_PITCH_THRESHOLD = 15
const PITCH_MARKER = 15;
// config.py  line 4:  GAZE_HEAD_YAW_THRESH = 20.0  => display range ±20
const YAW_RANGE = 20;
// config.py  line 5:  GAZE_HEAD_PITCH_THRESH = 20.0 => display range ±15
const PITCH_RANGE = 15;
// ─────────────────────────────────────────────────────────────────────────────

const POLL_INTERVAL_MS = 400;
const SPARKLINE_SAMPLES = 60;

// Zone colour config  (engine.py zone_bounds = [20, 40, 60, 80])
function zoneStyle(zone) {
  switch (zone) {
    case 'Safe':               return { bg: '#052e16', border: '#16a34a', text: '#4ade80', label: '#bbf7d0' };
    case 'Attention Required': return { bg: '#451a03', border: '#d97706', text: '#fbbf24', label: '#fde68a' };
    case 'Cognitive Overload': return { bg: '#431407', border: '#ea580c', text: '#fb923c', label: '#fed7aa' };
    case 'High Risk':          return { bg: '#450a0a', border: '#dc2626', text: '#f87171', label: '#fecaca' };
    case 'Critical Risk':      return { bg: '#3b0a0a', border: '#7f1d1d', text: '#ef4444', label: '#fca5a5' };
    default:                   return { bg: '#0f172a', border: '#334155', text: '#94a3b8', label: '#cbd5e1' };
  }
}

function sensorChipStyle(status) {
  switch (status) {
    case 'active':    return { bg: '#052e16', border: '#16a34a', text: '#4ade80' };
    case 'degraded':  return { bg: '#450a0a', border: '#dc2626', text: '#f87171' };
    case 'buffering': return { bg: '#451a03', border: '#d97706', text: '#fbbf24' };
    default:          return { bg: '#0f172a', border: '#334155', text: '#94a3b8' };
  }
}

const RECOVERY_STATES = ['IDLE', 'WARNING', 'INTERVENING', 'RECOVERED'];
function recoveryStepStyle(step, current) {
  if (step === current) {
    const map = {
      IDLE:        { bg: '#052e16', border: '#16a34a', text: '#4ade80' },
      WARNING:     { bg: '#451a03', border: '#d97706', text: '#fbbf24' },
      INTERVENING: { bg: '#450a0a', border: '#dc2626', text: '#f87171' },
      RECOVERED:   { bg: '#1e3a5f', border: '#3b82f6', text: '#93c5fd' },
    };
    return map[step] || { bg: '#0f172a', border: '#64748b', text: '#cbd5e1' };
  }
  return { bg: '#0f172a', border: '#1e293b', text: '#475569' };
}

function fmt(val, digits) {
  const d = digits !== undefined ? digits : 2;
  if (val === null || val === undefined) return '--';
  return typeof val === 'number' ? val.toFixed(d) : String(val);
}

function HBar({ value, max, color, markerAt, label, unit }) {
  const c = color || '#06b6d4';
  const u = unit !== undefined ? unit : '';
  const pct = Math.min(100, Math.max(0, ((value != null ? value : 0) / max) * 100));
  const markerPct = markerAt != null ? Math.min(100, (markerAt / max) * 100) : null;
  return (
    <div style={{ marginBottom: 10 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
        <span style={{ fontSize: 16, color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.08em' }}>{label}</span>
        <span style={{ fontSize: 20, fontWeight: 700, fontFamily: 'monospace', color: c }}>{fmt(value, 3)}{u}</span>
      </div>
      <div style={{ position: 'relative', height: 14, background: '#1e293b', borderRadius: 8, overflow: 'hidden' }}>
        <div style={{ width: pct + '%', height: '100%', background: c, borderRadius: 8, transition: 'width 0.3s' }} />
        {markerPct != null && (
          <div style={{ position: 'absolute', top: 0, left: markerPct + '%', width: 2, height: '100%', background: '#f59e0b' }} />
        )}
      </div>
    </div>
  );
}

function Gauge({ value, range, markerAt, label }) {
  const clamped = Math.min(range, Math.max(-range, value != null ? value : 0));
  const pct = ((clamped + range) / (range * 2)) * 100;
  const markerPosLeft  = ((-markerAt + range) / (range * 2)) * 100;
  const markerPosRight = (( markerAt + range) / (range * 2)) * 100;
  const color = Math.abs(clamped) > markerAt ? '#f87171' : '#06b6d4';
  return (
    <div style={{ marginBottom: 10 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
        <span style={{ fontSize: 16, color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.08em' }}>{label}</span>
        <span style={{ fontSize: 20, fontWeight: 700, fontFamily: 'monospace', color: color }}>{fmt(value, 1)}°</span>
      </div>
      <div style={{ position: 'relative', height: 14, background: '#1e293b', borderRadius: 8 }}>
        <div style={{ position: 'absolute', left: '50%', top: 0, width: 2, height: '100%', background: '#334155' }} />
        {clamped >= 0
          ? <div style={{ position: 'absolute', left: '50%', width: (pct - 50) + '%', height: '100%', background: color, borderRadius: '0 8px 8px 0', transition: 'width 0.3s' }} />
          : <div style={{ position: 'absolute', left: pct + '%', width: (50 - pct) + '%', height: '100%', background: color, borderRadius: '8px 0 0 8px', transition: 'width 0.3s, left 0.3s' }} />
        }
        <div style={{ position: 'absolute', left: markerPosLeft + '%',  top: 0, width: 2, height: '100%', background: '#f59e0b' }} />
        <div style={{ position: 'absolute', left: markerPosRight + '%', top: 0, width: 2, height: '100%', background: '#f59e0b' }} />
      </div>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, color: '#475569', marginTop: 2 }}>
        <span>-{range}°</span><span>0</span><span>+{range}°</span>
      </div>
    </div>
  );
}

function SimBadge() {
  return (
    <span style={{
      display: 'inline-block', fontSize: 11, fontWeight: 800, letterSpacing: '0.1em',
      padding: '2px 8px', borderRadius: 4,
      background: '#451a03', border: '1px solid #d97706', color: '#fbbf24',
      marginLeft: 8, verticalAlign: 'middle'
    }}>SIMULATED</span>
  );
}

function Sparkline({ data }) {
  const width = 600;
  const height = 80;
  if (!data || data.length < 2) {
    return (
      <div style={{ height: height, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#475569', fontSize: 14 }}>
        Collecting data…
      </div>
    );
  }
  const min = 0;
  const max = 100;
  const pts = data.map(function(v, i) {
    const x = (i / (data.length - 1)) * width;
    const y = height - ((v - min) / (max - min)) * height;
    return x.toFixed(1) + ',' + y.toFixed(1);
  }).join(' ');
  const areaBottom = width + ',' + height + ' 0,' + height;
  return (
    <svg viewBox={'0 0 ' + width + ' ' + height} style={{ width: '100%', height: height }} preserveAspectRatio="none">
      <defs>
        <linearGradient id="lmSparkGrad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%"   stopColor="#06b6d4" stopOpacity="0.35" />
          <stop offset="100%" stopColor="#06b6d4" stopOpacity="0.02" />
        </linearGradient>
      </defs>
      <polygon points={pts + ' ' + areaBottom} fill="url(#lmSparkGrad)" />
      <polyline points={pts} fill="none" stroke="#06b6d4" strokeWidth="2" strokeLinejoin="round" />
      {[20, 40, 60, 80].map(function(thresh) {
        const y = height - ((thresh - min) / (max - min)) * height;
        return <line key={thresh} x1="0" y1={y} x2={width} y2={y} stroke="#334155" strokeDasharray="4 4" strokeWidth="1" />;
      })}
    </svg>
  );
}

function Card({ title, children, style }) {
  const s = style || {};
  return (
    <div style={Object.assign({ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 16, padding: '20px 24px' }, s)}>
      {title && (
        <div style={{ fontSize: 13, fontWeight: 700, color: '#475569', textTransform: 'uppercase', letterSpacing: '0.12em', marginBottom: 16 }}>
          {title}
        </div>
      )}
      {children}
    </div>
  );
}

export default function LiveMonitor() {
  const [data, setData]           = useState(null);
  const [disconnected, setDisc]   = useState(false);
  const [sparkline, setSparkline] = useState([]);
  const pollRef = useRef(null);

  const poll = useCallback(function() {
    fetch('http://localhost:5000/driver/status')
      .then(function(res) {
        if (!res.ok) throw new Error('non-2xx');
        return res.json();
      })
      .then(function(json) {
        setData(json);
        setDisc(false);
        setSparkline(function(prev) {
          const next = prev.concat([json.csi != null ? json.csi : 0]);
          return next.length > SPARKLINE_SAMPLES ? next.slice(-SPARKLINE_SAMPLES) : next;
        });
      })
      .catch(function() { setDisc(true); });
  }, []);

  useEffect(function() {
    poll();
    pollRef.current = setInterval(poll, POLL_INTERVAL_MS);
    return function() { clearInterval(pollRef.current); };
  }, [poll]);

  const csi      = data && data.csi  != null  ? data.csi  : null;
  const zone     = data && data.zone != null  ? data.zone : null;
  const zs       = zoneStyle(zone);

  const ss       = (data && data.sensor_status)   || {};
  const re       = (data && data.recovery_engine) || {};
  const recovState = re.state || 'IDLE';

  const ear           = data ? data.ear            : null;
  const eyeClosedDur  = data ? data.eye_closed_duration : null;
  const yaw           = data ? data.yaw            : null;
  const pitch         = data ? data.pitch          : null;
  const gazeDir       = data ? data.gaze_direction : null;
  const gazeAwayDur   = data ? data.gaze_away_duration : null;

  const arousal   = data ? data.arousal   : null;
  const valence   = data ? data.valence   : null;
  const dominance = data ? data.dominance : null;

  const velocity  = data ? data.velocity  : null;
  const jerk      = data ? data.jerk      : null;
  const isAnomaly = data ? data.is_anomaly : null;
  const isSimKin  = data ? (data.is_simulated || false) : false;

  const ambientAction     = re.ambient_action     || null;
  const rerouteSuggestion = re.reroute_suggestion || null;
  const triggerReason     = re.trigger_reason     || null;
  const ambientIsSimulated = ambientAction ? /^SIMULATED/i.test(ambientAction) : false;

  return (
    <div style={{
      minHeight: '100vh', background: '#020617', color: '#e2e8f0',
      fontFamily: "'Inter', 'Segoe UI', system-ui, sans-serif",
      padding: '24px', boxSizing: 'border-box'
    }}>
      {/* DISCONNECTED BANNER */}
      {disconnected && (
        <div style={{
          position: 'fixed', top: 0, left: 0, right: 0, zIndex: 9999,
          background: '#7f1d1d', borderBottom: '3px solid #dc2626',
          color: '#fca5a5', textAlign: 'center',
          padding: '14px', fontSize: 22, fontWeight: 800, letterSpacing: '0.12em'
        }}>
          DISCONNECTED — backend unreachable
        </div>
      )}

      {/* PAGE HEADER */}
      <div style={{ marginBottom: 24, paddingTop: disconnected ? 60 : 0 }}>
        <div style={{ fontSize: 13, color: '#475569', fontWeight: 700, letterSpacing: '0.14em', textTransform: 'uppercase', marginBottom: 4 }}>
          NeuroPilot AI-X SDV
        </div>
        <h1 style={{ fontSize: 28, fontWeight: 900, margin: 0, color: '#e2e8f0', letterSpacing: '0.04em' }}>
          Live Monitor
          <span style={{ fontSize: 13, fontWeight: 600, color: '#64748b', marginLeft: 16 }}>
            polls every {POLL_INTERVAL_MS}ms · {SPARKLINE_SAMPLES}-sample sparkline
          </span>
        </h1>
      </div>

      {/* ROW 1: CSI + Sensor chips + Recovery stepper */}
      <div style={{ display: 'grid', gridTemplateColumns: '300px 1fr 420px', gap: 16, marginBottom: 16, alignItems: 'start' }}>

        {/* CSI Big Number */}
        <Card>
          <div style={{ fontSize: 20, fontWeight: 700, color: '#475569', textTransform: 'uppercase', letterSpacing: '0.12em', marginBottom: 8 }}>
            CSI
          </div>
          <div style={{ background: zs.bg, border: '2px solid ' + zs.border, borderRadius: 12, padding: '20px 16px', textAlign: 'center' }}>
            <div style={{ fontSize: 80, fontWeight: 900, fontFamily: 'monospace', color: zs.text, lineHeight: 1 }}>
              {csi !== null ? csi : '--'}
            </div>
            <div style={{ fontSize: 22, fontWeight: 700, color: zs.label, marginTop: 8, letterSpacing: '0.06em' }}>
              {zone || '--'}
            </div>
          </div>
        </Card>

        {/* Sensor Status Chips */}
        <Card title="Sensor Status">
          <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
            {['visual', 'acoustic', 'kinematic'].map(function(k) {
              const st = ss[k] || null;
              const cs = sensorChipStyle(st);
              return (
                <div key={k} style={{
                  flex: '1 1 0', minWidth: 120,
                  background: cs.bg, border: '1.5px solid ' + cs.border, borderRadius: 12,
                  padding: '16px 12px', textAlign: 'center'
                }}>
                  <div style={{ fontSize: 13, color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: 6 }}>{k}</div>
                  <div style={{ fontSize: 22, fontWeight: 800, color: cs.text }}>{st || '--'}</div>
                </div>
              );
            })}
          </div>
        </Card>

        {/* Recovery stepper */}
        <Card title="Recovery Engine">
          <div style={{ display: 'flex', gap: 8, marginBottom: 16 }}>
            {RECOVERY_STATES.map(function(step, i) {
              const rs = recoveryStepStyle(step, recovState);
              return (
                <React.Fragment key={step}>
                  <div style={{
                    flex: 1, background: rs.bg, border: '1.5px solid ' + rs.border, borderRadius: 8,
                    padding: '10px 6px', textAlign: 'center'
                  }}>
                    <div style={{ fontSize: 11, fontWeight: 800, color: rs.text, letterSpacing: '0.06em' }}>{step}</div>
                  </div>
                  {i < RECOVERY_STATES.length - 1 && (
                    <div style={{ display: 'flex', alignItems: 'center', color: '#334155', fontSize: 18 }}>›</div>
                  )}
                </React.Fragment>
              );
            })}
          </div>
          <div style={{ fontSize: 16, color: '#94a3b8', marginBottom: 6 }}>
            <span style={{ fontWeight: 700, color: '#cbd5e1' }}>Trigger: </span>
            {triggerReason || '--'}
          </div>
          <div style={{ fontSize: 15, color: '#94a3b8', marginBottom: 6 }}>
            <span style={{ fontWeight: 700, color: '#cbd5e1' }}>Action: </span>
            {ambientAction
              ? <>{ambientIsSimulated ? ambientAction.replace(/^SIMULATED:\s*/i, '') : ambientAction}<SimBadge /></>
              : '--'}
          </div>
          <div style={{ fontSize: 15, color: '#94a3b8' }}>
            <span style={{ fontWeight: 700, color: '#cbd5e1' }}>Reroute: </span>
            {rerouteSuggestion ? <>{rerouteSuggestion}<SimBadge /></> : '--'}
          </div>
        </Card>
      </div>

      {/* ROW 2: Vision / Acoustic / Kinematic */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 16, marginBottom: 16 }}>

        {/* Vision block */}
        <Card title="Vision">
          <HBar
            label="EAR"
            value={ear}
            max={0.5}
            color="#06b6d4"
            markerAt={EAR_THRESHOLD}
            unit=""
          />
          <HBar
            label="Eye Closed Duration"
            value={eyeClosedDur}
            max={EYE_CLOSED_TRIGGER}
            color="#f87171"
            markerAt={EYE_CLOSED_TRIGGER}
            unit="s"
          />
          <Gauge
            label="Yaw"
            value={yaw}
            range={YAW_RANGE}
            markerAt={YAW_MARKER}
          />
          <Gauge
            label="Pitch"
            value={pitch}
            range={PITCH_RANGE}
            markerAt={PITCH_MARKER}
          />
          <div style={{ display: 'flex', gap: 12, marginTop: 4 }}>
            <div style={{ flex: 1, background: '#0f1f2e', border: '1px solid #1e3a5f', borderRadius: 8, padding: '10px 12px' }}>
              <div style={{ fontSize: 13, color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 4 }}>Gaze</div>
              <div style={{ fontSize: 22, fontWeight: 700, color: '#06b6d4', fontFamily: 'monospace' }}>{gazeDir || '--'}</div>
            </div>
            <div style={{ flex: 1, background: '#0f1f2e', border: '1px solid #1e3a5f', borderRadius: 8, padding: '10px 12px' }}>
              <div style={{ fontSize: 13, color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 4 }}>Gaze Away</div>
              <div style={{ fontSize: 22, fontWeight: 700, color: '#06b6d4', fontFamily: 'monospace' }}>{fmt(gazeAwayDur, 1)}s</div>
            </div>
          </div>
        </Card>

        {/* Acoustic block */}
        <Card title="Acoustic">
          <HBar label="Arousal"   value={arousal}   max={1} color="#c084fc" />
          <HBar label="Valence"   value={valence}   max={1} color="#34d399" />
          <HBar label="Dominance" value={dominance} max={1} color="#fb923c" />
        </Card>

        {/* Kinematic block */}
        <Card title="Kinematic">
          <div style={{ fontSize: 12, color: '#64748b', marginBottom: 12 }}>
            SIMULATED steering data
            <SimBadge />
          </div>
          <div style={{ marginBottom: 12 }}>
            <div style={{ fontSize: 16, color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 4 }}>Velocity</div>
            <div style={{ fontSize: 52, fontWeight: 900, fontFamily: 'monospace', color: '#06b6d4', lineHeight: 1 }}>{fmt(velocity, 2)}</div>
            <div style={{ fontSize: 13, color: '#475569' }}>deg/s (mean abs)</div>
          </div>
          <div style={{ marginBottom: 12 }}>
            <div style={{ fontSize: 16, color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 4 }}>Jerk</div>
            <div style={{ fontSize: 52, fontWeight: 900, fontFamily: 'monospace', color: '#fb923c', lineHeight: 1 }}>{fmt(jerk, 2)}</div>
            <div style={{ fontSize: 13, color: '#475569' }}>deg/s² (mean abs)</div>
          </div>
          <div style={{
            background: isAnomaly ? '#450a0a' : '#052e16',
            border: '1.5px solid ' + (isAnomaly ? '#dc2626' : '#16a34a'),
            borderRadius: 10, padding: '12px 16px', textAlign: 'center', marginTop: 8
          }}>
            <div style={{ fontSize: 14, color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: 4 }}>Anomaly</div>
            <div style={{ fontSize: 40, fontWeight: 900, color: isAnomaly ? '#f87171' : '#4ade80' }}>
              {isAnomaly === null ? '--' : isAnomaly ? 'YES' : 'NO'}
            </div>
          </div>
          {isSimKin && (
            <div style={{ fontSize: 12, color: '#475569', marginTop: 8, textAlign: 'center' }}>
              source: kinematic.py line 159 — "is_simulated": True
            </div>
          )}
        </Card>
      </div>

      {/* ROW 3: CSI Sparkline */}
      <Card title={'CSI Sparkline — last ' + SPARKLINE_SAMPLES + ' samples (client-side)'}>
        <Sparkline data={sparkline} />
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, color: '#475569', marginTop: 4 }}>
          <span>← {SPARKLINE_SAMPLES} samples ago</span>
          <span>Now →</span>
        </div>
        <div style={{ display: 'flex', gap: 16, marginTop: 8, flexWrap: 'wrap' }}>
          {[['Safe','#4ade80','0–20'],['Attention','#fbbf24','20–40'],['Overload','#fb923c','40–60'],['High Risk','#f87171','60–80'],['Critical','#ef4444','80–100']].map(function(item) {
            return <span key={item[0]} style={{ fontSize: 12, color: item[1] }}>■ {item[0]} ({item[2]})</span>;
          })}
        </div>
      </Card>
    </div>
  );
}
