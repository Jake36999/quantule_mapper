import React, { Suspense, lazy, useEffect, useRef, useState } from 'react';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, 
  AreaChart, Area, ScatterChart, Scatter
} from 'recharts';
import { 
  Activity, Server, Database, Terminal, Zap, Play, Square, 
  ChevronDown, ChevronRight, FileText, Download, RefreshCw, 
  Wifi, WifiOff, LayoutDashboard, Clock, HardDrive
} from 'lucide-react';
import {
  apiFetch,
  startHunt,
  stopHunt,
  listDataFiles,
  stageHuntConfig,
  getRunStatus,
  getDebugSuites,
  runDebugTests,
  DebugSuite,
} from './api_client';
import TopologicalExplorer from './components/TopologicalExplorer';
import LedgerExplorer from './components/LedgerExplorer';
import FleetTelemetry from './components/FleetTelemetry';
import BacklogProgressWidget from './components/BacklogProgressWidget';
import TensorPointCloud from './components/TensorPointCloud';
import TerminalMatrix from './components/TerminalMatrix';
import { TelemetryProvider, useTelemetry } from './components/TelemetryProvider';

const VisualAnalysis = lazy(() => import('./components/VisualAnalysis'));

/** --- TYPES --- */
interface FileRecord {
  name: string;
  size: number;
  modified: number;
  type: string;
}

const formatActionableError = (err: unknown, fallback: string): string => {
  const raw = err instanceof Error ? err.message : String(err || fallback);
  const codeMatch = raw.match(/\b(4\d\d|5\d\d)\b/);
  const code = codeMatch ? codeMatch[1] : '503';
  if (raw.toLowerCase().includes('failed to fetch')) {
    return `Backend unreachable [${code}]. Ensure API Preview is running on 127.0.0.1:8000, then retry.`;
  }
  return raw || fallback;
};

const normalizeBacklogSourcePath = (value: string): string => {
  const trimmed = value.trim();
  if (!trimmed) {
    return 'backlog_queue.json';
  }
  const normalizedSlashes = trimmed.replace(/\\+/g, '/');
  const normalizedSegments = normalizedSlashes
    .split('/')
    .map(segment => segment.trim())
    .filter(Boolean)
    .map(segment => segment.replace(/\s+/g, '_'));
  return normalizedSegments.join('/');
};

/** --- API HOOKS --- */
const useDataVault = (active: boolean) => {
  const [files, setFiles] = useState<FileRecord[]>([]);
  
  const refresh = async () => {
    try {
      const data = await listDataFiles();
      setFiles(data.files || []);
    } catch (e) { console.error("Vault Error", e); }
  };

  useEffect(() => { if (active) refresh(); }, [active]);
  return { files, refresh };
};

const useObserverLog = (active: boolean) => {
  const [logContent, setLogContent] = useState<string>("");
  
  const refresh = async () => {
    try {
      const res = await apiFetch('/api/logs/observer');
      const data = await res.json();
      setLogContent(data.content || "");
    } catch (e) { console.error("Log Error", e); }
  };

  useEffect(() => { if (active) refresh(); }, [active]);
  return { logContent, refresh };
};

/** --- SUB-COMPONENTS --- */

type MissionStartPayload = {
  staged_path: string;
  hunt_name?: string;
  origin?: string;
  staged_at?: string;
  staged_config_hash?: string;
  mode?: 'evolution' | 'backlog';
  backlog_source?: string;
  worker_count?: number;
};

const MissionControl = ({ onStart, onStop }: { onStart: (payload: MissionStartPayload)=>void, onStop: ()=>void }) => {
  const [workerCount, setWorkerCount] = useState('1');
  const [huntName, setHuntName] = useState('BURN_IN_STRESS_TEST');
  const [configPath, setConfigPath] = useState('');
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [gens, setGens] = useState('');
  const [batch, setBatch] = useState('');
  const [populationSize, setPopulationSize] = useState('');
  const [seedsPerCandidate, setSeedsPerCandidate] = useState('');
  const [nGrid, setNGrid] = useState('');
  const [tSteps, setTSteps] = useState('');
  const [dt, setDt] = useState('');
  const [lDomain, setLDomain] = useState('');
  const [collapseThreshold, setCollapseThreshold] = useState('');
  const [mutationRate, setMutationRate] = useState('');
  const [crossoverRate, setCrossoverRate] = useState('');
  const [survivalFraction, setSurvivalFraction] = useState('');
  const [predatorSweepFrequency, setPredatorSweepFrequency] = useState('');
  const [minQueueDepth, setMinQueueDepth] = useState('');
  const [pollInterval, setPollInterval] = useState('');
  const [bleedWorkers, setBleedWorkers] = useState('');
  const [artifactGcMinAgeSeconds, setArtifactGcMinAgeSeconds] = useState('');
  const [jobLeaseTimeoutSeconds, setJobLeaseTimeoutSeconds] = useState('');
  const [resultLeaseTimeoutSeconds, setResultLeaseTimeoutSeconds] = useState('');
  const [workerHeartbeatTtlSeconds, setWorkerHeartbeatTtlSeconds] = useState('');
  const [maxQueueDepth, setMaxQueueDepth] = useState('');
  // Phase 4: Run mode selector
  const [runMode, setRunMode] = useState<'evolution' | 'backlog'>('evolution');
  const [backlogSource, setBacklogSource] = useState('backlog_queue.json');
  const [isStaging, setIsStaging] = useState(false);
  const [stageError, setStageError] = useState<string | null>(null);
  const [stagedPath, setStagedPath] = useState<string | null>(null);
  const [stagedHash, setStagedHash] = useState<string | null>(null);
  const [stagedAt, setStagedAt] = useState<string | null>(null);
  const [stageStatus, setStageStatus] = useState<string>('Ready to stage initiating config.');
  // New physics parameters for Gen 0
  const [paramD, setParamD] = useState('');
  const [paramEta, setParamEta] = useState('');
  const [paramA, setParamA] = useState('');
  const [paramRhoVac, setParamRhoVac] = useState('');

  const parseRequiredInt = (value: string, fieldName: string): number => {
    const parsed = Number(value);
    if (!Number.isFinite(parsed) || parsed <= 0 || !Number.isInteger(parsed)) {
      throw new Error(`${fieldName} must be a positive integer.`);
    }
    return parsed;
  };

  const parseRequiredFloat = (value: string, fieldName: string): number => {
    const parsed = Number(value);
    if (!Number.isFinite(parsed) || parsed <= 0) {
      throw new Error(`${fieldName} must be a positive number.`);
    }
    return parsed;
  };

  const parseOptionalInt = (value: string, fieldName: string): number | undefined => {
    if (value.trim() === '') {
      return undefined;
    }
    const parsed = Number(value);
    if (!Number.isFinite(parsed) || parsed <= 0 || !Number.isInteger(parsed)) {
      throw new Error(`${fieldName} must be a positive integer.`);
    }
    return parsed;
  };

  const parseOptionalFloat = (value: string, fieldName: string): number | undefined => {
    if (value.trim() === '') {
      return undefined;
    }
    const parsed = Number(value);
    if (!Number.isFinite(parsed) || parsed <= 0) {
      throw new Error(`${fieldName} must be a positive number.`);
    }
    return parsed;
  };

  const stageInputs = async () => {
    setIsStaging(true);
    setStageError(null);
    setStageStatus('Staging initiating config...');
    try {
      const trimmedConfigPath = configPath.trim();
      const normalizedBacklogSource = normalizeBacklogSourcePath(backlogSource);
      const payload = trimmedConfigPath
        ? {
            config_path: trimmedConfigPath,
            hunt_name: huntName.trim(),
            origin: 'UI_CONTROL',
            mode: runMode,
            backlog_source: runMode === 'backlog' ? normalizedBacklogSource : undefined,
            // New params at root if provided
            ...(paramD.trim() !== '' ? { param_D: Number(paramD) } : {}),
            ...(paramEta.trim() !== '' ? { param_eta: Number(paramEta) } : {}),
            ...(paramA.trim() !== '' ? { param_a: Number(paramA) } : {}),
            ...(paramRhoVac.trim() !== '' ? { param_rho_vac: Number(paramRhoVac) } : {}),
          }
        : (() => {
            const generatedPayload = {
              hunt_name: huntName.trim(),
              generations: parseRequiredInt(gens, 'Generations'),
              batch_size: parseRequiredInt(batch, 'Batch size'),
              population_size: parseRequiredInt(populationSize, 'Population size'),
              seeds_per_candidate: parseRequiredInt(seedsPerCandidate, 'Seeds per candidate'),
              n_grid: parseRequiredInt(nGrid, 'N grid'),
              t_steps: parseRequiredInt(tSteps, 'T steps'),
              dt: parseRequiredFloat(dt, 'DT'),
              origin: 'UI_CONTROL',
              mode: runMode,
              backlog_source: runMode === 'backlog' ? normalizedBacklogSource : undefined,
              l_domain: parseOptionalFloat(lDomain, 'L domain'),
              collapse_threshold: parseOptionalFloat(collapseThreshold, 'Collapse threshold'),
              mutation_rate: runMode === 'evolution' ? parseOptionalFloat(mutationRate, 'Mutation rate') : undefined,
              crossover_rate: runMode === 'evolution' ? parseOptionalFloat(crossoverRate, 'Crossover rate') : undefined,
              survival_fraction: runMode === 'evolution' ? parseOptionalFloat(survivalFraction, 'Survival fraction') : undefined,
              predator_sweep_frequency: runMode === 'evolution' ? parseOptionalInt(predatorSweepFrequency, 'Predator sweep frequency') : undefined,
              min_queue_depth: parseOptionalInt(minQueueDepth, 'Min queue depth'),
              poll_interval: parseOptionalFloat(pollInterval, 'Poll interval'),
              bleed_workers: parseOptionalInt(bleedWorkers, 'Bleed workers'),
              artifact_gc_min_age_seconds: parseOptionalInt(artifactGcMinAgeSeconds, 'Artifact GC min age seconds'),
              job_lease_timeout_seconds: parseOptionalInt(jobLeaseTimeoutSeconds, 'Job lease timeout seconds'),
              result_lease_timeout_seconds: parseOptionalInt(resultLeaseTimeoutSeconds, 'Result lease timeout seconds'),
              worker_heartbeat_ttl_seconds: parseOptionalInt(workerHeartbeatTtlSeconds, 'Worker heartbeat TTL seconds'),
              max_queue_depth: parseOptionalInt(maxQueueDepth, 'Max queue depth'),
              // New params at root if provided
              ...(paramD.trim() !== '' ? { param_D: Number(paramD) } : {}),
              ...(paramEta.trim() !== '' ? { param_eta: Number(paramEta) } : {}),
              ...(paramA.trim() !== '' ? { param_a: Number(paramA) } : {}),
              ...(paramRhoVac.trim() !== '' ? { param_rho_vac: Number(paramRhoVac) } : {}),
            };
            return generatedPayload;
          })();

      const response = await stageHuntConfig(payload);
      if (response.status === 'success') {
        setStagedPath(response.staged_path || null);
        setStagedHash(response.config_hash || null);
        setStagedAt(response.staged_at || null);
        setStageStatus('Initiating config staged and ready to launch.');
      } else {
        setStageError(response.message || 'Failed to stage control inputs.');
        setStageStatus('Staging failed.');
        setStagedPath(null);
        setStagedHash(null);
        setStagedAt(null);
      }
    } catch (err) {
      setStageError(formatActionableError(err, 'Failed to stage control inputs.'));
      setStageStatus('Staging failed.');
      setStagedPath(null);
      setStagedHash(null);
      setStagedAt(null);
    } finally {
      setIsStaging(false);
    }
  };

  const handleInitiate = async () => {
    if (stageError || !stagedPath) {
      return;
    }
    onStart({
      staged_path: stagedPath,
      hunt_name: huntName.trim(),
      origin: 'UI_CONTROL',
      staged_at: stagedAt || undefined,
      staged_config_hash: stagedHash || undefined,
      mode: runMode,
      backlog_source: runMode === 'backlog' ? normalizeBacklogSourcePath(backlogSource) : undefined,
      worker_count: parseOptionalInt(workerCount, 'Worker count') || 1,
    });
  };
  
  return (
    <div className="bg-slate-900 border border-slate-800 rounded-lg p-6">
      <h3 className="text-white font-bold mb-4 flex items-center gap-2"><Zap size={18} className="text-yellow-400"/> Mission Control</h3>
      {/* Phase 4: Run Mode Selector */}
      <div className="mb-4 rounded border border-slate-700 bg-slate-950/60 p-3">
        <label className="text-xs text-slate-400 block mb-2 font-semibold">RUN MODE</label>
        <div className="flex gap-4">
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="radio"
              name="runMode"
              value="evolution"
              checked={runMode === 'evolution'}
              onChange={() => setRunMode('evolution')}
              className="accent-cyan-400"
            />
            <span className={`text-sm font-semibold ${runMode === 'evolution' ? 'text-cyan-300' : 'text-slate-400'}`}>Evolution (NSGA-II)</span>
          </label>
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="radio"
              name="runMode"
              value="backlog"
              checked={runMode === 'backlog'}
              onChange={() => setRunMode('backlog')}
              className="accent-amber-400"
            />
            <span className={`text-sm font-semibold ${runMode === 'backlog' ? 'text-amber-300' : 'text-slate-400'}`}>Backlog Replay</span>
          </label>
        </div>
        {runMode === 'backlog' && (
          <div className="mt-3">
            <label htmlFor="mc-backlog-source" className="text-xs text-slate-400 block mb-1">BACKLOG SOURCE FILE</label>
            <input
              id="mc-backlog-source"
              title="Backlog source file path"
              type="text"
              value={backlogSource}
              onChange={e => setBacklogSource(e.target.value)}
              placeholder="backlog_queue.json"
              className="w-full bg-slate-950 border border-amber-700 rounded p-2 text-white font-mono text-sm"
            />
            {normalizeBacklogSourcePath(backlogSource) !== (backlogSource.trim() || 'backlog_queue.json') && (
              <p className="mt-1 text-[11px] text-amber-300/90">
                Normalized path: <span className="font-mono">{normalizeBacklogSourcePath(backlogSource)}</span>
              </p>
            )}
            <p className="mt-1 text-[11px] text-amber-500/80">Backlog mode replays configs from this queue — no NSGA-II evolution operators are invoked. Use <span className="font-mono">prep_backlog.py</span> to populate the queue.</p>
          </div>
        )}
      </div>
      <div className="grid grid-cols-2 gap-4 mb-6">
        <div>
          <label htmlFor="mc-worker-count" className="text-xs text-amber-400 block mb-1 font-bold">WORKER COUNT (GPUs)</label>
          <input id="mc-worker-count" title="Number of GPU workers to launch" type="number" value={workerCount} min={1} max={8} step={1} onChange={e => setWorkerCount(e.target.value)} className="w-full bg-slate-950 border border-amber-700 rounded p-2 text-amber-100 font-mono text-sm shadow-[0_0_8px_rgba(217,119,6,0.3)]" />
        </div>
        <div className="col-span-2">
          <label htmlFor="mc-hunt-name" className="text-xs text-slate-400 block mb-1">HUNT NAME</label>
          <input id="mc-hunt-name" title="Hunt name" type="text" value={huntName} onChange={e => setHuntName(e.target.value)} placeholder="BURN_IN_STRESS_TEST" className="w-full bg-slate-950 border border-slate-700 rounded p-2 text-white font-mono text-sm" />
        </div>
        <div className="col-span-2">
          <label htmlFor="mc-config-path" className="text-xs text-slate-400 block mb-1">EXISTING CONFIG PATH (OPTIONAL)</label>
          <input id="mc-config-path" title="Existing config path" type="text" value={configPath} onChange={e => setConfigPath(e.target.value)} placeholder="input_configs/your_config.json" className="w-full bg-slate-950 border border-slate-700 rounded p-2 text-white font-mono text-sm" />
          <p className="mt-1 text-[11px] text-slate-500">Provide a path to stage an existing config JSON, or leave blank and fill all fields below to generate an initiating config.</p>
        </div>
        <div>
          <label htmlFor="mc-generations" className="text-xs text-slate-400 block mb-1">GENERATIONS</label>
          <input id="mc-generations" title="Generations" type="number" value={gens} min={1} onChange={e => setGens(e.target.value)} className="w-full bg-slate-950 border border-slate-700 rounded p-2 text-white font-mono text-sm" />
        </div>
        <div>
          <label htmlFor="mc-batch-size" className="text-xs text-slate-400 block mb-1">BATCH SIZE</label>
          <input id="mc-batch-size" title="Batch size" type="number" value={batch} min={1} onChange={e => setBatch(e.target.value)} className="w-full bg-slate-950 border border-slate-700 rounded p-2 text-white font-mono text-sm" />
        </div>
        <div>
          <label htmlFor="mc-population-size" className="text-xs text-slate-400 block mb-1">POPULATION SIZE</label>
          <input id="mc-population-size" title="Population size" type="number" value={populationSize} min={1} onChange={e => setPopulationSize(e.target.value)} className="w-full bg-slate-950 border border-slate-700 rounded p-2 text-white font-mono text-sm" />
        </div>
        <div>
          <label htmlFor="mc-seeds-per-candidate" className="text-xs text-slate-400 block mb-1">SEEDS / CANDIDATE</label>
          <input id="mc-seeds-per-candidate" title="Seeds per candidate" type="number" value={seedsPerCandidate} min={1} onChange={e => setSeedsPerCandidate(e.target.value)} className="w-full bg-slate-950 border border-slate-700 rounded p-2 text-white font-mono text-sm" />
        </div>
        <div>
          <label htmlFor="mc-n-grid" className="text-xs text-slate-400 block mb-1">N GRID</label>
          <input id="mc-n-grid" title="N grid" type="number" value={nGrid} min={1} onChange={e => setNGrid(e.target.value)} className="w-full bg-slate-950 border border-slate-700 rounded p-2 text-white font-mono text-sm" />
        </div>
        <div>
          <label htmlFor="mc-t-steps" className="text-xs text-slate-400 block mb-1">T STEPS</label>
          <input id="mc-t-steps" title="T steps" type="number" value={tSteps} min={1} onChange={e => setTSteps(e.target.value)} className="w-full bg-slate-950 border border-slate-700 rounded p-2 text-white font-mono text-sm" />
        </div>
        <div>
          <label htmlFor="mc-dt" className="text-xs text-slate-400 block mb-1">DT</label>
          <input id="mc-dt" title="Delta time" type="number" value={dt} min={0.000001} step={0.0001} onChange={e => setDt(e.target.value)} className="w-full bg-slate-950 border border-slate-700 rounded p-2 text-white font-mono text-sm" />
        </div>
      </div>
      <div className="mb-6 rounded border border-slate-800 bg-slate-950/80 p-3">
        <button
          type="button"
          onClick={() => setShowAdvanced(prev => !prev)}
          className="w-full flex items-center justify-between text-left text-sm font-semibold text-cyan-300 hover:text-cyan-200"
        >
          <span>Advanced Options (Optional)</span>
          {showAdvanced ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
        </button>
        {showAdvanced && (
          <div className="mt-3 grid grid-cols-2 gap-4">
            <div>
              <label htmlFor="mc-l-domain" className="text-xs text-slate-400 block mb-1">L DOMAIN</label>
              <input id="mc-l-domain" title="L domain" type="number" value={lDomain} min={0.000001} step={0.001} onChange={e => setLDomain(e.target.value)} className="w-full bg-slate-950 border border-slate-700 rounded p-2 text-white font-mono text-sm" />
            </div>
            <div>
              <label htmlFor="mc-collapse-threshold" className="text-xs text-slate-400 block mb-1">COLLAPSE THRESHOLD</label>
              <input id="mc-collapse-threshold" title="Collapse threshold" type="number" value={collapseThreshold} min={0.000001} step={0.001} onChange={e => setCollapseThreshold(e.target.value)} className="w-full bg-slate-950 border border-slate-700 rounded p-2 text-white font-mono text-sm" />
            </div>
            {/* Evolution-only controls hidden in backlog mode */}
            {runMode === 'evolution' && (
              <>
            <div>
              <label htmlFor="mc-mutation-rate" className="text-xs text-slate-400 block mb-1">MUTATION RATE</label>
              <input id="mc-mutation-rate" title="Mutation rate" type="number" value={mutationRate} min={0.000001} step={0.001} onChange={e => setMutationRate(e.target.value)} className="w-full bg-slate-950 border border-slate-700 rounded p-2 text-white font-mono text-sm" />
            </div>
            <div>
              <label htmlFor="mc-crossover-rate" className="text-xs text-slate-400 block mb-1">CROSSOVER RATE</label>
              <input id="mc-crossover-rate" title="Crossover rate" type="number" value={crossoverRate} min={0.000001} step={0.001} onChange={e => setCrossoverRate(e.target.value)} className="w-full bg-slate-950 border border-slate-700 rounded p-2 text-white font-mono text-sm" />
            </div>
            <div>
              <label htmlFor="mc-survival-fraction" className="text-xs text-slate-400 block mb-1">SURVIVAL FRACTION</label>
              <input id="mc-survival-fraction" title="Survival fraction" type="number" value={survivalFraction} min={0.000001} step={0.001} onChange={e => setSurvivalFraction(e.target.value)} className="w-full bg-slate-950 border border-slate-700 rounded p-2 text-white font-mono text-sm" />
            </div>
            <div>
              <label htmlFor="mc-predator-sweep-frequency" className="text-xs text-slate-400 block mb-1">PREDATOR SWEEP FREQUENCY</label>
              <input id="mc-predator-sweep-frequency" title="Predator sweep frequency" type="number" value={predatorSweepFrequency} min={1} step={1} onChange={e => setPredatorSweepFrequency(e.target.value)} className="w-full bg-slate-950 border border-slate-700 rounded p-2 text-white font-mono text-sm" />
            </div>
              </>
            )}
            <div>
              <label htmlFor="mc-min-queue-depth" className="text-xs text-slate-400 block mb-1">MIN QUEUE DEPTH</label>
              <input id="mc-min-queue-depth" title="Min queue depth" type="number" value={minQueueDepth} min={1} step={1} onChange={e => setMinQueueDepth(e.target.value)} className="w-full bg-slate-950 border border-slate-700 rounded p-2 text-white font-mono text-sm" />
            </div>
            <div>
              <label htmlFor="mc-poll-interval" className="text-xs text-slate-400 block mb-1">POLL INTERVAL</label>
              <input id="mc-poll-interval" title="Poll interval" type="number" value={pollInterval} min={0.000001} step={0.1} onChange={e => setPollInterval(e.target.value)} className="w-full bg-slate-950 border border-slate-700 rounded p-2 text-white font-mono text-sm" />
            </div>
            <div>
              <label htmlFor="mc-bleed-workers" className="text-xs text-slate-400 block mb-1">BLEED WORKERS</label>
              <input id="mc-bleed-workers" title="Bleed workers" type="number" value={bleedWorkers} min={1} step={1} onChange={e => setBleedWorkers(e.target.value)} className="w-full bg-slate-950 border border-slate-700 rounded p-2 text-white font-mono text-sm" />
            </div>
            <div>
              <label htmlFor="mc-artifact-gc-min-age-seconds" className="text-xs text-slate-400 block mb-1">ARTIFACT GC MIN AGE (S)</label>
              <input id="mc-artifact-gc-min-age-seconds" title="Artifact GC min age seconds" type="number" value={artifactGcMinAgeSeconds} min={1} step={1} onChange={e => setArtifactGcMinAgeSeconds(e.target.value)} className="w-full bg-slate-950 border border-slate-700 rounded p-2 text-white font-mono text-sm" />
            </div>
            <div>
              <label htmlFor="mc-job-lease-timeout-seconds" className="text-xs text-slate-400 block mb-1">JOB LEASE TIMEOUT (S)</label>
              <input id="mc-job-lease-timeout-seconds" title="Job lease timeout seconds" type="number" value={jobLeaseTimeoutSeconds} min={1} step={1} onChange={e => setJobLeaseTimeoutSeconds(e.target.value)} className="w-full bg-slate-950 border border-slate-700 rounded p-2 text-white font-mono text-sm" />
            </div>
            <div>
              <label htmlFor="mc-result-lease-timeout-seconds" className="text-xs text-slate-400 block mb-1">RESULT LEASE TIMEOUT (S)</label>
              <input id="mc-result-lease-timeout-seconds" title="Result lease timeout seconds" type="number" value={resultLeaseTimeoutSeconds} min={1} step={1} onChange={e => setResultLeaseTimeoutSeconds(e.target.value)} className="w-full bg-slate-950 border border-slate-700 rounded p-2 text-white font-mono text-sm" />
            </div>
            <div>
              <label htmlFor="mc-worker-heartbeat-ttl-seconds" className="text-xs text-slate-400 block mb-1">WORKER HEARTBEAT TTL (S)</label>
              <input id="mc-worker-heartbeat-ttl-seconds" title="Worker heartbeat TTL seconds" type="number" value={workerHeartbeatTtlSeconds} min={1} step={1} onChange={e => setWorkerHeartbeatTtlSeconds(e.target.value)} className="w-full bg-slate-950 border border-slate-700 rounded p-2 text-white font-mono text-sm" />
            </div>
            <div>
              <label htmlFor="mc-max-queue-depth" className="text-xs text-slate-400 block mb-1">MAX QUEUE DEPTH</label>
              <input id="mc-max-queue-depth" title="Max queue depth" type="number" value={maxQueueDepth} min={1} step={1} onChange={e => setMaxQueueDepth(e.target.value)} className="w-full bg-slate-950 border border-slate-700 rounded p-2 text-white font-mono text-sm" />
            </div>
            {/* New Gen 0 Physics Parameters */}
            <div>
              <label htmlFor="mc-param-d" className="text-xs text-cyan-400 block mb-1">param_D</label>
              <input id="mc-param-d" title="param_D" type="number" value={paramD} onChange={e => setParamD(e.target.value)} className="w-full bg-slate-950 border border-cyan-700 rounded p-2 text-cyan-100 font-mono text-sm" placeholder="Enter param_D (optional)" />
            </div>
            <div>
              <label htmlFor="mc-param-eta" className="text-xs text-cyan-400 block mb-1">param_eta</label>
              <input id="mc-param-eta" title="param_eta" type="number" value={paramEta} onChange={e => setParamEta(e.target.value)} className="w-full bg-slate-950 border border-cyan-700 rounded p-2 text-cyan-100 font-mono text-sm" placeholder="Enter param_eta (optional)" />
            </div>
            <div>
              <label htmlFor="mc-param-a" className="text-xs text-cyan-400 block mb-1">param_a</label>
              <input id="mc-param-a" title="param_a" type="number" value={paramA} onChange={e => setParamA(e.target.value)} className="w-full bg-slate-950 border border-cyan-700 rounded p-2 text-cyan-100 font-mono text-sm" placeholder="Enter param_a (optional)" />
            </div>
            <div>
              <label htmlFor="mc-param-rho-vac" className="text-xs text-cyan-400 block mb-1">param_rho_vac</label>
              <input id="mc-param-rho-vac" title="param_rho_vac" type="number" value={paramRhoVac} onChange={e => setParamRhoVac(e.target.value)} className="w-full bg-slate-950 border border-cyan-700 rounded p-2 text-cyan-100 font-mono text-sm" placeholder="Enter param_rho_vac (optional)" />
            </div>
          </div>
        )}
      </div>
      <div className="mb-4 rounded border border-slate-800 bg-slate-950 p-3 text-xs">
        {!isStaging && !stageError && <div className="text-slate-300">{stageStatus}</div>}
        {isStaging && <div className="text-amber-300">Staging control inputs...</div>}
        {!isStaging && stageError && <div className="text-rose-300">Stage failed: {stageError}</div>}
        {!isStaging && !stageError && stagedPath && (
          <div className="text-emerald-300">
            <div>Staged config: <span className="font-mono">{stagedPath}</span></div>
            {stagedHash && <div>Hash: <span className="font-mono">{stagedHash.slice(0, 12)}</span></div>}
          </div>
        )}
      </div>
      <div className="flex gap-3">
        <button onClick={() => void stageInputs()} disabled={isStaging} className="flex-1 bg-cyan-700 hover:bg-cyan-600 disabled:bg-slate-700 disabled:text-slate-400 text-white font-bold py-3 rounded transition-colors">
          GENERATE/STAGE INITIATING CONFIG
        </button>
        <button onClick={() => void handleInitiate()} disabled={isStaging || !!stageError || !stagedPath} className="flex-1 bg-emerald-600 hover:bg-emerald-500 disabled:bg-slate-700 disabled:text-slate-400 text-white font-bold py-3 rounded flex items-center justify-center gap-2 transition-colors">
          <Play size={16} fill="white" /> INITIATE HUNT
        </button>
        <button onClick={onStop} className="flex-1 bg-rose-900/50 hover:bg-rose-900 border border-rose-800 text-rose-200 font-bold py-3 rounded flex items-center justify-center gap-2 transition-colors">
          <Square size={16} fill="currentColor" /> ABORT
        </button>
      </div>
    </div>
  );
};

const DataVault = ({ active }: { active: boolean }) => {
  const { files, refresh } = useDataVault(active);
  
  const formatSize = (bytes: number) => {
    if (bytes < 1024) return bytes + " B";
    if (bytes < 1024*1024) return (bytes/1024).toFixed(1) + " KB";
    return (bytes/(1024*1024)).toFixed(1) + " MB";
  };

  return (
    <div className="h-full flex flex-col bg-slate-900 border border-slate-800 rounded-lg overflow-hidden">
      <div className="p-4 border-b border-slate-800 flex justify-between items-center bg-slate-900/80 backdrop-blur">
        <h3 className="text-white font-bold flex items-center gap-2"><Database size={16} className="text-cyan-400"/> Simulation Data Vault</h3>
        <button title="Refresh data vault" onClick={refresh} className="p-1 hover:bg-slate-800 rounded"><RefreshCw size={14} className="text-slate-400"/></button>
      </div>
      <div className="flex-1 overflow-y-auto p-2">
        <table className="w-full text-left text-xs text-slate-400">
          <thead className="text-slate-500 font-mono uppercase bg-slate-950/50 sticky top-0">
            <tr>
              <th className="p-3">Filename</th>
              <th className="p-3">Size</th>
              <th className="p-3">Modified</th>
              <th className="p-3 text-right">Action</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800">
            {files.map(f => (
              <tr key={f.name} className="hover:bg-slate-800/50 transition-colors group">
                <td className="p-3 font-mono text-slate-300 flex items-center gap-2">
                  <FileText size={14} className={f.name.includes('.h5') ? "text-purple-400" : "text-yellow-400"}/>
                  {f.name}
                </td>
                <td className="p-3 font-mono">{formatSize(f.size)}</td>
                <td className="p-3">{new Date(f.modified * 1000).toLocaleString()}</td>
                <td className="p-3 text-right">
                  <a href={`/api/data/download/${encodeURI(f.name)}`} download className="text-cyan-500 hover:text-cyan-300 font-bold opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-end gap-1">
                    <Download size={14}/> GET
                  </a>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

const ObserverConsole = ({ active }: { active: boolean }) => {
  const { logContent, refresh } = useObserverLog(active);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
  }, [logContent]);

  return (
    <div className="h-full flex flex-col bg-slate-950 border border-slate-800 rounded-lg overflow-hidden font-mono text-xs">
      <div className="p-2 border-b border-slate-800 bg-slate-900 flex justify-between items-center">
        <div className="flex items-center gap-2 text-slate-400"><Terminal size={14}/> Observer Daemon Log (Last 2000 lines)</div>
        <button title="Refresh observer logs" onClick={refresh} className="p-1 hover:bg-slate-800 rounded text-slate-400"><RefreshCw size={14}/></button>
      </div>
      <div ref={scrollRef} className="flex-1 overflow-y-auto p-4 text-slate-300 whitespace-pre-wrap">
        {logContent || "Connecting to Observer Daemon..."}
      </div>
    </div>
  );
};

const DebugConsole = ({ active }: { active: boolean }) => {
  const [suites, setSuites] = useState<DebugSuite[]>([]);
  const [selectedSuite, setSelectedSuite] = useState('e2e');
  const [isRunning, setIsRunning] = useState(false);
  const [runOutput, setRunOutput] = useState('');
  const [runStatus, setRunStatus] = useState<string>('idle');

  const refreshSuites = async () => {
    try {
      const response = await getDebugSuites();
      const suiteList = response.suites || [];
      setSuites(suiteList);
      if (suiteList.length > 0 && !suiteList.find(s => s.id === selectedSuite)) {
        setSelectedSuite(suiteList[0].id);
      }
    } catch (e) {
      console.error('Debug suite error', e);
    }
  };

  const handleRunSuite = async () => {
    setIsRunning(true);
    setRunStatus('running');
    try {
      const response = await runDebugTests(selectedSuite);
      setRunStatus(response.status || 'error');
      setRunOutput(response.output || response.message || 'No output captured.');
    } catch (e) {
      setRunStatus('error');
      setRunOutput(e instanceof Error ? e.message : 'Debug test run failed.');
    } finally {
      setIsRunning(false);
    }
  };

  useEffect(() => {
    if (!active) return;
    void refreshSuites();
  }, [active]);

  return (
    <div className="grid grid-cols-1 xl:grid-cols-2 gap-6 h-full">
      <div className="bg-slate-900 border border-slate-800 rounded-lg p-4 flex flex-col min-h-[420px]">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-white font-bold">Debug Test Runner</h3>
          <button title="Refresh debug suites" onClick={() => void refreshSuites()} className="p-1 hover:bg-slate-800 rounded text-slate-400"><RefreshCw size={14}/></button>
        </div>
        <div className="flex gap-3 mb-3">
          <select title="Debug test suite" value={selectedSuite} onChange={(e) => setSelectedSuite(e.target.value)} className="flex-1 bg-slate-950 border border-slate-700 rounded p-2 text-white text-sm font-mono">
            {suites.length === 0 ? (
              <option value="" className="text-slate-900 bg-slate-100">No suites discovered</option>
            ) : (
              suites.map((suite) => (
                <option
                  key={suite.id}
                  value={suite.id}
                  className="text-slate-900 bg-slate-100"
                >
                  {suite.label} ({suite.id})
                </option>
              ))
            )}
          </select>
          <button onClick={() => void handleRunSuite()} disabled={isRunning || suites.length === 0} className="bg-blue-600 hover:bg-blue-500 disabled:bg-slate-700 disabled:text-slate-400 text-white font-bold px-4 py-2 rounded text-sm">
            {isRunning ? 'RUNNING...' : 'RUN TESTS'}
          </button>
        </div>
        <div className="text-xs text-slate-400 mb-2">Status: <span className="font-mono text-slate-300">{runStatus}</span></div>
        <div className="flex-1 overflow-auto rounded border border-slate-800 bg-slate-950 p-3 text-xs font-mono text-slate-300 whitespace-pre-wrap">
          {runOutput || 'No test run output yet.'}
        </div>
      </div>

      <TerminalMatrix active={active} />
    </div>
  );
};

// --- MAIN LAYOUT ---

function IRERDashboardContent() {
  const [activeTool, setActiveTool] = useState('overview');
  const [systemStatus, setSystemStatus] = useState<'running' | 'idle'>('idle');
  const [activeSessionDir, setActiveSessionDir] = useState<string | null>(null);
  const [activeRunId, setActiveRunId] = useState<string | null>(null);
  const [activeRunMode, setActiveRunMode] = useState<'evolution' | 'backlog'>('evolution');
  const [queueRemaining, setQueueRemaining] = useState<number | null>(null);
  const [chunksCompleted, setChunksCompleted] = useState<number>(0);
  const [purgeCyclesCompleted, setPurgeCyclesCompleted] = useState<number>(0);
  const [runs, setRuns] = useState<any[]>([]);
  const [metricsHistory, setMetricsHistory] = useState<any[]>([]);
  const { connected: isConnected, events: telemetryEvents } = useTelemetry();
  const telemetryCursorRef = useRef(0);

  const chartData = runs.length > 0
    ? runs
    : metricsHistory.map((m, idx) => ({
        generation: idx,
        log_prime_sse: m.log_prime_sse ?? m.sse ?? null,
      }));

  useEffect(() => {
    const recent = telemetryEvents.slice(telemetryCursorRef.current);
    telemetryCursorRef.current = telemetryEvents.length;
    if (recent.length === 0) {
      return;
    }
    const metricEvents = recent
      .filter((item) => item && item.type === 'metrics')
      .map((item: any) => ({
        sse: item.sse,
        pcs: item.pcs,
        ic: item.ic,
        timestamp: item.timestamp || item.ts,
        log_prime_sse: item.sse,
      }));
    if (metricEvents.length > 0) {
      setMetricsHistory((prev) => [...prev, ...metricEvents].slice(-100));
    }
    const historyEvents = recent.filter((item) => item && Array.isArray(item.payload));
    historyEvents.forEach((item) => {
      if (item.type === 'pde_history') {
        setMetricsHistory((prev) => [...prev, ...(item.payload as any[])].slice(-100));
      }
    });
    const lastStatus = [...recent].reverse().find((item) => item && item.type === 'status' && typeof item.state === 'string');
    if (lastStatus) {
      setSystemStatus(lastStatus.state === 'running' ? 'running' : 'idle');
    }
    const lastRuns = [...recent].reverse().find((item) => item && Array.isArray(item.runs));
    if (lastRuns) {
      setRuns(lastRuns.runs as any[]);
    }
  }, [telemetryEvents]);

  useEffect(() => {
    let mounted = true;

    const refreshActiveRun = async () => {
      try {
        const response = await getRunStatus();
        if (!mounted) {
          return;
        }
        if (response.status === 'active' && response.active_run?.session_dir) {
          setActiveSessionDir(response.active_run.session_dir);
          setActiveRunId(response.active_run.run_id || null);
          setActiveRunMode((response.observability?.mode as 'evolution' | 'backlog') || 'evolution');
          setQueueRemaining(response.observability?.queue_remaining ?? null);
          setChunksCompleted(response.observability?.chunks_completed || 0);
          setPurgeCyclesCompleted(response.observability?.purge_cycles_completed || 0);
        } else {
          setActiveSessionDir(null);
          setActiveRunId(null);
          setActiveRunMode('evolution');
          setQueueRemaining(null);
          setChunksCompleted(0);
          setPurgeCyclesCompleted(0);
        }
      } catch (_err) {
        if (mounted) {
          setActiveSessionDir(null);
          setActiveRunId(null);
          setActiveRunMode('evolution');
          setQueueRemaining(null);
          setChunksCompleted(0);
          setPurgeCyclesCompleted(0);
        }
      }
    };

    void refreshActiveRun();
    const timer = window.setInterval(() => {
      void refreshActiveRun();
    }, activeSessionDir ? 4000 : 12000);

    return () => {
      mounted = false;
      window.clearInterval(timer);
    };
  }, [activeSessionDir]);

  const handleStart = async (payload: MissionStartPayload) => {
    const result = await startHunt(payload);
    if (result.status === 'success') {
      console.log('Hunt started:', result.job_id);
    } else {
      console.error('Hunt start failed:', result.message);
    }
  };

  const handleStop = async () => {
    const result = await stopHunt();
    console.log('Stop signal sent:', result);
  };

  const NAV_ITEMS = [
    { id: 'overview', label: 'Mission Overview', icon: <Activity size={16}/> },
    { id: 'control', label: 'Mission Control', icon: <Zap size={16}/> },
    { id: 'fleet', label: 'Fleet Telemetry', icon: <Wifi size={16}/> },
    { id: 'visual', label: 'Visual Analysis', icon: <ChevronRight size={16}/> },
    { id: 'ledger', label: 'Ledger Explorer', icon: <Database size={16}/> },
    { id: 'topology', label: 'Topological Exploration', icon: <HardDrive size={16}/> },
    { id: 'tensor', label: 'Tensor Point Cloud', icon: <HardDrive size={16}/> },
    { id: 'data', label: 'Data Vault', icon: <Database size={16}/> },
    { id: 'logs', label: 'Observer Logs', icon: <Terminal size={16}/> },
    { id: 'debug', label: 'Debug Console', icon: <Server size={16}/> },
  ];

  return (
    <div className="flex h-screen bg-slate-950 text-slate-200 font-sans overflow-hidden">
      {/* Sidebar */}
      <div className="w-64 bg-slate-900 border-r border-slate-800 flex flex-col">
        <div className="p-4 border-b border-slate-800 flex items-center gap-3">
          <div className="h-8 w-8 bg-blue-600 rounded flex items-center justify-center font-bold text-white shadow-lg">A</div>
          <div><h1 className="font-bold text-sm text-white">Aletheia/IRER</h1><div className="text-[10px] text-slate-400">V13.8 Command</div></div>
        </div>
        <div className="flex-1 p-2 space-y-1">
          {NAV_ITEMS.map(item => (
            <button key={item.id} onClick={() => setActiveTool(item.id)} className={`w-full flex items-center gap-3 px-3 py-2 text-xs font-bold rounded transition-colors ${activeTool === item.id ? 'bg-blue-600/20 text-blue-400 border border-blue-600/30' : 'text-slate-400 hover:bg-slate-800'}`}>
              {item.icon} {item.label}
            </button>
          ))}
        </div>
        <div className="p-4 border-t border-slate-800 text-[10px] text-slate-600 font-mono flex items-center gap-2">
          {isConnected ? <Wifi size={12} className="text-emerald-500" /> : <WifiOff size={12} className="text-amber-500" />}
          {isConnected ? "SYSTEM ONLINE" : "OFFLINE"}
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col min-w-0">
        <header className="h-14 border-b border-slate-800 bg-slate-900/50 flex items-center justify-between px-6 backdrop-blur">
          <h2 className="text-white font-bold capitalize flex items-center gap-2">
            <LayoutDashboard size={16} className="text-slate-500"/> / {activeTool.replace('_', ' ')}
          </h2>
          <div className="text-[11px] font-mono text-slate-300 truncate max-w-[60%]">
            {activeSessionDir ? `Active: ${activeSessionDir}` : 'No Active Hunt'}
          </div>
        </header>
        <main className="flex-1 p-6 overflow-y-auto overflow-x-hidden">
          {activeTool === 'overview' && (
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 min-h-full">
              <div className="lg:col-span-2 bg-slate-900 border border-slate-800 rounded p-4">
                <h3 className="text-white mb-4">Convergence Telemetry</h3>
                <div className="relative h-[90%] min-h-[280px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={chartData}>
                      <CartesianGrid stroke="#1e293b"/>
                      <XAxis dataKey="generation" stroke="#64748b"/>
                      <YAxis stroke="#64748b"/>
                      <Tooltip contentStyle={{backgroundColor: '#020617', borderColor: '#1e293b'}}/>
                      <Area type="monotone" dataKey="log_prime_sse" stroke="#3b82f6" fill="#3b82f622" isAnimationActive={false}/>
                    </AreaChart>
                  </ResponsiveContainer>
                  {chartData.length === 0 && (
                    <div className="absolute inset-0 flex items-center justify-center text-slate-500 text-sm">
                      Waiting for telemetry... start a hunt to populate convergence history.
                    </div>
                  )}
                </div>
              </div>
              <div className="bg-slate-900 border border-slate-800 rounded p-4">
                <h3 className="text-white mb-4">System Status</h3>
                <div className="text-sm text-slate-400">Active Runs: {runs.length}</div>
                <div className="text-sm text-slate-400 mt-1">Run Mode: <span className="text-slate-200 font-mono uppercase">{activeRunMode}</span></div>
                {activeRunMode === 'backlog' && (
                  <>
                    <div className="text-sm text-slate-400 mt-1">Queue Remaining: <span className="text-slate-200 font-mono">{queueRemaining ?? 'n/a'}</span></div>
                    <div className="text-sm text-slate-400 mt-1">Chunks Completed: <span className="text-slate-200 font-mono">{chunksCompleted}</span></div>
                    <div className="text-sm text-slate-400 mt-1">Purge Cycles: <span className="text-slate-200 font-mono">{purgeCyclesCompleted}</span></div>
                  </>
                )}
                <div className="mt-2 text-xs text-slate-500 font-mono break-all">
                  {activeSessionDir || 'No Active Hunt'}
                </div>
                <div className="mt-4 w-full rounded-lg border border-slate-700 bg-slate-950/80 p-3 min-h-[260px] flex items-center justify-center">
                  {systemStatus === 'running' ? (
                    <img
                      src="/holographic_solver_perfect_loop.gif"
                      alt="Solver active heartbeat"
                      className="w-full max-w-md rounded border border-cyan-500 shadow-[0_0_18px_rgba(6,182,212,0.45)]"
                    />
                  ) : (
                    <div className="w-full max-w-md h-[220px] rounded border border-slate-700 bg-slate-900/60 flex flex-col items-center justify-center text-slate-500 font-mono tracking-wide">
                      <div className="text-3xl mb-2">||</div>
                      <div>SOLVER STANDBY</div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}
          {activeTool === 'control' && <MissionControl onStart={handleStart} onStop={handleStop} />}
          {activeTool === 'fleet' && (
            <div className="space-y-6 h-full">
              <BacklogProgressWidget active={activeTool === 'fleet'} />
              <FleetTelemetry active={activeTool === 'fleet'} />
            </div>
          )}
          {activeTool === 'ledger' && (
            <LedgerExplorer
              activeRunId={activeRunId}
              activeSessionDir={activeSessionDir}
              defaultPageSize={50}
            />
          )}
          {activeTool === 'visual' && (
            <Suspense fallback={<div className="text-sm text-slate-400">Loading visual analysis...</div>}>
              <VisualAnalysis active={activeTool === 'visual'} />
            </Suspense>
          )}
          {activeTool === 'topology' && <TopologicalExplorer />}
          {activeTool === 'tensor' && <TensorPointCloud active={activeTool === 'tensor'} />}
          {activeTool === 'data' && <DataVault active={activeTool === 'data'} />}
          {activeTool === 'logs' && <ObserverConsole active={activeTool === 'logs'} />}
          {activeTool === 'debug' && <DebugConsole active={activeTool === 'debug'} />}
        </main>
      </div>
    </div>
  );
}

export default function IRERDashboard() {
  return (
    <TelemetryProvider>
      <IRERDashboardContent />
    </TelemetryProvider>
  );
}
