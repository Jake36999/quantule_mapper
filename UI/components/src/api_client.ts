export type StartHuntRequest = {
  staged_path: string;
  hunt_name?: string;
  origin?: string;
  staged_at?: string;
  staged_config_hash?: string;
  mode?: 'evolution' | 'backlog';
  backlog_source?: string;
};

export type StageHuntRequest = {
  config_path?: string;
  hunt_name?: string;
  generations?: number;
  batch_size?: number;
  origin?: string;
  population_size?: number;
  seeds_per_candidate?: number;
  n_grid?: number;
  t_steps?: number;
  dt?: number;
  l_domain?: number;
  collapse_threshold?: number;
  mutation_rate?: number;
  crossover_rate?: number;
  survival_fraction?: number;
  predator_sweep_frequency?: number;
  min_queue_depth?: number;
  poll_interval?: number;
  bleed_workers?: number;
  artifact_gc_min_age_seconds?: number;
  job_lease_timeout_seconds?: number;
  result_lease_timeout_seconds?: number;
  worker_heartbeat_ttl_seconds?: number;
  max_queue_depth?: number;
  mode?: 'evolution' | 'backlog';
  backlog_source?: string;
};

export type StageHuntResponse = {
  status: 'success' | 'error';
  config_hash?: string;
  staged_path?: string;
  staged_at?: string;
  message?: string;
};

export type ActiveRunResponse = {
  status: 'success' | 'error';
  active_run?: {
    hunt_name?: string;
    run_id?: string;
    session_dir?: string;
    db_file?: string;
    provenance_dir?: string;
    created_at?: string;
  };
  observability?: {
    mode?: 'evolution' | 'backlog';
    queue_remaining?: number | null;
    chunks_completed?: number;
    purge_cycles_completed?: number;
    backlog_source?: string | null;
    manifest_path?: string | null;
  };
  message?: string;
};

export type RunStatusResponse = {
  status: 'idle' | 'active' | 'completed' | 'errored' | 'unknown';
  active_run?: ActiveRunResponse['active_run'] | null;
  last_run?: unknown | null;
  observability?: ActiveRunResponse['observability'] | null;
  message?: string;
  updated_at?: string;
};

export type LedgerRunSummary = {
  hunt_name: string;
  run_id: string;
  session_name?: string;
  session_dir: string;
  db_file: string;
  provenance_dir?: string;
  created_at?: string;
  is_active?: boolean;
};

export type LedgerRunsResponse = {
  status: 'success' | 'error';
  runs: LedgerRunSummary[];
  message?: string;
};

export type LedgerRow = {
  config_hash: string;
  generation: number | null;
  status: string | null;
  fitness: number | null;
  origin: string | null;
  timestamp: string | null;
  log_prime_sse: number | null;
  bragg_peaks_detected: number | null;
  pcs: number | null;
  collapse_event_count: number | null;
  param_D: number | null;
  param_eta: number | null;
  param_rho_vac: number | null;
};

export type LedgerRowsResponse = {
  status: 'success' | 'error';
  run_id?: string | null;
  session_dir?: string | null;
  db_file?: string | null;
  rows: LedgerRow[];
  pagination?: {
    limit: number;
    offset: number;
    returned: number;
    total: number;
  } | null;
  message?: string;
};

export type FleetWorkerHealth = {
  worker_id: string;
  last_heartbeat_epoch: number;
  age_seconds: number;
  state: 'active' | 'stale' | 'historical';
  in_flight_claims: number;
};

export type FleetTelemetryResponse = {
  status: 'success' | 'error';
  timestamp?: string;
  queue_depth: number;
  dlq_count: number;
  active_workers: string[];
  stale_workers: string[];
  historical_workers?: string[];
  total_claims_processed: number;
  worker_heartbeat_ttl_seconds?: number;
  historical_worker_cutoff_seconds?: number | null;
  workers: FleetWorkerHealth[];
  message?: string;
};

export type DebugTerminalFeed = {
  id: string;
  name: string;
  path: string;
  lines: string[];
};

export type DebugTerminalResponse = {
  status: 'success' | 'error';
  feeds: DebugTerminalFeed[];
  combined: string[];
  updated_at: string;
};

export type DebugSuite = {
  id: string;
  label: string;
  command: string;
};

export type DebugSuitesResponse = {
  status: 'success' | 'error';
  suites: DebugSuite[];
};

export type DebugRunTestsResponse = {
  status: 'success' | 'fail' | 'error';
  suite?: string;
  exit_code?: number;
  duration_seconds?: number;
  output?: string;
  started_at?: string;
  message?: string;
};

export type VisualizerPlugin = {
  id: string;
  name?: string;
  script?: string;
  description?: string;
  script_exists?: boolean;
  resolved_path?: string;
};

export type VisualizerManifestResponse = {
  status: 'success' | 'error';
  manifest_path?: string;
  plugins?: VisualizerPlugin[];
  manifest?: {
    kind?: string;
    version?: string;
    plugins?: VisualizerPlugin[];
  };
  message?: string;
};

export type ArtifactSearchResult = {
  name: string;
  relative_path: string;
  source: string;
  size: number;
  modified: number;
  type: string;
};

export type ArtifactSearchResponse = {
  status: 'success' | 'error';
  query?: string;
  limit?: number;
  searched?: {
    active_session?: string | null;
    archive_runs?: string;
  };
  results: ArtifactSearchResult[];
  artifacts?: ArtifactSearchResult[];
  message?: string;
};

export type ExecuteVisualizerResponse = {
  status: 'success' | 'error';
  plugin_id?: string;
  artifact_path?: string;
  execution_id?: string;
  message?: string;
  mocked?: boolean;
  submitted_at?: string;
};

export type ExecuteVisualizerRequest = {
  plugin_id: string;
  artifact_path: string;
};

export type VisualRun = {
  run_id: string;
  config_hash?: string | null;
  git_commit?: string | null;
  seed?: number | null;
  run_type?: string | null;
  start_time?: string | null;
  end_time?: string | null;
  completed_at?: string | null;
  source_mtime?: number | null;
  status?: string | null;
  artifact_path: string;
  artifact_kind: string;
  hdf5_path?: string | null;
  rolling_buffer_expiry?: string | null;
  available_fields: string[];
  validation_status?: string | null;
  frame_count?: number;
  grid_shape?: number[] | null;
  source_datasets?: Record<string, string>;
};

export type VisualRunsResponse = {
  status: 'success' | 'error';
  runs: VisualRun[];
  message?: string;
};

export type VisualMetricsResponse = {
  status: 'success' | 'error';
  run_id: string;
  series: Array<Record<string, number>>;
  stability_metrics?: Record<string, unknown> | string | null;
  message?: string;
};

export type VisualRenderRequest = {
  tile: 'density_volume' | 'density_isosurface' | 'orthogonal_slices' | 'validation_dashboard';
  field: string;
  frame: number;
  resolution: number;
  threshold?: number;
  normalization?: 'linear' | 'none';
};

export type VisualRenderManifest = {
  cache_key: string;
  adapter_version: string;
  run_id: string;
  config_hash?: string | null;
  tile: VisualRenderRequest['tile'];
  field: string;
  frame: number;
  frame_count: number;
  shape: number[];
  dtype: string;
  normalization: string;
  scalar_range: { min: number; max: number };
  downsampling: { stride: number; max_resolution: number };
  threshold?: number | null;
  source_dataset?: string | null;
  source_artifact: string;
  assets: Record<string, string>;
  labels: { raw_or_derived: string };
};

export type VisualRenderResponse = {
  status: 'success' | 'error';
  manifest: VisualRenderManifest;
  message?: string;
};

export type VisualRerunResponse = {
  status: string;
  config_hash?: string;
  message?: string;
  gpu_gate?: {
    status?: string;
    last_utilization_percent?: number | null;
    idle_for_seconds?: number;
    waited_seconds?: number;
    threshold_percent?: number;
    required_idle_seconds?: number;
    message?: string;
  } | null;
  matched_run_ids?: string[];
};
// Centralized API client for React frontend
const deriveApiBase = (): string => {
  const configured = process.env.REACT_APP_API_BASE_URL;
  if (typeof window === 'undefined') {
    return configured && configured.trim() ? configured.trim() : '';
  }
  const { hostname, port } = window.location;
  const isLocalHost = hostname === 'localhost' || hostname === '127.0.0.1';

  if (configured && configured.trim()) {
    const configuredValue = configured.trim();
    try {
      const parsed = new URL(configuredValue);
      const parsedHost = parsed.hostname;
      const parsedIsLocal = parsedHost === 'localhost' || parsedHost === '127.0.0.1';
      if (isLocalHost && !parsedIsLocal) {
        return 'http://127.0.0.1:8000';
      }
      return configuredValue;
    } catch {
      if (configuredValue.startsWith('/')) {
        return '';
      }
      if (isLocalHost) {
        return 'http://127.0.0.1:8000';
      }
      return configuredValue;
    }
  }

  if (isLocalHost && port && port !== '8000') {
    return 'http://127.0.0.1:8000';
  }
  return '';
};

const API_BASE = deriveApiBase();

export async function apiFetch(path: string, options?: RequestInit) {
  const url = `${API_BASE}${path}`;
  return fetch(url, { ...options, credentials: 'include' });
}

// ---- Control Endpoints ----
export async function startHunt(payload: StartHuntRequest) {
  return apiFetch('/api/control/start', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  }).then(r => r.json());
}

export async function stageHuntConfig(payload: StageHuntRequest): Promise<StageHuntResponse> {
  return apiFetch('/api/control/stage', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  }).then(r => r.json());
}

export async function stopHunt() {
  return apiFetch('/api/control/stop', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' }
  }).then(r => r.json());
}

export async function getActiveRun(): Promise<ActiveRunResponse> {
  return apiFetch('/api/run/active').then(r => r.json());
}

export async function getRunStatus(): Promise<RunStatusResponse> {
  return apiFetch('/api/run/status').then(r => r.json());
}

export async function getDebugTerminals(lines = 200): Promise<DebugTerminalResponse> {
  return apiFetch(`/api/debug/terminals?lines=${encodeURIComponent(String(lines))}`).then(r => r.json());
}

export async function getDebugSuites(): Promise<DebugSuitesResponse> {
  return apiFetch('/api/debug/tests/suites').then(r => r.json());
}

export async function runDebugTests(suite: string): Promise<DebugRunTestsResponse> {
  return apiFetch('/api/debug/tests/run', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ suite }),
  }).then(r => r.json());
}

// ---- Data Endpoints ----
export async function listDataFiles() {
  return apiFetch('/api/data/files').then(r => r.json());
}

export async function getLedgerRuns(): Promise<LedgerRunsResponse> {
  return apiFetch('/api/data/ledger/runs').then(r => r.json());
}

export async function getLedgerRows(runId: string, limit = 50, offset = 0): Promise<LedgerRowsResponse> {
  const safeRunId = encodeURIComponent(runId);
  const params = new URLSearchParams({
    limit: String(limit),
    offset: String(offset),
  });
  return apiFetch(`/api/data/ledger/${safeRunId}?${params.toString()}`).then(r => r.json());
}

export async function getSystemTelemetry(ttlSeconds?: number, includeHistorical = false): Promise<FleetTelemetryResponse> {
  const params = new URLSearchParams();
  if (typeof ttlSeconds === 'number' && Number.isFinite(ttlSeconds)) {
    params.set('ttl_seconds', String(ttlSeconds));
  }
  if (includeHistorical) {
    params.set('include_historical', 'true');
  }
  const suffix = params.toString() ? `?${params.toString()}` : '';
  return apiFetch(`/api/system/telemetry${suffix}`).then(r => r.json());
}

export async function downloadData(filePath: string) {
  return apiFetch(`/api/data/download/${encodeURI(filePath)}`);
}

export async function getVisualizerPlugins(): Promise<VisualizerManifestResponse> {
  return apiFetch('/api/plugins/visualizers').then(r => r.json());
}

export async function searchArtifacts(query: string): Promise<ArtifactSearchResponse> {
  const q = encodeURIComponent(query || '');
  return apiFetch(`/api/data/artifacts/search?q=${q}`).then(r => r.json());
}

export async function getTensorBuffer(configHash: string): Promise<ArrayBuffer> {
  const response = await apiFetch(`/api/data/tensor/${encodeURIComponent(configHash)}`);
  if (!response.ok) {
    throw new Error(`Tensor fetch failed: ${response.status}`);
  }
  return response.arrayBuffer();
}

export async function getPointCloudBuffer(
  configHash: string,
  datasetName: string,
  threshold = 0.05,
): Promise<ArrayBuffer> {
  const response = await apiFetch(
    `/api/data/pointcloud/hash/${encodeURIComponent(configHash)}/${encodeURIComponent(datasetName)}?threshold=${encodeURIComponent(String(threshold))}`,
  );
  if (!response.ok) {
    throw new Error(`Point cloud fetch failed: ${response.status}`);
  }
  return response.arrayBuffer();
}

export async function executeVisualizer(pluginId: string, artifactPath: string): Promise<ExecuteVisualizerResponse> {
  const payload: ExecuteVisualizerRequest = {
    plugin_id: pluginId,
    artifact_path: artifactPath,
  };
  return apiFetch('/api/plugins/execute', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  }).then(r => r.json());
}

export async function listVisualRuns(limit = 50, includeArtifacts = false): Promise<VisualRunsResponse> {
  const params = new URLSearchParams({ limit: String(limit) });
  if (includeArtifacts) {
    params.set('include_artifacts', 'true');
  }
  return apiFetch(`/api/visual/runs?${params.toString()}`).then(r => r.json());
}

export async function getVisualMetrics(runId: string): Promise<VisualMetricsResponse> {
  return apiFetch(`/api/visual/runs/${encodeURIComponent(runId)}/metrics`).then(r => r.json());
}

export async function renderVisualRun(runId: string, payload: VisualRenderRequest): Promise<VisualRenderResponse> {
  return apiFetch(`/api/visual/runs/${encodeURIComponent(runId)}/render`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  }).then(r => r.json());
}

export async function requestVisualRerun(configHash: string): Promise<VisualRerunResponse> {
  return apiFetch('/api/visual/rerun', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      config_hash: configHash,
      wait_for_gpu_idle: true,
      gpu_idle_percent: 20,
      gpu_idle_seconds: 60,
      gpu_wait_timeout_seconds: 900,
    }),
  }).then(r => r.json());
}

// Example usage:
// const res = await apiFetch('/api/stream/status');
