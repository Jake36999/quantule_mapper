// Enums matching V11_5_F3 Sentinel Codes
export enum SentinelCode {
  SUCCESS = 0,
  ARTIFACT_ERROR = 1,
  METRIC_ERROR = 2,
  SCHEMA_ERROR = 3,
}

// V12_F2: Adaptive Steering Strategies
export enum SteeringStrategy {
  MANUAL = 'MANUAL',
  STABILIZER = 'STABILIZER',        // Dampen instability (Target: Low H-Norm + Low IC)
  SEARCH = 'SEARCH',                // Hunt for structure (Target: High PLI/Novelty)
  CONVERGENCE = 'CONVERGENCE'       // Golden Path (Target: High PCS + Low H-Norm)
}

// V12_F4: Designed Experiment Profiles
export enum ExperimentProfile {
  NONE = 'NONE',
  GEOMETRIC_STRESS = 'GEOMETRIC_STRESS', // Maximize Stability + Deform Geometry
  SPECTRAL_LOCK = 'SPECTRAL_LOCK',       // Regimes with log-prime alignment < epsilon
  ENTROPY_HARVEST = 'ENTROPY_HARVEST'    // Maximize IC without crash
}

export interface ProfileDefinition {
    id: string;
    name: string;
    description: string;
    weights: Record<string, number>;
    mode: string;
}

export interface SteeringSurface {
    axis_x: string;
    axis_y: string;
    data: Array<Array<{x: number, y: number, fitness: number, metrics: any}>>;
}

export enum SemanticClass {
  GRAVITY_WELL = 'GRAVITY_WELL',
  ENTROPY_CASCADE = 'ENTROPY_CASCADE',
  PRIME_RESONANCE = 'PRIME_RESONANCE',
  NULL_MANIFOLD = 'NULL_MANIFOLD'
}

export interface SimulationConfig {
  evolutionary: {
    generations: number;
    population: number;
    mutation_rate: number;
    w_sse: number;
    w_stab: number;
  };
  physics: {
    grid_size: number;
    time_steps: number;
    dt: number;
    sncgl_epsilon: number; 
    sncgl_lambda: number;  
    sncgl_g_nonlocal: number; 
    sdg_alpha: number; 
    sdg_rho_vac: number; 
  };
  system: {
    max_runtime_minutes: number;
    auto_save_interval: number;
  };
  profile_id?: string; // V12 Extension
}

export interface SimulationRun {
  uuid: string;
  config_hash: string;
  generation: number;
  timestamp: string;
  status: SentinelCode;
  
  log_prime_sse: number;
  
  h_norm_l2: number;
  pcs_score: number;
  noether_check: number;

  // V12.0 Field Awareness Metrics
  pli_score: number;
  ic_score: number;
  p11_drift?: number;
  topo_entropy?: number;
  chiral_order?: number;
  
  Phase_Coherence_Score_PCS?: number;
  Principled_Localization_Index_PLI?: number;
  Informational_Compressibility_IC?: number;

  visual_type?: VisualTaxonomy;
  quantule_id?: string;
  analogue_match?: StructuralAnalogy;

  composite_fitness: number;
  
  artifacts: {
    config: string;
    hdf5: string;
    provenance: string;
  };
}

// --- ALETHEIA INTEGRATION TYPES ---

export interface SpectralDataPoint {
  k: number;
  amplitude: number;
  target: number;
  null_hypothesis: number;
}

export interface SimulationMetrics {
  timestamp: number;
  log_prime_sse: number;       // Spectral fidelity (Lower is better)
  hamiltonian_norm_L2: number; // Geometric stability (Lower is better)
  pai: number;                 // Phenomenological Stability Index (1.0 = Healthy, 0.0 = Crash)
  phase_coherence: number;     // Order parameter (PCS)
  betti_0: number;             // Topological "spots"
  betti_1: number;             // Topological "tunnels"
  null_hypothesis_sse: number; // For Falsifiability Protocol
  job_uuid?: string;
}

export interface ProvenanceArtifact {
  run_id: string;
  config_hash: string;
  timestamp: string;
  schema_version: string;
  spectral_validation_results: {
    log_prime_sse: number;
    sse_total_directional?: number;
    validation_status: "VALID" | "INVALID";
  };
  aletheia_sensorium_metrics: {
    Phase_Coherence_Score_PCS: number;
    Principled_Localization_Index_PLI: number;
    Informational_Compressibility_IC: number;
  };
  geometric_stability: {
      h_norm_l2: number;
      p11_drift: number;
  };
  falsification_flags: {
    sentinel_code: number;
    sse_null_phase_scramble: number;
    sse_null_target_shuffle: number;
  };
}

export interface AIAnalysis {
  id: string;
  timestamp: string;
  status: 'STABLE' | 'CRITICAL' | 'OPTIMIZED' | 'ANALYZING';
  message: string;
  recommendation?: any;
  confidence: number;
}

export interface ExperimentalCampaign {
  title: string;
  rationale: string;
  target_metrics: string[];
  config_overrides: any;
}

export interface AnalystFinding {
  category: 'CORRELATION' | 'ANOMALY' | 'TREND' | 'SEMANTIC';
  observation: string;
  significance: 'HIGH' | 'MEDIUM' | 'LOW';
  semantic_class?: SemanticClass;
}

export interface AnalystReport {
  report_id: string;
  generated_at: string;
  run_window_start: number;
  run_window_end: number;
  executive_summary: string;
  findings: AnalystFinding[];
  proposed_campaign: ExperimentalCampaign;
}

export interface TelemetryPoint {
  generation: number;
  best_sse: number;
  best_h_norm: number;
  best_pcs: number;
  best_uuid: string;
}

export interface SystemLog {
  id: string;
  timestamp: string;
  level: 'INFO' | 'WARNING' | 'CRITICAL' | 'STEER';
  message: string;
  source: 'ORCHESTRATOR' | 'WORKER' | 'VALIDATOR' | 'FLEET_MANAGER' | 'STEERING_ENGINE' | 'ANALYST_AGENT';
}

export interface DCONode {
  id: string;
  name: string;
  type: 'ORCHESTRATION' | 'PHYSICS' | 'VALIDATION';
  status: 'IDLE' | 'RUNNING' | 'COMPLETE' | 'FAILED';
  assigned_vm: string;
  artifact_out: string;
}

export interface DCOEdge {
  from: string;
  to: string;
  label: string;
}

export interface FleetVM {
  id: string;
  name: string;
  role: 'CONTROL_PLANE' | 'WORKER_GPU' | 'WORKER_CPU';
  status: 'ONLINE' | 'OFFLINE' | 'BUSY';
  ip: string;
  load: number;
  current_task?: string;
}

export type DevTaskStatus = 'PLANNING' | 'PENDING_APPROVAL' | 'EXECUTING' | 'COMPLETED' | 'REJECTED' | 'FAILED';

export interface FileOperation {
  action: 'MODIFY' | 'CREATE' | 'DELETE';
  path: string;
  description: string;
}

export interface AgentPlan {
  phases: {
    id: string;
    lens: string;
    status: 'PENDING' | 'ACTIVE' | 'DONE';
    steps: string[];
  }[];
  file_ops: FileOperation[];
}

export interface CodeDiff {
  file_path: string;
  original_code: string;
  modified_code: string;
}

export interface DevelopmentTask {
  task_id: string;
  status: DevTaskStatus;
  user_prompt: string;
  agent_plan: AgentPlan | null;
  code_diff: CodeDiff | null;
  explanation: string | null;
  execution_log: string[];
  created_at: string;
  updated_at: string;
}

export enum VisualTaxonomy {
  RADIAL_BURST = 'RADIAL_BURST',
  SPIRAL_DECAY = 'SPIRAL_DECAY',
  NODE_CASCADE = 'NODE_CASCADE',
  INTERLOCKED_ECHOES = 'INTERLOCKED_ECHOES',
  SPLIT_CORE = 'SPLIT_CORE',
  UNDEFINED = 'UNDEFINED'
}

export interface StructuralAnalogy {
  analogue_name: string;
  physical_principle: string;
  similarity_score: number;
}