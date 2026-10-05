import React, { useEffect, useMemo, useRef, useState } from 'react';
import Plot from 'react-plotly.js';
import { Copy, RefreshCw, RotateCcw, Layers, Activity, Pause, Play, SlidersHorizontal } from 'lucide-react';

import '@kitware/vtk.js/Rendering/Profiles/Geometry';
import '@kitware/vtk.js/Rendering/Profiles/Volume';
import vtkActor from '@kitware/vtk.js/Rendering/Core/Actor';
import vtkColorTransferFunction from '@kitware/vtk.js/Rendering/Core/ColorTransferFunction';
import vtkDataArray from '@kitware/vtk.js/Common/Core/DataArray';
import vtkGenericRenderWindow from '@kitware/vtk.js/Rendering/Misc/GenericRenderWindow';
import vtkImageData from '@kitware/vtk.js/Common/DataModel/ImageData';
import vtkMapper from '@kitware/vtk.js/Rendering/Core/Mapper';
import vtkPiecewiseFunction from '@kitware/vtk.js/Common/DataModel/PiecewiseFunction';
import vtkPolyData from '@kitware/vtk.js/Common/DataModel/PolyData';
import vtkVolume from '@kitware/vtk.js/Rendering/Core/Volume';
import vtkVolumeMapper from '@kitware/vtk.js/Rendering/Core/VolumeMapper';

import {
  apiFetch,
  getVisualMetrics,
  listVisualRuns,
  renderVisualRun,
  requestVisualRerun,
  VisualMetricsResponse,
  VisualRenderManifest,
  VisualRenderRequest,
  VisualRun,
} from '../api_client';

type TileId = VisualRenderRequest['tile'];
type PlaybackState = 'idle' | 'playing' | 'paused' | 'complete';

const TILE_OPTIONS: Array<{ id: TileId; label: string; field: string; tone: string; description: string }> = [
  { id: 'density_volume', label: 'Density volume', field: 'rho', tone: '#22d3ee', description: 'Transparent scalar volume' },
  { id: 'density_isosurface', label: 'Density isosurface', field: 'rho', tone: '#f59e0b', description: 'Threshold mesh surface' },
  { id: 'orthogonal_slices', label: 'Orthogonal slices', field: 'rho', tone: '#a78bfa', description: 'XY / XZ / YZ planes' },
  { id: 'validation_dashboard', label: 'Validation dashboard', field: 'rho', tone: '#34d399', description: 'Conservation telemetry' },
];

const formatBytes = (path: string): string => path.split(/[\\/]/).slice(-3).join('/');

const shortHash = (value?: string | null): string => {
  if (!value) {
    return 'unknown';
  }
  return value.length > 12 ? `${value.slice(0, 12)}...` : value;
};

const hashHue = (value?: string | null): number => {
  const source = value || 'unknown';
  let acc = 0;
  for (let idx = 0; idx < source.length; idx += 1) {
    acc = (acc * 31 + source.charCodeAt(idx)) % 360;
  }
  return acc;
};

const runCompletedAt = (run?: VisualRun | null): string | null => {
  if (!run) {
    return null;
  }
  if (run.completed_at) {
    return run.completed_at;
  }
  if (run.end_time) {
    return run.end_time;
  }
  if (run.start_time) {
    return run.start_time;
  }
  if (typeof run.source_mtime === 'number' && Number.isFinite(run.source_mtime)) {
    return new Date(run.source_mtime * 1000).toISOString();
  }
  return null;
};

const formatRunCompletedAt = (run?: VisualRun | null): string => {
  const value = runCompletedAt(run);
  if (!value) {
    return 'Completion time unknown';
  }
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return value;
  }
  return parsed.toLocaleString(undefined, {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
};

const metricX = (series: Array<Record<string, number>>) => series.map((row, idx) => row.step ?? idx);

function ViewLegend({
  tile,
  onTileChange,
  manifest,
}: {
  tile: TileId;
  onTileChange: (tile: TileId) => void;
  manifest: VisualRenderManifest | null;
}) {
  return (
    <div className="bg-slate-900 border border-slate-800 rounded p-3 min-h-[520px]">
      <div className="text-xs font-bold text-white flex items-center gap-2 mb-3">
        <SlidersHorizontal size={14} className="text-cyan-300" /> Select view
      </div>
      <div className="space-y-2">
        {TILE_OPTIONS.map((option) => {
          const selected = option.id === tile;
          return (
            <button
              type="button"
              key={option.id}
              onClick={() => onTileChange(option.id)}
              className={`w-full text-left border rounded p-2 transition-colors ${selected ? 'bg-slate-800 border-cyan-600 text-white' : 'bg-slate-950 border-slate-800 text-slate-400 hover:border-slate-600'}`}
            >
              <span className="flex items-center gap-2">
                <span className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: option.tone }} />
                <span className="text-xs font-bold">{option.label}</span>
              </span>
              <span className="block text-[10px] text-slate-500 mt-1">{option.description}</span>
            </button>
          );
        })}
      </div>
      <div className="mt-4 border-t border-slate-800 pt-3 space-y-2">
        <div className="text-[10px] uppercase tracking-wide text-slate-500">Legend</div>
        <div className="flex items-center gap-2 text-[11px] text-slate-300">
          <span className="h-2 w-8 rounded bg-gradient-to-r from-cyan-800 via-cyan-400 to-amber-200" />
          Density intensity
        </div>
        <div className="flex items-center gap-2 text-[11px] text-slate-300">
          <span className="h-2 w-8 rounded border border-slate-500 bg-slate-900" />
          Scene grid plane
        </div>
        <div className="text-[11px] text-slate-500">
          Frame {manifest ? `${manifest.frame + 1} / ${manifest.frame_count}` : 'not rendered'}
        </div>
      </div>
    </div>
  );
}

function SliceCanvases({ manifest }: { manifest: VisualRenderManifest }) {
  const [slices, setSlices] = useState<Record<string, number[][]> | null>(null);

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      const url = manifest.assets.slices;
      if (!url) {
        setSlices(null);
        return;
      }
      const response = await apiFetch(url);
      const payload = await response.json();
      if (!cancelled) {
        setSlices(payload);
      }
    };
    void load();
    return () => {
      cancelled = true;
    };
  }, [manifest.cache_key, manifest.assets.slices]);

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-3 h-full">
      {(['xy', 'xz', 'yz'] as const).map((key) => (
        <SliceCanvas key={key} label={key.toUpperCase()} values={slices?.[key] || null} />
      ))}
    </div>
  );
}

function SliceCanvas({ label, values }: { label: string; values: number[][] | null }) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !values || values.length === 0 || values[0].length === 0) {
      return;
    }
    const height = values.length;
    const width = values[0].length;
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    if (!ctx) {
      return;
    }
    const image = ctx.createImageData(width, height);
    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        const v = Math.max(0, Math.min(1, Number(values[y][x] || 0)));
        const idx = (y * width + x) * 4;
        image.data[idx] = Math.round(30 + v * 225);
        image.data[idx + 1] = Math.round(80 + v * 145);
        image.data[idx + 2] = Math.round(140 + v * 90);
        image.data[idx + 3] = 255;
      }
    }
    ctx.putImageData(image, 0, 0);
  }, [values]);

  return (
    <div className="bg-slate-950 border border-slate-800 rounded p-2 min-h-[220px] flex flex-col">
      <div className="text-[11px] text-slate-400 font-mono mb-2">{label}</div>
      <div className="flex-1 min-h-0 grid place-items-center">
        <canvas ref={canvasRef} className="max-w-full max-h-full w-full h-auto image-render-pixelated" />
      </div>
    </div>
  );
}

function VTKViewport({ manifest }: { manifest: VisualRenderManifest | null }) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const vtkRef = useRef<any>(null);
  const cameraStateRef = useRef<{ position: number[]; focalPoint: number[]; viewUp: number[] } | null>(null);
  const [message, setMessage] = useState('Select a run and render a tile.');

  useEffect(() => {
    if (!containerRef.current || vtkRef.current) {
      return;
    }
    const genericRenderWindow = vtkGenericRenderWindow.newInstance({ background: [0.015, 0.022, 0.035] });
    genericRenderWindow.setContainer(containerRef.current);
    genericRenderWindow.resize();
    vtkRef.current = genericRenderWindow;

    const resize = () => genericRenderWindow.resize();
    window.addEventListener('resize', resize);
    return () => {
      window.removeEventListener('resize', resize);
      genericRenderWindow.delete();
      vtkRef.current = null;
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      const genericRenderWindow = vtkRef.current;
      if (!genericRenderWindow || !manifest) {
        return;
      }
      const renderer = genericRenderWindow.getRenderer();
      const renderWindow = genericRenderWindow.getRenderWindow();
      const camera = renderer.getActiveCamera();
      if (camera && renderer.getActors().length + renderer.getVolumes().length > 0) {
        cameraStateRef.current = {
          position: camera.getPosition(),
          focalPoint: camera.getFocalPoint(),
          viewUp: camera.getViewUp(),
        };
      }

      renderer.removeAllViewProps();
      setMessage('Preparing render asset...');

      try {
        if (manifest.tile === 'density_volume') {
          const response = await apiFetch(manifest.assets.volume);
          const values = new Float32Array(await response.arrayBuffer());
          if (cancelled) {
            return;
          }
          const [z, y, x] = manifest.shape;
          const imageData = vtkImageData.newInstance();
          imageData.setDimensions(x, y, z);
          imageData.setSpacing([1, 1, 1]);
          imageData.getPointData().setScalars(
            vtkDataArray.newInstance({ name: manifest.field, values, numberOfComponents: 1 }),
          );

          const mapper = vtkVolumeMapper.newInstance();
          mapper.setInputData(imageData);
          mapper.setSampleDistance(0.8);

          const volume = vtkVolume.newInstance();
          volume.setMapper(mapper);

          const ctf = vtkColorTransferFunction.newInstance();
          ctf.addRGBPoint(0.0, 0.02, 0.08, 0.14);
          ctf.addRGBPoint(0.35, 0.1, 0.55, 0.75);
          ctf.addRGBPoint(0.7, 0.95, 0.72, 0.28);
          ctf.addRGBPoint(1.0, 1.0, 0.96, 0.78);

          const ofun = vtkPiecewiseFunction.newInstance();
          ofun.addPoint(0.0, 0.0);
          ofun.addPoint(0.18, 0.03);
          ofun.addPoint(0.55, 0.18);
          ofun.addPoint(1.0, 0.72);

          volume.getProperty().setRGBTransferFunction(0, ctf);
          volume.getProperty().setScalarOpacity(0, ofun);
          volume.getProperty().setScalarOpacityUnitDistance(0, 2.5);
          volume.getProperty().setInterpolationTypeToFastLinear();
          volume.getProperty().setShade(false);
          renderer.addVolume(volume);
        } else if (manifest.tile === 'density_isosurface') {
          const [pointsResponse, normalsResponse, polysResponse] = await Promise.all([
            apiFetch(manifest.assets.points),
            apiFetch(manifest.assets.normals),
            apiFetch(manifest.assets.polys),
          ]);
          const points = new Float32Array(await pointsResponse.arrayBuffer());
          const normals = new Float32Array(await normalsResponse.arrayBuffer());
          const polys = new Uint32Array(await polysResponse.arrayBuffer());
          if (cancelled) {
            return;
          }

          const polyData = vtkPolyData.newInstance();
          polyData.getPoints().setData(points, 3);
          polyData.getPolys().setData(polys);
          polyData.getPointData().setNormals(
            vtkDataArray.newInstance({ name: 'normals', values: normals, numberOfComponents: 3 }),
          );

          const mapper = vtkMapper.newInstance();
          mapper.setInputData(polyData);
          const actor = vtkActor.newInstance();
          actor.setMapper(mapper);
          actor.getProperty().setColor(0.33, 0.82, 0.88);
          actor.getProperty().setOpacity(0.88);
          actor.getProperty().setSpecular(0.25);
          actor.getProperty().setSpecularPower(18);
          renderer.addActor(actor);
        }

        const saved = cameraStateRef.current;
        if (saved) {
          const nextCamera = renderer.getActiveCamera();
          nextCamera.setPosition(...saved.position);
          nextCamera.setFocalPoint(...saved.focalPoint);
          nextCamera.setViewUp(...saved.viewUp);
        } else {
          renderer.resetCamera();
        }
        renderWindow.render();
        setMessage('');
      } catch (err) {
        setMessage(err instanceof Error ? err.message : 'Render failed.');
      }
    };

    if (manifest?.tile === 'orthogonal_slices' || manifest?.tile === 'validation_dashboard') {
      const genericRenderWindow = vtkRef.current;
      if (genericRenderWindow) {
        genericRenderWindow.getRenderer().removeAllViewProps();
        genericRenderWindow.getRenderWindow().render();
      }
      return;
    }

    void load();
    return () => {
      cancelled = true;
    };
  }, [manifest]);

  return (
    <div className="relative h-full min-h-[520px] bg-slate-950 border border-slate-800 rounded overflow-hidden">
      <div
        className="absolute inset-0 opacity-70"
        style={{
          backgroundImage:
            'linear-gradient(rgba(34,211,238,0.10) 1px, transparent 1px), linear-gradient(90deg, rgba(34,211,238,0.10) 1px, transparent 1px), radial-gradient(circle at 50% 35%, rgba(14,165,233,0.20), transparent 45%)',
          backgroundSize: '36px 36px, 36px 36px, 100% 100%',
        }}
      />
      <div ref={containerRef} className="absolute inset-0" />
      <div
        className="pointer-events-none absolute inset-0 opacity-35"
        style={{
          backgroundImage:
            'linear-gradient(rgba(125,211,252,0.18) 1px, transparent 1px), linear-gradient(90deg, rgba(125,211,252,0.18) 1px, transparent 1px)',
          backgroundSize: '48px 48px',
        }}
      />
      <div className="pointer-events-none absolute left-4 bottom-4 right-4 h-px bg-cyan-300/40" />
      <div className="pointer-events-none absolute left-4 bottom-4 top-4 w-px bg-cyan-300/30" />
      <div className="pointer-events-none absolute right-4 top-4 text-[10px] font-mono text-cyan-200/70">
        3D FIELD VIEW
      </div>
      {message ? (
        <div className="absolute inset-0 grid place-items-center text-sm text-slate-400 bg-slate-950/70">
          {message}
        </div>
      ) : null}
    </div>
  );
}

function ValidationDashboard({ metrics, compact = false }: { metrics: VisualMetricsResponse | null; compact?: boolean }) {
  const series = metrics?.series || [];
  const x = metricX(series);
  const energy = series.map((row) => row.energy ?? null);
  const invariant = series.map((row) => row.C_invariant ?? null);

  return (
    <div className={`${compact ? 'h-full min-h-0' : 'h-full min-h-[520px]'} bg-slate-950 border border-slate-800 rounded p-3`}>
      <Plot
        data={[
          { x, y: energy, type: 'scatter', mode: 'lines', name: 'energy', line: { color: '#67e8f9' } },
          { x, y: invariant, type: 'scatter', mode: 'lines', name: 'C invariant', line: { color: '#fbbf24' } },
        ]}
        layout={{
          autosize: true,
          paper_bgcolor: '#020617',
          plot_bgcolor: '#020617',
          font: { color: '#cbd5e1' },
          margin: { l: 44, r: 18, t: 18, b: 38 },
          xaxis: { gridcolor: '#1e293b', title: { text: 'step' } },
          yaxis: { gridcolor: '#1e293b' },
          legend: { orientation: 'h' },
        }}
        config={{ displayModeBar: false, responsive: true }}
        style={{ width: '100%', height: '100%' }}
        useResizeHandler
      />
    </div>
  );
}

export default function VisualAnalysis({ active }: { active: boolean }) {
  const [runs, setRuns] = useState<VisualRun[]>([]);
  const [selectedRunId, setSelectedRunId] = useState('');
  const [tile, setTile] = useState<TileId>('density_volume');
  const [frame, setFrame] = useState(0);
  const [resolution, setResolution] = useState(96);
  const [threshold, setThreshold] = useState(0.45);
  const [normalization, setNormalization] = useState<'linear' | 'none'>('linear');
  const [includeArtifacts, setIncludeArtifacts] = useState(false);
  const [manifest, setManifest] = useState<VisualRenderManifest | null>(null);
  const [metrics, setMetrics] = useState<VisualMetricsResponse | null>(null);
  const [configHash, setConfigHash] = useState('');
  const [status, setStatus] = useState('Ready.');
  const [loading, setLoading] = useState(false);
  const [playbackState, setPlaybackState] = useState<PlaybackState>('idle');
  const [frameUnlocked, setFrameUnlocked] = useState(false);
  const playbackRunRef = useRef(0);

  const selectedRun = useMemo(
    () => runs.find((run) => run.run_id === selectedRunId) || null,
    [runs, selectedRunId],
  );
  const selectedTile = TILE_OPTIONS.find((option) => option.id === tile) || TILE_OPTIONS[0];
  const frameCount = Math.max(1, selectedRun?.frame_count || 1);

  const refreshRuns = async () => {
    setLoading(true);
    setStatus('Scanning renderable artifacts...');
    try {
      const response = await listVisualRuns(80, includeArtifacts);
      const nextRuns = response.runs || [];
      setRuns(nextRuns);
      if (!selectedRunId && nextRuns.length > 0) {
        setSelectedRunId(nextRuns[0].run_id);
      } else if (selectedRunId && !nextRuns.some((run) => run.run_id === selectedRunId)) {
        setSelectedRunId(nextRuns[0]?.run_id || '');
        setManifest(null);
      }
      setStatus(nextRuns.length ? `Loaded ${nextRuns.length} renderable run${nextRuns.length === 1 ? '' : 's'}.` : 'No renderable completed runs found.');
    } catch (err) {
      setStatus(err instanceof Error ? err.message : 'Failed to load visual runs.');
    } finally {
      setLoading(false);
    }
  };

  const loadMetrics = async (runId: string) => {
    try {
      const response = await getVisualMetrics(runId);
      setMetrics(response);
    } catch {
      setMetrics(null);
    }
  };

  const renderFrame = async (targetFrame: number, reason: 'manual' | 'playback' | 'view' = 'manual') => {
    if (!selectedRun) {
      setStatus('Select a completed run first.');
      return;
    }
    setLoading(true);
    setStatus(`${reason === 'playback' ? 'Playing' : 'Rendering'} ${selectedTile.label.toLowerCase()} frame ${targetFrame + 1}/${frameCount}...`);
    try {
      if (tile === 'validation_dashboard') {
        await loadMetrics(selectedRun.run_id);
        setManifest({
          cache_key: `dashboard-${selectedRun.run_id}`,
          adapter_version: 'visual-analysis-v1',
          run_id: selectedRun.run_id,
          config_hash: selectedRun.config_hash,
          tile,
          field: 'rho',
          frame: targetFrame,
          frame_count: frameCount,
          shape: selectedRun.grid_shape || [],
          dtype: 'float32',
          normalization: 'linear',
          scalar_range: { min: 0, max: 0 },
          downsampling: { stride: 1, max_resolution: resolution },
          threshold: null,
          source_dataset: null,
          source_artifact: selectedRun.artifact_path,
          assets: {},
          labels: { raw_or_derived: 'VALIDATION TELEMETRY' },
        });
      } else {
        const response = await renderVisualRun(selectedRun.run_id, {
          tile,
          field: selectedTile.field,
          frame: targetFrame,
          resolution,
          threshold,
          normalization,
        });
        setManifest(response.manifest);
        void loadMetrics(selectedRun.run_id);
      }
      setStatus(reason === 'playback' ? 'Playback running.' : 'Render ready.');
    } catch (err) {
      setPlaybackState('paused');
      setFrameUnlocked(true);
      setStatus(err instanceof Error ? err.message : 'Render request failed.');
    } finally {
      setLoading(false);
    }
  };

  const startPlayback = async () => {
    if (!selectedRun) {
      setStatus('Select a completed run first.');
      return;
    }
    playbackRunRef.current += 1;
    setFrameUnlocked(false);
    setPlaybackState('playing');
    setFrame(0);
    await renderFrame(0, 'playback');
    if (frameCount <= 1) {
      setPlaybackState('complete');
      setFrameUnlocked(true);
      setStatus('Single-frame run rendered. Frame selection is enabled.');
    }
  };

  const pausePlayback = () => {
    playbackRunRef.current += 1;
    setPlaybackState('paused');
    setFrameUnlocked(true);
    setStatus('Playback paused. Frame selection is enabled.');
  };

  const resumePlayback = () => {
    if (!selectedRun) {
      return;
    }
    playbackRunRef.current += 1;
    setFrameUnlocked(false);
    setPlaybackState('playing');
    setStatus('Playback resumed.');
    if (frame >= frameCount - 1) {
      setFrame(0);
      void renderFrame(0, 'playback');
    }
  };

  const submitRerun = async () => {
    const hash = configHash.trim();
    if (!hash) {
      setStatus('Enter a config hash.');
      return;
    }
    setLoading(true);
    setStatus('Waiting for GPU utilization to stay below 20% for 60s before rerun gating...');
    try {
      const response = await requestVisualRerun(hash);
      const gpuSuffix = response.gpu_gate?.status
        ? ` GPU gate: ${response.gpu_gate.status}${typeof response.gpu_gate.last_utilization_percent === 'number' ? ` (${response.gpu_gate.last_utilization_percent}%)` : ''}.`
        : '';
      setStatus(`${response.message || response.status}${gpuSuffix}`);
    } catch (err) {
      setStatus(err instanceof Error ? err.message : 'Rerun request failed.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (active) {
      void refreshRuns();
    }
  }, [active, includeArtifacts]);

  useEffect(() => {
    if (selectedRun) {
      playbackRunRef.current += 1;
      setPlaybackState('idle');
      setFrameUnlocked(false);
      setManifest(null);
      setFrame((prev) => Math.min(prev, frameCount - 1));
      setConfigHash(selectedRun.config_hash || '');
      void loadMetrics(selectedRun.run_id);
    }
  }, [selectedRunId]);

  useEffect(() => {
    if (!active || !selectedRun || playbackState === 'idle') {
      return;
    }
    const timer = window.setTimeout(() => {
      void renderFrame(frame, playbackState === 'playing' ? 'playback' : 'view');
    }, 350);
    return () => window.clearTimeout(timer);
  }, [active, selectedRunId, tile, resolution, threshold, normalization]);

  useEffect(() => {
    if (!active || !selectedRun || playbackState !== 'playing' || loading || frameCount <= 1) {
      return;
    }
    const token = playbackRunRef.current;
    const timer = window.setTimeout(() => {
      if (token !== playbackRunRef.current) {
        return;
      }
      const nextFrame = frame + 1;
      if (nextFrame >= frameCount) {
        setPlaybackState('complete');
        setFrameUnlocked(true);
        setStatus('Playback complete. Frame selection is enabled.');
        return;
      }
      setFrame(nextFrame);
      void renderFrame(nextFrame, 'playback');
    }, 850);
    return () => window.clearTimeout(timer);
  }, [active, selectedRunId, playbackState, loading, frame, frameCount]);

  const contextRows = [
    ['Run ID', selectedRun?.run_id || 'none'],
    ['Config', shortHash(selectedRun?.config_hash)],
    ['Seed', selectedRun?.seed ?? 'unknown'],
    ['Completed', formatRunCompletedAt(selectedRun)],
    ['Grid', selectedRun?.grid_shape?.join(' x ') || 'unknown'],
    ['Frame', `${frame + 1} / ${frameCount}`],
    ['Source', selectedRun ? formatBytes(selectedRun.artifact_path) : 'none'],
    ['Reduction', manifest ? `stride ${manifest.downsampling.stride}, max ${manifest.downsampling.max_resolution}` : 'not rendered'],
    ['Dataset', manifest?.source_dataset || 'not rendered'],
    ['Validation', selectedRun?.validation_status || 'unknown'],
  ];

  return (
    <div className="grid grid-cols-1 2xl:grid-cols-[320px_minmax(0,1fr)_360px] gap-4 min-h-[calc(100vh-7rem)]">
      <aside className="bg-slate-900 border border-slate-800 rounded p-3 min-h-[620px]">
        <div className="flex items-center justify-between mb-3">
          <div className="text-sm font-bold text-white flex items-center gap-2"><Layers size={15} /> Runs</div>
          <button
            type="button"
            title="Refresh runs"
            onClick={() => void refreshRuns()}
            className="h-8 w-8 grid place-items-center rounded bg-slate-800 hover:bg-slate-700 text-slate-300"
          >
            <RefreshCw size={14} className={loading ? 'animate-spin' : ''} />
          </button>
        </div>
        <label className="mb-3 flex items-center gap-2 text-[11px] text-slate-400">
          <input
            type="checkbox"
            checked={includeArtifacts}
            onChange={(event) => setIncludeArtifacts(event.target.checked)}
            className="accent-cyan-500"
          />
          Include loose NPZ/static artifacts
        </label>
        <div className="space-y-2 max-h-[420px] overflow-y-auto pr-1">
          {runs.map((run) => {
            const selected = run.run_id === selectedRunId;
            const hue = hashHue(run.config_hash || run.run_id);
            return (
              <button
                type="button"
                key={run.run_id}
                onClick={() => setSelectedRunId(run.run_id)}
                style={{ borderLeftColor: `hsl(${hue} 78% 55%)`, borderLeftWidth: 4 }}
                className={`w-full text-left border rounded p-3 ${selected ? 'bg-cyan-950/30 border-cyan-700 text-cyan-100' : 'bg-slate-950 border-slate-800 text-slate-300 hover:border-slate-600'}`}
              >
                <div className="flex items-start justify-between gap-2">
                  <div className="min-w-0">
                    <div className="text-xs font-bold truncate">{run.run_id}</div>
                    <div className="text-[11px] text-cyan-200/80 mt-1">{formatRunCompletedAt(run)}</div>
                  </div>
                  <span className="shrink-0 rounded border border-slate-700 px-1.5 py-0.5 text-[10px] text-slate-300">
                    {run.frame_count || 1}f
                  </span>
                </div>
                <div className="mt-2 flex flex-wrap gap-1.5">
                  <span className="rounded bg-slate-800 px-1.5 py-0.5 text-[10px] font-mono text-slate-300">{shortHash(run.config_hash)}</span>
                  <span className="rounded bg-slate-800 px-1.5 py-0.5 text-[10px] text-slate-300">seed {run.seed ?? 'n/a'}</span>
                  <span className="rounded bg-slate-800 px-1.5 py-0.5 text-[10px] text-slate-300">{run.grid_shape?.join('x') || 'grid ?'}</span>
                </div>
                <div className="text-[11px] text-slate-500 mt-2">{run.status || 'COMPLETE'} / {run.artifact_kind} / {run.validation_status || 'UNKNOWN'}</div>
              </button>
            );
          })}
          {runs.length === 0 ? <div className="text-xs text-slate-500 border border-slate-800 rounded p-3">No authoritative completed HDF5 runs available. Enable loose artifacts for diagnostic NPZ/static outputs.</div> : null}
        </div>

        <div className="mt-4 pt-4 border-t border-slate-800">
          <div className="text-xs text-slate-400 mb-2">Config-hash rerun</div>
          <div className="flex gap-2">
            <input
              value={configHash}
              onChange={(event) => setConfigHash(event.target.value)}
              className="min-w-0 flex-1 bg-slate-950 border border-slate-700 rounded px-2 py-2 text-xs text-white font-mono"
              placeholder="config hash"
            />
            <button
              type="button"
              title="Request exact rerun"
              onClick={() => void submitRerun()}
              className="h-9 w-9 grid place-items-center rounded bg-slate-800 hover:bg-slate-700 text-slate-300"
            >
              <RotateCcw size={14} />
            </button>
          </div>
        </div>
      </aside>

      <section className="min-w-0 space-y-3">
        <div className="bg-slate-900 border border-slate-800 rounded p-3">
          <div className="grid grid-cols-1 xl:grid-cols-[220px_1fr_260px_240px] gap-3 items-end">
            <div>
              <label className="block text-[11px] text-slate-400 mb-1">Tile</label>
              <select
                value={tile}
                onChange={(event) => setTile(event.target.value as TileId)}
                className="w-full bg-slate-950 border border-slate-700 rounded p-2 text-sm text-white"
              >
                {TILE_OPTIONS.map((option) => <option key={option.id} value={option.id}>{option.label}</option>)}
              </select>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              <label className="text-[11px] text-slate-400">
                Frame
                <input
                  type="range"
                  min={0}
                  max={Math.max(0, frameCount - 1)}
                  value={frame}
                  disabled={!frameUnlocked}
                  onChange={(event) => {
                    const next = Number(event.target.value);
                    setFrame(next);
                    void renderFrame(next, 'manual');
                  }}
                  className="w-full mt-2 disabled:opacity-40"
                />
                <span className="block mt-1 text-[10px] text-slate-500">
                  {frameUnlocked ? 'Frame selection enabled' : 'Locked while video playback is active'}
                </span>
              </label>
              <label className="text-[11px] text-slate-400">
                Threshold {threshold.toFixed(2)}
                <input
                  type="range"
                  min={0.05}
                  max={0.95}
                  step={0.01}
                  value={threshold}
                  disabled={tile !== 'density_isosurface'}
                  onChange={(event) => setThreshold(Number(event.target.value))}
                  className="w-full mt-2 disabled:opacity-40"
                />
              </label>
            </div>
            <div className="grid grid-cols-2 gap-3">
              <label className="text-[11px] text-slate-400">
                Resolution
                <select
                  value={resolution}
                  onChange={(event) => setResolution(Number(event.target.value))}
                  className="mt-1 w-full bg-slate-950 border border-slate-700 rounded p-2 text-sm text-white"
                >
                  {[32, 48, 64, 96].map((value) => <option key={value} value={value}>{value}^3 cap</option>)}
                </select>
              </label>
              <label className="text-[11px] text-slate-400">
                Normalization
                <select
                  value={normalization}
                  onChange={(event) => setNormalization(event.target.value as 'linear' | 'none')}
                  className="mt-1 w-full bg-slate-950 border border-slate-700 rounded p-2 text-sm text-white"
                >
                  <option value="linear">Linear</option>
                  <option value="none">None</option>
                </select>
              </label>
            </div>
            <div className="grid grid-cols-2 gap-2">
              <button
                type="button"
                onClick={() => void startPlayback()}
                disabled={loading || !selectedRun}
                className="inline-flex items-center justify-center gap-2 rounded bg-cyan-600 hover:bg-cyan-500 disabled:bg-slate-700 disabled:text-slate-400 text-white font-bold px-3 py-2 text-sm"
              >
                <Play size={15} /> Render / Play
              </button>
              {playbackState === 'playing' ? (
                <button
                  type="button"
                  onClick={pausePlayback}
                  className="inline-flex items-center justify-center gap-2 rounded bg-slate-700 hover:bg-slate-600 text-white font-bold px-3 py-2 text-sm"
                >
                  <Pause size={15} /> Pause
                </button>
              ) : (
                <button
                  type="button"
                  onClick={resumePlayback}
                  disabled={!selectedRun || playbackState === 'idle'}
                  className="inline-flex items-center justify-center gap-2 rounded bg-slate-700 hover:bg-slate-600 disabled:bg-slate-800 disabled:text-slate-500 text-white font-bold px-3 py-2 text-sm"
                >
                  <Play size={15} /> Resume
                </button>
              )}
            </div>
          </div>
          <div className="mt-2 flex flex-wrap items-center gap-3 text-xs text-slate-400">
            <span>{status}</span>
            <span className="font-mono text-slate-500">Playback: {playbackState.toUpperCase()}</span>
          </div>
        </div>

        <div className="grid grid-cols-1 xl:grid-cols-[180px_minmax(0,1fr)] gap-3">
          <ViewLegend tile={tile} onTileChange={setTile} manifest={manifest} />
          {manifest?.tile === 'orthogonal_slices' ? (
            <div className="min-h-[520px]"><SliceCanvases manifest={manifest} /></div>
          ) : manifest?.tile === 'validation_dashboard' ? (
            <ValidationDashboard metrics={metrics} />
          ) : (
            <VTKViewport manifest={manifest} />
          )}
        </div>
      </section>

      <aside className="bg-slate-900 border border-slate-800 rounded p-3 min-h-[620px]">
        <div className="text-sm font-bold text-white flex items-center gap-2 mb-3"><Activity size={15} /> Context</div>
        <div className="space-y-2">
          {contextRows.map(([label, value]) => (
            <div key={String(label)} className="bg-slate-950 border border-slate-800 rounded p-2">
              <div className="text-[10px] uppercase tracking-wide text-slate-500">{label}</div>
              <div className="text-xs text-slate-200 font-mono break-all mt-1">{String(value)}</div>
            </div>
          ))}
        </div>
        {selectedRun?.config_hash ? (
          <button
            type="button"
            title="Copy config hash"
            onClick={() => navigator.clipboard?.writeText(selectedRun.config_hash || '')}
            className="mt-3 inline-flex items-center gap-2 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 px-3 py-2 text-xs"
          >
            <Copy size={13} /> Copy config hash
          </button>
        ) : null}
        <div className="mt-4 border-t border-slate-800 pt-3">
          <div className="text-xs font-bold text-slate-300 mb-2">Telemetry</div>
          <div className="h-52 bg-slate-950 border border-slate-800 rounded">
            <ValidationDashboard metrics={metrics} compact />
          </div>
        </div>
        <div className="mt-3 text-[11px] text-amber-200 bg-amber-950/20 border border-amber-900/40 rounded p-2">
          {manifest?.labels.raw_or_derived || 'No rendered visualization yet.'}
        </div>
      </aside>
    </div>
  );
}
