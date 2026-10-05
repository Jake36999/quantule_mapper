import React, { useEffect, useMemo, useState } from 'react';
import {
  executeVisualizer,
  getVisualizerPlugins,
  searchArtifacts,
  ArtifactSearchResult,
  VisualizerPlugin,
} from '../api_client';
import { WebGLViewer } from './WebGLViewer';

type LogEntry = {
  ts: string;
  message: string;
  level: 'INFO' | 'ERROR';
};

type DropdownState = 'loading' | 'empty' | 'error' | 'ready';

const formatBackendError = (message: string): string => {
  const codeMatch = String(message || '').match(/\b(4\d\d|5\d\d)\b/);
  const code = codeMatch ? codeMatch[1] : '503';
  return `Backend unreachable [${code}]`;
};

const TopologicalExplorer = () => {
  const [query, setQuery] = useState('rho_history');
  const [artifacts, setArtifacts] = useState<ArtifactSearchResult[]>([]);
  const [plugins, setPlugins] = useState<VisualizerPlugin[]>([]);
  const [selectedArtifact, setSelectedArtifact] = useState<string>('');
  const [selectedPlugin, setSelectedPlugin] = useState<string>('');
  const [logLines, setLogLines] = useState<LogEntry[]>([]);
  const [isLoadingArtifacts, setIsLoadingArtifacts] = useState(false);
  const [isLoadingPlugins, setIsLoadingPlugins] = useState(false);
  const [isExecuting, setIsExecuting] = useState(false);
  const [viewerMessage, setViewerMessage] = useState('Execution output will render here (GIF/image/JSON) in the next pass.');
  const [selectedDataset, setSelectedDataset] = useState('psi_final');
  const [pluginDropdownState, setPluginDropdownState] = useState<DropdownState>('loading');
  const [pluginDropdownMessage, setPluginDropdownMessage] = useState('Fetching...');

  const appendLog = (message: string, level: 'INFO' | 'ERROR' = 'INFO') => {
    setLogLines((prev) => [
      ...prev,
      { ts: new Date().toISOString(), message, level },
    ].slice(-200));
  };

  const h5Artifacts = useMemo(
    () => artifacts.filter((artifact) => String(artifact.type).toLowerCase() === '.h5'),
    [artifacts]
  );

  const selectedArtifactFilename = useMemo(() => {
    if (!selectedArtifact) {
      return '';
    }
    const normalized = selectedArtifact.replace(/\\/g, '/');
    const parts = normalized.split('/').filter(Boolean);
    return parts.length > 0 ? parts[parts.length - 1] : '';
  }, [selectedArtifact]);

  const selectedArtifactConfigHash = useMemo(() => {
    const match = selectedArtifactFilename.match(/^rho_history_([A-Fa-f0-9]{64})\.h5$/);
    return match ? match[1] : '';
  }, [selectedArtifactFilename]);

  const loadPlugins = async () => {
    setIsLoadingPlugins(true);
    setPluginDropdownState('loading');
    setPluginDropdownMessage('Fetching...');
    try {
      const response = await getVisualizerPlugins();
      if (response.status !== 'success') {
        appendLog(response.message || 'Visualizer plugin API returned an error.', 'ERROR');
        setPlugins([]);
        setSelectedPlugin('');
        setPluginDropdownState('error');
        setPluginDropdownMessage(formatBackendError(response.message || '503'));
        return;
      }
      const pluginList = response.plugins || response.manifest?.plugins || [];
      setPlugins(pluginList);
      if (pluginList.length > 0 && !pluginList.some((plugin) => plugin.id === selectedPlugin)) {
        setSelectedPlugin(pluginList[0].id);
      }
      if (pluginList.length === 0) {
        setPluginDropdownState('empty');
        setPluginDropdownMessage('No artifacts found.');
      } else {
        setPluginDropdownState('ready');
        setPluginDropdownMessage(`Loaded ${pluginList.length} plugin${pluginList.length === 1 ? '' : 's'}.`);
      }
      appendLog(`Loaded ${pluginList.length} visualizer plugins.`);
    } catch (err) {
      appendLog(`Failed to load visualizer plugins: ${err instanceof Error ? err.message : 'unknown error'}`, 'ERROR');
      setPluginDropdownState('error');
      setPluginDropdownMessage(formatBackendError(err instanceof Error ? err.message : '503'));
    } finally {
      setIsLoadingPlugins(false);
    }
  };

  const runSearch = async (q: string) => {
    setIsLoadingArtifacts(true);
    try {
      const response = await searchArtifacts(q);
      if (response.status !== 'success') {
        appendLog(response.message || 'Artifact search API returned an error.', 'ERROR');
        setArtifacts([]);
        setSelectedArtifact('');
        return;
      }
      const hits = response.results || response.artifacts || [];
      setArtifacts(hits);
      const firstH5 = hits.find((artifact) => String(artifact.type).toLowerCase() === '.h5');
      setSelectedArtifact((prev) => {
        if (prev && hits.some((artifact) => artifact.relative_path === prev)) {
          return prev;
        }
        return firstH5?.relative_path || '';
      });
      appendLog(`Artifact search returned ${hits.length} total hit(s), ${hits.filter((a) => String(a.type).toLowerCase() === '.h5').length} .h5 file(s).`);
    } catch (err) {
      appendLog(`Artifact search failed: ${err instanceof Error ? err.message : 'unknown error'}`, 'ERROR');
    } finally {
      setIsLoadingArtifacts(false);
    }
  };

  const handleExecute = async () => {
    if (!selectedPlugin || !selectedArtifact) {
      appendLog('Select a visualizer plugin and an .h5 artifact before executing.', 'ERROR');
      return;
    }

    setIsExecuting(true);
    appendLog(`Submitting mock execution for plugin '${selectedPlugin}' on artifact '${selectedArtifact}'.`);

    try {
      const response = await executeVisualizer(selectedPlugin, selectedArtifact);
      if (response.status === 'success') {
        setViewerMessage(
          `Execution queued (mock): ${response.execution_id || 'n/a'}\n` +
          `Plugin: ${response.plugin_id}\nArtifact: ${response.artifact_path}`
        );
        appendLog(response.message || 'Mock execution accepted by backend.');
      } else {
        appendLog(response.message || 'Execution failed.', 'ERROR');
      }
    } catch (err) {
      appendLog(`Execute request failed: ${err instanceof Error ? err.message : 'unknown error'}`, 'ERROR');
    } finally {
      setIsExecuting(false);
    }
  };

  useEffect(() => {
    void loadPlugins();
    void runSearch(query);
  }, []);

  return (
    <div className="grid grid-cols-1 xl:grid-cols-12 gap-6 h-full min-h-[640px]">
      <section className="xl:col-span-4 bg-slate-900 border border-slate-800 rounded-lg p-4 flex flex-col">
        <h3 className="text-white font-bold mb-3">Topological Exploration Controls</h3>

        <label htmlFor="artifact-search" className="text-xs text-slate-400 mb-1">SEARCH ARTIFACTS</label>
        <div className="flex gap-2 mb-3">
          <input
            id="artifact-search"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="rho_history"
            className="flex-1 bg-slate-950 border border-slate-700 rounded p-2 text-white text-sm font-mono"
          />
          <button
            onClick={() => void runSearch(query)}
            disabled={isLoadingArtifacts}
            className="px-3 py-2 rounded bg-blue-600 hover:bg-blue-500 disabled:bg-slate-700 text-white text-sm font-bold"
          >
            {isLoadingArtifacts ? 'SEARCH...' : 'SEARCH'}
          </button>
        </div>

        <div className="mb-4 border border-slate-800 rounded bg-slate-950/70 min-h-[220px] max-h-[260px] overflow-y-auto">
          <div className="text-[11px] uppercase tracking-wide text-slate-500 px-3 py-2 border-b border-slate-800">Artifacts (.h5 selectable)</div>
          {artifacts.length === 0 && (
            <div className="px-3 py-4 text-xs text-slate-500">No artifacts found.</div>
          )}
          {artifacts.map((artifact) => {
            const isH5 = String(artifact.type).toLowerCase() === '.h5';
            const isActive = selectedArtifact === artifact.relative_path;
            return (
              <button
                key={`${artifact.source}:${artifact.relative_path}`}
                disabled={!isH5}
                onClick={() => setSelectedArtifact(artifact.relative_path)}
                className={`w-full text-left px-3 py-2 border-b border-slate-900 text-xs transition-colors ${
                  isActive ? 'bg-cyan-600/20 text-cyan-300' : 'text-slate-300 hover:bg-slate-800/70'
                } ${!isH5 ? 'opacity-45 cursor-not-allowed' : ''}`}
              >
                <div className="font-mono truncate">{artifact.relative_path}</div>
                <div className="text-[10px] text-slate-500">{artifact.source} · {artifact.type}</div>
              </button>
            );
          })}
        </div>

        <label htmlFor="visualizer-plugin" className="text-xs text-slate-400 mb-1">SELECT VISUALIZER SCRIPT</label>
        <select
          id="visualizer-plugin"
          title="Visualizer plugin selection"
          value={selectedPlugin}
          onChange={(e) => setSelectedPlugin(e.target.value)}
          className="mb-4 bg-slate-950 border border-slate-700 rounded p-2 text-white text-sm"
          disabled={pluginDropdownState !== 'ready'}
        >
          {plugins.length === 0 ? (
            <option value="" className="text-slate-900 bg-slate-100">
              {pluginDropdownState === 'loading'
                ? 'Fetching...'
                : pluginDropdownState === 'error'
                  ? pluginDropdownMessage
                  : 'No artifacts found.'}
            </option>
          ) : (
            plugins.map((plugin) => (
              <option key={plugin.id} value={plugin.id} className="text-slate-900 bg-slate-100">
                {plugin.name || plugin.id}
              </option>
            ))
          )}
        </select>
        <div className={`mb-4 text-xs rounded border p-2 ${pluginDropdownState === 'loading' ? 'animate-pulse text-cyan-300 border-cyan-900/40 bg-cyan-950/20' : pluginDropdownState === 'error' ? 'text-rose-300 border-rose-900/40 bg-rose-950/20' : pluginDropdownState === 'empty' ? 'text-amber-300 border-amber-900/40 bg-amber-950/20' : 'text-slate-400 border-slate-800 bg-slate-950/40'}`}>
          <div>{pluginDropdownMessage}</div>
          {pluginDropdownState === 'error' && (
            <button
              onClick={() => void loadPlugins()}
              className="mt-2 px-2 py-1 rounded bg-rose-900/40 hover:bg-rose-900/60 text-rose-200 text-[11px]"
            >
              Retry
            </button>
          )}
        </div>

        <button
          onClick={() => void handleExecute()}
          disabled={isExecuting || !selectedPlugin || !selectedArtifact}
          className="w-full bg-emerald-600 hover:bg-emerald-500 disabled:bg-slate-700 disabled:text-slate-400 text-white font-bold py-3 rounded"
        >
          {isExecuting ? 'EXECUTING...' : 'EXECUTE ANALYSIS'}
        </button>
      </section>

      <section className="xl:col-span-8 grid grid-rows-[2fr_1fr] gap-4 min-h-[640px]">
        <div className="bg-slate-900 border border-slate-800 rounded-lg p-4 overflow-hidden flex flex-col">
          <h3 className="text-white font-bold mb-3">Result Viewer</h3>
          <div className="mb-3 flex flex-wrap items-center gap-2">
            <label htmlFor="dataset-select" className="text-xs text-slate-400">DATASET</label>
            <select
              id="dataset-select"
              title="WebGL dataset selection"
              value={selectedDataset}
              onChange={(e) => setSelectedDataset(e.target.value)}
              className="bg-slate-950 border border-slate-700 rounded p-2 text-white text-sm"
            >
              <option value="psi_final">psi_final</option>
              <option value="A_final">A_final</option>
              <option value="N_a_stage">N_a_stage</option>
              <option value="N_b_stage">N_b_stage</option>
              <option value="N_c_stage">N_c_stage</option>
            </select>
            <div className="text-xs text-slate-500 font-mono truncate">
              {selectedArtifactFilename ? `artifact: ${selectedArtifactFilename}` : 'Select an .h5 artifact to enable WebGL viewer'}
            </div>
          </div>

          <div className="flex-1 rounded border border-slate-800 bg-slate-950/70 p-4 text-slate-300 text-sm overflow-auto">
            {selectedArtifactFilename && selectedArtifactConfigHash ? (
              <WebGLViewer configHash={selectedArtifactConfigHash} datasetName={selectedDataset} />
            ) : (
              <div className="h-full flex items-center justify-center text-slate-500 whitespace-pre-wrap">
                {selectedArtifactFilename && !selectedArtifactConfigHash
                  ? 'Selected artifact filename does not expose a canonical config_hash.'
                  : viewerMessage}
              </div>
            )}
          </div>
        </div>

        <div className="bg-slate-900 border border-slate-800 rounded-lg p-4 overflow-hidden flex flex-col">
          <h3 className="text-white font-bold mb-3">Execution Log</h3>
          <div className="flex-1 rounded border border-slate-800 bg-slate-950 p-3 overflow-y-auto font-mono text-xs space-y-1">
            {logLines.length === 0 && <div className="text-slate-500">No execution logs yet.</div>}
            {logLines.map((line, idx) => (
              <div key={`${line.ts}-${idx}`} className={line.level === 'ERROR' ? 'text-rose-300' : 'text-slate-300'}>
                <span className="text-slate-500">[{line.ts}]</span> {line.message}
              </div>
            ))}
          </div>
        </div>
      </section>
    </div>
  );
};

export default TopologicalExplorer;
