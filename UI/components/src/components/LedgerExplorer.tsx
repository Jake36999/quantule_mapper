import React, { useEffect, useState } from 'react';
import { Database, RefreshCw } from 'lucide-react';
import {
  getLedgerRows,
  getLedgerRuns,
  LedgerRow,
  LedgerRunSummary,
} from '../api_client';

type LedgerExplorerProps = {
  activeRunId: string | null;
  activeSessionDir: string | null;
  defaultPageSize?: number;
};

const PAGE_SIZE_OPTIONS = [25, 50, 100, 250];

const formatNumber = (value: number | null | undefined) => {
  if (value == null || Number.isNaN(value)) {
    return 'n/a';
  }
  return Number(value).toFixed(4);
};

const formatTimestamp = (value: string | null | undefined) => {
  if (!value) {
    return 'n/a';
  }
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? value : parsed.toLocaleString();
};

export default function LedgerExplorer({
  activeRunId,
  activeSessionDir,
  defaultPageSize = 50,
}: LedgerExplorerProps) {
  const [runOptions, setRunOptions] = useState<LedgerRunSummary[]>([]);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(activeRunId);
  const [rows, setRows] = useState<LedgerRow[]>([]);
  const [pageSize, setPageSize] = useState<number>(defaultPageSize);
  const [offset, setOffset] = useState<number>(0);
  const [totalRows, setTotalRows] = useState<number>(0);
  const [isLoadingRuns, setIsLoadingRuns] = useState(false);
  const [isLoadingRows, setIsLoadingRows] = useState(false);
  const [runErrorMessage, setRunErrorMessage] = useState<string | null>(null);
  const [rowsErrorMessage, setRowsErrorMessage] = useState<string | null>(null);

  const selectedRun = runOptions.find(run => run.run_id === selectedRunId) || null;
  const currentPage = Math.floor(offset / pageSize) + 1;
  const totalPages = Math.max(1, Math.ceil(totalRows / pageSize) || 1);
  const rangeStart = totalRows === 0 ? 0 : offset + 1;
  const rangeEnd = Math.min(offset + rows.length, totalRows);

  const refreshRuns = async () => {
    setIsLoadingRuns(true);
    setRunErrorMessage(null);
    try {
      const response = await getLedgerRuns();
      if (response.status !== 'success') {
        throw new Error(response.message || 'Failed to load ledger runs.');
      }
      const options = response.runs || [];
      setRunOptions(options);
      setSelectedRunId(prev => {
        if (prev && options.some(run => run.run_id === prev)) {
          return prev;
        }
        if (activeRunId && options.some(run => run.run_id === activeRunId)) {
          return activeRunId;
        }
        return options[0]?.run_id || null;
      });
    } catch (err) {
      setRunErrorMessage(err instanceof Error ? err.message : 'Failed to load ledger runs.');
      setRunOptions([]);
      setSelectedRunId(null);
    } finally {
      setIsLoadingRuns(false);
    }
  };

  useEffect(() => {
    void refreshRuns();
  }, [activeRunId]);

  useEffect(() => {
    setOffset(0);
  }, [selectedRunId, pageSize]);

  useEffect(() => {
    if (!selectedRunId) {
      setRows([]);
      setTotalRows(0);
      return;
    }

    let mounted = true;
    const refreshRows = async () => {
      setIsLoadingRows(true);
      setRowsErrorMessage(null);
      try {
        const response = await getLedgerRows(selectedRunId, pageSize, offset);
        if (!mounted) {
          return;
        }
        if (response.status !== 'success') {
          throw new Error(response.message || 'Failed to load ledger rows.');
        }
        setRows(response.rows || []);
        setTotalRows(response.pagination?.total || 0);
      } catch (err) {
        if (mounted) {
          setRows([]);
          setTotalRows(0);
          setRowsErrorMessage(err instanceof Error ? err.message : 'Failed to load ledger rows.');
        }
      } finally {
        if (mounted) {
          setIsLoadingRows(false);
        }
      }
    };

    void refreshRows();
    return () => {
      mounted = false;
    };
  }, [selectedRunId, pageSize, offset]);

  return (
    <div className="h-full flex flex-col gap-6">
      <div className="bg-slate-900 border border-slate-800 rounded-lg p-4">
        <div className="flex items-center justify-between gap-4 mb-4">
          <h3 className="text-white font-bold flex items-center gap-2"><Database size={16} className="text-cyan-400"/> Ledger Explorer</h3>
          <button
            title="Refresh ledger runs"
            onClick={() => void refreshRuns()}
            className="px-3 py-2 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 flex items-center gap-2 text-sm"
          >
            <RefreshCw size={14} /> Refresh
          </button>
        </div>

        <div className="grid grid-cols-1 xl:grid-cols-3 gap-4">
          <div className="xl:col-span-2">
            <label htmlFor="ledger-run-select" className="text-xs text-slate-400 block mb-1">HUNT / LEDGER</label>
            <select
              id="ledger-run-select"
              title="Ledger run selector"
              value={selectedRunId || ''}
              onChange={(e) => setSelectedRunId(e.target.value || null)}
              disabled={isLoadingRuns || runOptions.length === 0}
              className="w-full bg-slate-950 border border-slate-700 rounded p-2 text-white text-sm font-mono"
            >
              {runOptions.length === 0 ? (
                <option value="" className="text-slate-900 bg-slate-100">No runs discovered</option>
              ) : (
                runOptions.map((run) => (
                  <option key={run.run_id} value={run.run_id} className="text-slate-900 bg-slate-100">
                    {run.is_active ? '[ACTIVE] ' : ''}{run.hunt_name}_{run.run_id}
                  </option>
                ))
              )}
            </select>
            <div className={`mt-2 text-[11px] ${runErrorMessage ? 'text-rose-300' : runOptions.length === 0 ? 'text-amber-300' : 'text-slate-500'}`}>
              {isLoadingRuns
                ? 'Loading discovered runs...'
                : runErrorMessage
                  ? `Run list unavailable: ${runErrorMessage}`
                  : runOptions.length === 0
                    ? 'No runs discovered. Start a run and click Refresh.'
                    : `Discovered ${runOptions.length} run${runOptions.length === 1 ? '' : 's'}.`}
            </div>
          </div>
          <div>
            <label htmlFor="ledger-page-size" className="text-xs text-slate-400 block mb-1">ROWS PER PAGE</label>
            <select
              id="ledger-page-size"
              title="Ledger page size"
              value={pageSize}
              onChange={(e) => setPageSize(Number(e.target.value))}
              className="w-full bg-slate-950 border border-slate-700 rounded p-2 text-white text-sm font-mono"
            >
              {PAGE_SIZE_OPTIONS.map((size) => (
                <option key={size} value={size} className="text-slate-900 bg-slate-100">{size}</option>
              ))}
            </select>
          </div>
        </div>

        <div className="mt-3 text-xs text-slate-400 space-y-1">
          <div>Active Session: <span className="text-slate-300 font-mono break-all">{activeSessionDir || 'none'}</span></div>
          <div>Selected DB: <span className="text-slate-300 font-mono break-all">{selectedRun?.db_file || 'n/a'}</span></div>
        </div>
      </div>

      <div className="h-full flex flex-col bg-slate-900 border border-slate-800 rounded-lg overflow-hidden">
        <div className="p-4 border-b border-slate-800 flex items-center justify-between bg-slate-900/80 backdrop-blur">
          <div>
            <h4 className="text-white font-bold">Ledger Rows</h4>
            <div className="text-xs text-slate-400 mt-1">
              Showing <span className="text-slate-200 font-mono">{rangeStart}-{rangeEnd}</span> of <span className="text-slate-200 font-mono">{totalRows}</span>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setOffset(Math.max(0, offset - pageSize))}
              disabled={offset === 0 || isLoadingRows}
              className="px-3 py-2 rounded bg-slate-800 hover:bg-slate-700 disabled:bg-slate-800/40 disabled:text-slate-600 text-slate-300 text-sm"
            >
              Previous
            </button>
            <button
              onClick={() => setOffset(offset + pageSize)}
              disabled={isLoadingRows || offset + pageSize >= totalRows}
              className="px-3 py-2 rounded bg-slate-800 hover:bg-slate-700 disabled:bg-slate-800/40 disabled:text-slate-600 text-slate-300 text-sm"
            >
              Next
            </button>
          </div>
        </div>

        <div className="text-xs text-slate-500 px-4 py-2 border-b border-slate-800 bg-slate-950/40">
          Page <span className="font-mono text-slate-300">{currentPage}</span> of <span className="font-mono text-slate-300">{totalPages}</span>
        </div>

        <div className="flex-1 overflow-auto">
          {rowsErrorMessage ? (
            <div className="p-6 text-sm text-rose-300">{rowsErrorMessage}</div>
          ) : !selectedRunId ? (
            <div className="p-6 text-sm text-slate-400">Select a run to inspect its ledger.</div>
          ) : isLoadingRows ? (
            <div className="p-6 text-sm text-amber-300">Loading ledger rows...</div>
          ) : rows.length === 0 ? (
            <div className="p-6 text-sm text-slate-400">No ledger rows found for the selected run.</div>
          ) : (
            <table className="w-full text-left text-xs text-slate-400">
              <thead className="text-slate-500 font-mono uppercase bg-slate-950/50 sticky top-0">
                <tr>
                  <th className="p-3">Generation</th>
                  <th className="p-3">Timestamp</th>
                  <th className="p-3">Status</th>
                  <th className="p-3">Fitness</th>
                  <th className="p-3">Log SSE</th>
                  <th className="p-3">PCS</th>
                  <th className="p-3">Bragg</th>
                  <th className="p-3">D</th>
                  <th className="p-3">Eta</th>
                  <th className="p-3">Rho Vac</th>
                  <th className="p-3">Config Hash</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800">
                {rows.map((row) => (
                  <tr key={`${row.config_hash}-${row.generation ?? 'na'}`} className="hover:bg-slate-800/50 transition-colors">
                    <td className="p-3 font-mono text-slate-300">{row.generation ?? 'n/a'}</td>
                    <td className="p-3 text-slate-300">{formatTimestamp(row.timestamp)}</td>
                    <td className="p-3 font-mono text-slate-300">{row.status || 'n/a'}</td>
                    <td className="p-3 font-mono">{formatNumber(row.fitness)}</td>
                    <td className="p-3 font-mono">{formatNumber(row.log_prime_sse)}</td>
                    <td className="p-3 font-mono">{formatNumber(row.pcs)}</td>
                    <td className="p-3 font-mono">{row.bragg_peaks_detected ?? 'n/a'}</td>
                    <td className="p-3 font-mono">{formatNumber(row.param_D)}</td>
                    <td className="p-3 font-mono">{formatNumber(row.param_eta)}</td>
                    <td className="p-3 font-mono">{formatNumber(row.param_rho_vac)}</td>
                    <td className="p-3 font-mono text-slate-300">{row.config_hash}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>
    </div>
  );
}