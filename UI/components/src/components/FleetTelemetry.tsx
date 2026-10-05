import React, { useEffect, useMemo, useState } from 'react';
import { RefreshCw, Server } from 'lucide-react';
import {
  FleetTelemetryResponse,
  FleetWorkerHealth,
  getSystemTelemetry,
} from '../api_client';

type FleetTelemetryProps = {
  active: boolean;
};

const HEALTHY_DOT = 'h-2.5 w-2.5 rounded-full bg-emerald-500';
const STALE_DOT = 'h-2.5 w-2.5 rounded-full bg-rose-500';
const HISTORICAL_DOT = 'h-2.5 w-2.5 rounded-full bg-slate-500';

const formatHeartbeat = (epoch: number) => {
  if (!Number.isFinite(epoch) || epoch <= 0) {
    return 'n/a';
  }
  return new Date(epoch * 1000).toLocaleString();
};

export default function FleetTelemetry({ active }: FleetTelemetryProps) {
  const [payload, setPayload] = useState<FleetTelemetryResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showHistorical, setShowHistorical] = useState(false);

  const workers: FleetWorkerHealth[] = payload?.workers || [];

  const summary = useMemo(() => {
    return {
      queueDepth: payload?.queue_depth ?? 0,
      claimsProcessed: payload?.total_claims_processed ?? 0,
      activeCount: payload?.active_workers?.length ?? 0,
      staleCount: payload?.stale_workers?.length ?? 0,
      historicalCount: payload?.historical_workers?.length ?? 0,
      ttlSeconds: payload?.worker_heartbeat_ttl_seconds ?? 0,
      updatedAt: payload?.timestamp ?? 'n/a',
    };
  }, [payload]);

  const refresh = async () => {
    setLoading(true);
    try {
      const response = await getSystemTelemetry(undefined, showHistorical);
      if (response.status !== 'success') {
        throw new Error(response.message || 'Failed to load telemetry');
      }
      setPayload(response);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load telemetry');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (!active) {
      return;
    }
    let mounted = true;
    const guardedRefresh = async () => {
      if (!mounted) {
        return;
      }
      await refresh();
    };

    void guardedRefresh();
    const timer = window.setInterval(() => {
      void guardedRefresh();
    }, 4000);

    return () => {
      mounted = false;
      window.clearInterval(timer);
    };
  }, [active, showHistorical]);

  const dotClass = (state: FleetWorkerHealth['state']) => {
    if (state === 'active') {
      return HEALTHY_DOT;
    }
    if (state === 'historical') {
      return HISTORICAL_DOT;
    }
    return STALE_DOT;
  };

  const labelClass = (state: FleetWorkerHealth['state']) => {
    if (state === 'active') {
      return 'text-emerald-300';
    }
    if (state === 'historical') {
      return 'text-slate-400';
    }
    return 'text-rose-300';
  };

  return (
    <div className="space-y-6 h-full">
      <div className="bg-slate-900 border border-slate-800 rounded-lg p-4">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-white font-bold flex items-center gap-2"><Server size={16} className="text-cyan-400"/> Fleet Telemetry</h3>
          <button
            title="Refresh Fleet Telemetry"
            onClick={() => void refresh()}
            className="p-1 hover:bg-slate-800 rounded text-slate-400"
          >
            <RefreshCw size={14} />
          </button>
        </div>
        {error && <div className="text-sm text-rose-300 mb-3">{error}</div>}
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4">
          <div className="rounded border border-slate-700 bg-slate-950/70 p-3">
            <div className="text-xs text-slate-400 uppercase tracking-wide">Queue Depth</div>
            <div className="text-2xl text-slate-100 font-mono mt-1">{summary.queueDepth}</div>
          </div>
          <div className="rounded border border-slate-700 bg-slate-950/70 p-3">
            <div className="text-xs text-slate-400 uppercase tracking-wide">Claims Processed</div>
            <div className="text-2xl text-slate-100 font-mono mt-1">{summary.claimsProcessed}</div>
          </div>
          <div className="rounded border border-emerald-700/60 bg-emerald-950/20 p-3">
            <div className="text-xs text-emerald-300 uppercase tracking-wide">Active Workers</div>
            <div className="text-2xl text-emerald-200 font-mono mt-1">{summary.activeCount}</div>
          </div>
          <div className="rounded border border-rose-700/60 bg-rose-950/20 p-3">
            <div className="text-xs text-rose-300 uppercase tracking-wide">Stale Workers</div>
            <div className="text-2xl text-rose-200 font-mono mt-1">{summary.staleCount}</div>
          </div>
        </div>
        <div className="mt-3 flex flex-wrap items-center gap-3 text-xs text-slate-500 font-mono">
          <span>TTL: {summary.ttlSeconds || 'n/a'}s | Updated: {summary.updatedAt}</span>
          <label className="inline-flex items-center gap-2 text-slate-400">
            <input
              type="checkbox"
              checked={showHistorical}
              onChange={(event) => setShowHistorical(event.target.checked)}
              className="accent-cyan-500"
            />
            Historical/offline ({summary.historicalCount})
          </label>
        </div>
      </div>

      <div className="h-full flex flex-col bg-slate-900 border border-slate-800 rounded-lg overflow-hidden">
        <div className="p-4 border-b border-slate-800 bg-slate-900/80 backdrop-blur">
          <h4 className="text-white font-bold">Worker Health</h4>
        </div>
        <div className="flex-1 overflow-y-auto p-2">
          {loading && workers.length === 0 ? (
            <div className="p-3 text-sm text-slate-400">Loading fleet telemetry...</div>
          ) : workers.length === 0 ? (
            <div className="p-3 text-sm text-slate-400">No worker heartbeats detected yet.</div>
          ) : (
            <table className="w-full text-left text-xs text-slate-400">
              <thead className="text-slate-500 font-mono uppercase bg-slate-950/50 sticky top-0">
                <tr>
                  <th className="p-3">Worker</th>
                  <th className="p-3">Health</th>
                  <th className="p-3">Age (s)</th>
                  <th className="p-3">In Flight</th>
                  <th className="p-3">Last Heartbeat</th>
                  <th className="p-3">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800">
                {workers.map((worker) => (
                  <tr key={worker.worker_id} className="hover:bg-slate-800/50 transition-colors">
                    <td className="p-3 font-mono text-slate-300">{worker.worker_id}</td>
                    <td className="p-3">
                      <span className="inline-flex items-center gap-2">
                        <span className={dotClass(worker.state)} />
                        <span className={labelClass(worker.state)}>
                          {worker.state.toUpperCase()}
                        </span>
                      </span>
                    </td>
                    <td className="p-3 font-mono">{worker.age_seconds.toFixed(1)}</td>
                    <td className="p-3 font-mono">{worker.in_flight_claims}</td>
                    <td className="p-3 text-slate-300">{formatHeartbeat(worker.last_heartbeat_epoch)}</td>
                    <td className="p-3">
                      <button
                        className="px-2 py-1 text-xs rounded bg-rose-700 hover:bg-rose-800 text-white font-bold"
                        title="Kill Worker"
                        onClick={async () => {
                          await fetch(`/api/control/kill_worker/${worker.worker_id}`, { method: 'POST' });
                          await refresh();
                        }}
                      >Kill</button>
                    </td>
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
