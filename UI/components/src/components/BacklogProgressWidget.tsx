import React, { useEffect, useRef, useState } from 'react';
import { getSystemTelemetry } from '../api_client';

type TelemetryData = {
  queue_depth: number;
  total_claims_processed: number;
  dlq_count: number;
  active_workers: string[];
};

type BacklogProgressWidgetProps = {
  active: boolean;
};

export default function BacklogProgressWidget({ active }: BacklogProgressWidgetProps) {
  const [data, setData] = useState<TelemetryData | null>(null);
  const [jpm, setJpm] = useState<number>(0);
  const [error, setError] = useState<string | null>(null);

  const lastProcessed = useRef<number>(-1);
  const lastTime = useRef<number>(Date.now());

  useEffect(() => {
    if (!active) {
      return;
    }

    let mounted = true;

    const fetchTelemetry = async () => {
      try {
        const json = await getSystemTelemetry();
        if (!mounted) {
          return;
        }
        if (json.status !== 'success') {
          throw new Error(json.message || 'Failed to fetch telemetry');
        }

        const payload: TelemetryData = {
          queue_depth: Number(json.queue_depth || 0),
          total_claims_processed: Number(json.total_claims_processed || 0),
          dlq_count: Number(json.dlq_count || 0),
          active_workers: Array.isArray(json.active_workers) ? json.active_workers : [],
        };

        setData(payload);
        setError(null);

        const now = Date.now();
        if (lastProcessed.current === -1) {
          // Prime baseline to avoid a first-sample throughput spike.
          lastProcessed.current = payload.total_claims_processed;
          lastTime.current = now;
          return;
        }

        const timeDiffMinutes = (now - lastTime.current) / 60000;
        if (timeDiffMinutes > 0) {
          const processedDiff = payload.total_claims_processed - lastProcessed.current;
          const currentJpm = Math.max(0, processedDiff / timeDiffMinutes);
          setJpm((prev) => (prev === 0 ? currentJpm : (prev * 0.7) + (currentJpm * 0.3)));
        }
        lastProcessed.current = payload.total_claims_processed;
        lastTime.current = now;
      } catch (err) {
        if (!mounted) {
          return;
        }
        setError(err instanceof Error ? err.message : 'Failed to fetch backlog telemetry');
      }
    };

    void fetchTelemetry();
    const interval = window.setInterval(() => {
      void fetchTelemetry();
    }, 3000);

    return () => {
      mounted = false;
      window.clearInterval(interval);
    };
  }, [active]);

  if (!data) {
    return (
      <div className="bg-slate-900 border border-slate-800 rounded-lg p-4 text-slate-500 text-xs animate-pulse">
        Initializing Logistics...
      </div>
    );
  }

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-lg p-4 flex flex-col gap-4 shadow-xl">
      <h3 className="text-white font-bold text-sm tracking-widest uppercase">The Great Ingestion (Logistics)</h3>
      {error && <div className="text-sm text-rose-300">{error}</div>}

      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4">
        <div className="bg-slate-950 p-3 rounded border border-slate-800 flex flex-col items-center justify-center">
          <span className="text-slate-500 text-[10px] font-mono uppercase">Queue Remaining</span>
          <span className="text-cyan-400 text-2xl font-bold font-mono">{data.queue_depth}</span>
        </div>

        <div className="bg-slate-950 p-3 rounded border border-slate-800 flex flex-col items-center justify-center">
          <span className="text-slate-500 text-[10px] font-mono uppercase">Total Processed</span>
          <span className="text-emerald-400 text-2xl font-bold font-mono">{data.total_claims_processed}</span>
        </div>

        <div className="bg-slate-950 p-3 rounded border border-slate-800 flex flex-col items-center justify-center relative overflow-hidden">
          <span className="text-slate-500 text-[10px] font-mono uppercase relative z-10">Throughput (JPM)</span>
          <span className="text-white text-2xl font-bold font-mono relative z-10">{jpm.toFixed(1)}</span>
          <div className={`absolute bottom-0 left-0 w-full bg-blue-500/20 transition-all duration-1000 ${jpm > 0 ? 'h-full animate-pulse' : 'h-0'}`}></div>
        </div>

        <div className="bg-slate-950 p-3 rounded border border-red-900/50 flex flex-col items-center justify-center">
          <span className="text-red-400/70 text-[10px] font-mono uppercase">Dead-Letter Queue</span>
          <span className="text-red-500 text-2xl font-bold font-mono">{data.dlq_count}</span>
        </div>
      </div>

      <div className="flex justify-between items-center text-[10px] font-mono text-slate-500">
        <span>ACTIVE WORKERS: {data.active_workers.length}</span>
        {jpm > 0 && data.queue_depth > 0 ? (
          <span>EST. TIME REMAINING: {((data.queue_depth / jpm) / 60).toFixed(2)} HOURS</span>
        ) : (
          <span>EST. TIME REMAINING: --</span>
        )}
      </div>
    </div>
  );
}
