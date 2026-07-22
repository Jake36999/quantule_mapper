import React, { useEffect, useMemo, useRef, useState } from 'react';

import { getDebugTerminals } from '../api_client';
import { useTelemetry } from './TelemetryProvider';

type LogFeed = {
  id: string;
  name: string;
  lines: string[];
};

type TerminalMatrixProps = {
  active: boolean;
};

const MAX_LINES = 300;

export default function TerminalMatrix({ active }: TerminalMatrixProps) {
  const [feeds, setFeeds] = useState<LogFeed[]>([]);
  const [activeFeed, setActiveFeed] = useState<string>('combined');
  const [combinedLines, setCombinedLines] = useState<string[]>([]);
  const [heartbeatTick, setHeartbeatTick] = useState<number>(0);
  const scrollRef = useRef<HTMLDivElement>(null);
  const eventCursorRef = useRef(0);
  const { connected, lastEventAt, events } = useTelemetry();

  const appendLine = (feedId: string, line: string) => {
    setCombinedLines((prev) => {
      if (prev[prev.length - 1] === line) {
        return prev;
      }
      return [...prev, line].slice(-MAX_LINES);
    });
    setFeeds((prevFeeds) => {
      const hit = prevFeeds.find((f) => f.id === feedId);
      if (!hit) {
        return [...prevFeeds, { id: feedId, name: feedId.toUpperCase(), lines: [line] }];
      }
      return prevFeeds.map((f) => {
        if (f.id !== feedId) {
          return f;
        }
        const nextLines = f.lines[f.lines.length - 1] === line ? f.lines : [...f.lines, line].slice(-MAX_LINES);
        return { ...f, lines: nextLines };
      });
    });
  };

  useEffect(() => {
    if (!active) {
      return;
    }

    const fetchHistory = async () => {
      try {
        const data = await getDebugTerminals(100);
        if (data.status === 'success') {
          setFeeds(data.feeds || []);
          setCombinedLines(data.combined || []);
        }
      } catch (err) {
        console.error('Failed to load terminal history', err);
      }
    };

    void fetchHistory();
  }, [active]);

  useEffect(() => {
    if (!active) {
      return;
    }

    const timer = window.setInterval(() => {
      setHeartbeatTick((prev) => prev + 1);
    }, 5000);

    return () => window.clearInterval(timer);
  }, [active]);

  useEffect(() => {
    if (!active) {
      eventCursorRef.current = events.length;
      return;
    }
    const nextEvents = events.slice(eventCursorRef.current);
    eventCursorRef.current = events.length;
    nextEvents.forEach((event) => {
      if (event.type === 'terminal_log' && typeof event.line === 'string') {
        appendLine(String(event.feed_id || event.source || 'backend'), event.line);
      }
    });
  }, [active, events]);

  const currentLines = useMemo(() => {
    if (activeFeed === 'combined') {
      return combinedLines;
    }
    return feeds.find((f) => f.id === activeFeed)?.lines || [];
  }, [activeFeed, combinedLines, feeds]);

  const staleEventStream = useMemo(() => {
    if (!active || !connected) {
      return false;
    }
    if (!lastEventAt) {
      return combinedLines.length === 0;
    }
    return Date.now() - lastEventAt > 30000;
  }, [active, combinedLines.length, connected, heartbeatTick, lastEventAt]);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [currentLines]);

  const getColorForLine = (line: string) => {
    if (line.includes('ERROR') || line.includes('CRITICAL') || line.includes('FAIL')) return 'text-red-400';
    if (line.includes('WARN')) return 'text-yellow-400';
    if (line.includes('SUCCESS') || line.includes('PASS')) return 'text-emerald-400';
    return 'text-slate-300';
  };

  // Utility functions for CLEAR and DOWNLOAD
  const handleClearLogs = () => {
    if (activeFeed === 'combined') {
      setCombinedLines([]);
      setFeeds(prev => prev.map(f => ({ ...f, lines: [] })));
    } else {
      setFeeds(prev => prev.map(f => f.id === activeFeed ? { ...f, lines: [] } : f));
    }
  };

  const handleDownloadLogs = () => {
    const element = document.createElement("a");
    const file = new Blob([currentLines.join("\n")], { type: 'text/plain' });
    element.href = URL.createObjectURL(file);
    element.download = `${activeFeed}_logs_${new Date().toISOString().replace(/[:.]/g, '-')}.txt`;
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);
  };

  return (
    <div className="bg-slate-950 border border-slate-800 rounded-lg flex flex-col h-[420px] overflow-hidden shadow-xl">
      <div className="flex items-center justify-between bg-slate-900 border-b border-slate-800 px-3 py-2">
        <div className="text-white font-bold text-sm">Terminal Matrix</div>
        <div className="flex items-center gap-3">
          {staleEventStream && (
            <div className="text-[10px] font-mono text-amber-300">No telemetry events in 30s. Check runtime logs/tasks.</div>
          )}
          <div className={`text-xs font-mono ${connected ? 'text-emerald-400' : 'text-amber-400'}`}>
            {connected ? 'LIVE' : 'DISCONNECTED'}
          </div>
        </div>
      </div>

      <div className="flex bg-slate-900 border-b border-slate-800 p-2 gap-2 flex-wrap items-center">
        <button
          onClick={() => setActiveFeed('combined')}
          className={`px-3 py-1 rounded text-xs font-mono font-bold ${activeFeed === 'combined' ? 'bg-cyan-600 text-white' : 'bg-slate-800 text-slate-400 hover:bg-slate-700'}`}
        >
          ALL_SYSTEMS
        </button>
        {feeds.map((f) => (
          <button
            key={f.id}
            onClick={() => setActiveFeed(f.id)}
            className={`px-3 py-1 rounded text-xs font-mono font-bold ${activeFeed === f.id ? 'bg-cyan-600 text-white' : 'bg-slate-800 text-slate-400 hover:bg-slate-700'}`}
          >
            {f.name.toUpperCase()}
          </button>
        ))}

        {/* Spacer to push utility buttons to the right */}
        <div className="flex-1"></div>

        {/* Utility Buttons */}
        <button 
          onClick={handleClearLogs}
          className="px-3 py-1 rounded text-xs font-mono font-bold bg-slate-800 text-slate-400 hover:bg-red-900/50 hover:text-red-300 transition-colors"
          title="Clear current log view"
        >
          CLEAR
        </button>
        <button 
          onClick={handleDownloadLogs}
          className="px-3 py-1 rounded text-xs font-mono font-bold bg-slate-800 text-slate-400 hover:bg-emerald-900/50 hover:text-emerald-300 transition-colors"
          title="Download current logs"
        >
          DOWNLOAD
        </button>
      </div>

      <div ref={scrollRef} className="flex-1 p-4 overflow-y-auto font-mono text-[11px] leading-relaxed tracking-tight bg-black">
        {currentLines.length === 0 ? (
          <div className="text-slate-600 italic">Awaiting telemetry streams...</div>
        ) : (
          currentLines.map((line, idx) => (
            <div key={`${idx}-${line.slice(0, 32)}`} className={`${getColorForLine(line)} hover:bg-slate-900 px-1 rounded`}>
              {line}
            </div>
          ))
        )}
      </div>
    </div>
  );
}
