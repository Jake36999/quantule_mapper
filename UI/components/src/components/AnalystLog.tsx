import React, { useState, useMemo } from 'react';
import { ChevronDown, ChevronRight, Terminal, AlertTriangle, Brain } from 'lucide-react';

interface LogEntry {
  id: string;
  timestamp: string;
  level: 'INFO' | 'WARNING' | 'CRITICAL' | 'STEER';
  message: string;
  details?: string;
}

export const AnalystLog: React.FC<{ logs: LogEntry[] }> = ({ logs }) => {
  // Filter consecutive duplicates based on message content
  const filteredLogs = useMemo(() => {
    if (!logs || logs.length === 0) return [];
    
    const uniqueLogs: LogEntry[] = [];
    let lastMessage = "";

    // Process logs (assuming logs are ordered newest first based on Dashboard implementation)
    // We iterate normally but check against the last added unique log
    for (const log of logs) {
        if (log.message !== lastMessage) {
            uniqueLogs.push(log);
            lastMessage = log.message;
        }
    }
    return uniqueLogs;
  }, [logs]);

  return (
    <div className="w-full h-full bg-slate-900/50 border border-slate-800 rounded-lg flex flex-col font-mono text-xs">
      <div className="p-3 border-b border-slate-800 flex items-center gap-2 text-slate-400 font-bold uppercase tracking-wider">
        <Terminal className="w-4 h-4" />
        System / Analyst Logs
      </div>
      <div className="flex-1 overflow-y-auto p-2 space-y-1 scrollbar-thin scrollbar-thumb-slate-700">
        {filteredLogs.length === 0 && (
            <div className="text-slate-600 text-center mt-10">No active telemetry...</div>
        )}
        {filteredLogs.map((log) => (
          <LogItem key={log.id} entry={log} />
        ))}
      </div>
    </div>
  );
};

const LogItem: React.FC<{ entry: LogEntry }> = ({ entry }) => {
  const [isOpen, setIsOpen] = useState(false);
  
  const getIcon = () => {
    switch (entry.level) {
      case 'CRITICAL': return <AlertTriangle className="w-3 h-3 text-rose-500" />;
      case 'STEER': return <Brain className="w-3 h-3 text-gold-500" />;
      default: return <div className="w-3 h-3 rounded-full bg-slate-600" />;
    }
  };

  // Format timestamp safely
  const timeStr = entry.timestamp.includes('T') 
    ? entry.timestamp.split('T')[1]?.split('.')[0] 
    : entry.timestamp;

  return (
    <div className="border border-slate-800/50 rounded bg-black/20 hover:bg-black/40 transition-colors">
      <div 
        className="flex items-center gap-2 p-2 cursor-pointer"
        onClick={() => setIsOpen(!isOpen)}
      >
        <span className="text-slate-600 font-mono text-[10px] w-14">{timeStr}</span>
        {getIcon()}
        <span className={`flex-1 truncate ${entry.level === 'CRITICAL' ? 'text-rose-400' : 'text-slate-300'}`}>
            {entry.message}
        </span>
        {entry.details && (
            isOpen ? <ChevronDown className="w-3 h-3 text-slate-500" /> : <ChevronRight className="w-3 h-3 text-slate-500" />
        )}
      </div>
      
      {isOpen && entry.details && (
        <div className="p-2 pt-0 pl-16 text-slate-500 whitespace-pre-wrap border-t border-slate-800/30 mt-1 text-[10px]">
            {entry.details}
        </div>
      )}
    </div>
  );
};
