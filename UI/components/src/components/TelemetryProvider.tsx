import React, { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react';

export type TelemetryEvent = {
  type?: string;
  ts?: string;
  timestamp?: string;
  source?: string;
  severity?: string;
  feed_id?: string;
  line?: string;
  state?: string;
  details?: string;
  message?: string;
  payload?: unknown;
  [key: string]: unknown;
};

type TelemetryContextValue = {
  connected: boolean;
  lastEventAt: number | null;
  events: TelemetryEvent[];
  send: (payload: unknown) => void;
};

const MAX_EVENTS = 500;
const TelemetryContext = createContext<TelemetryContextValue | null>(null);

const telemetryUrl = (): string => {
  const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
  const configured = process.env.REACT_APP_API_BASE_URL;
  if (configured && configured.trim()) {
    try {
      const parsed = new URL(configured.trim());
      parsed.protocol = parsed.protocol === 'https:' ? 'wss:' : 'ws:';
      parsed.pathname = '/ws/telemetry';
      parsed.search = '';
      parsed.hash = '';
      return parsed.toString();
    } catch {
      // Fall through to the current host for relative or malformed config.
    }
  }
  return `${protocol}://${window.location.host}/ws/telemetry`;
};

const flattenEvents = (raw: unknown): TelemetryEvent[] => {
  if (Array.isArray(raw)) {
    return raw.flatMap(flattenEvents);
  }
  if (raw && typeof raw === 'object') {
    const event = raw as TelemetryEvent;
    if (Array.isArray(event.payload) && !event.type) {
      return event.payload.flatMap(flattenEvents);
    }
    return [event];
  }
  return [];
};

export function TelemetryProvider({ children }: { children: React.ReactNode }) {
  const [connected, setConnected] = useState(false);
  const [lastEventAt, setLastEventAt] = useState<number | null>(null);
  const [events, setEvents] = useState<TelemetryEvent[]>([]);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectRef = useRef<number | null>(null);
  const retryRef = useRef(0);
  const closedRef = useRef(false);

  const appendEvents = useCallback((next: TelemetryEvent[]) => {
    if (next.length === 0) {
      return;
    }
    setLastEventAt(Date.now());
    setEvents((prev) => {
      const seen = new Set(prev.slice(-80).map((event) => `${event.type}|${event.ts || event.timestamp}|${event.feed_id}|${event.line || event.message || ''}`));
      const filtered = next.filter((event) => {
        const key = `${event.type}|${event.ts || event.timestamp}|${event.feed_id}|${event.line || event.message || ''}`;
        if (seen.has(key)) {
          return false;
        }
        seen.add(key);
        return true;
      });
      return [...prev, ...filtered].slice(-MAX_EVENTS);
    });
  }, []);

  useEffect(() => {
    closedRef.current = false;

    const connect = () => {
      if (closedRef.current) {
        return;
      }
      const socket = new WebSocket(telemetryUrl());
      wsRef.current = socket;

      socket.onopen = () => {
        retryRef.current = 0;
        setConnected(true);
      };

      socket.onmessage = (event) => {
        try {
          appendEvents(flattenEvents(JSON.parse(event.data)));
        } catch (err) {
          console.error('Telemetry parse error:', err);
        }
      };

      socket.onerror = () => {
        setConnected(false);
      };

      socket.onclose = () => {
        setConnected(false);
        if (closedRef.current) {
          return;
        }
        const delay = Math.min(15000, 1000 * 2 ** retryRef.current);
        retryRef.current = Math.min(retryRef.current + 1, 4);
        reconnectRef.current = window.setTimeout(connect, delay);
      };
    };

    connect();
    return () => {
      closedRef.current = true;
      if (reconnectRef.current !== null) {
        window.clearTimeout(reconnectRef.current);
      }
      if (wsRef.current && (wsRef.current.readyState === WebSocket.OPEN || wsRef.current.readyState === WebSocket.CONNECTING)) {
        wsRef.current.close();
      }
    };
  }, [appendEvents]);

  const send = useCallback((payload: unknown) => {
    const socket = wsRef.current;
    if (!socket || socket.readyState !== WebSocket.OPEN) {
      return;
    }
    socket.send(JSON.stringify(payload));
  }, []);

  const value = useMemo(
    () => ({ connected, lastEventAt, events, send }),
    [connected, events, lastEventAt, send],
  );

  return <TelemetryContext.Provider value={value}>{children}</TelemetryContext.Provider>;
}

export function useTelemetry() {
  const value = useContext(TelemetryContext);
  if (!value) {
    throw new Error('useTelemetry must be used inside TelemetryProvider');
  }
  return value;
}
