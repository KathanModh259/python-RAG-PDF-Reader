import { useCallback } from 'react';
import { invoke } from '@tauri-apps/api/core';

const API_BASE = 'http://127.0.0.1:8765';

interface HealthResponse {
  status: string;
  documents_indexed: number;
  model_loaded: boolean;
}

interface QueryResponse {
  answer: string;
  sources: Array<{
    text: string;
    heading: string;
    source: string;
    score: number;
  }>;
  confidence: number;
}

interface StatsResponse {
  documents_indexed: number;
  model_loaded: boolean;
  vector_store_type: string;
}

interface UploadResponse {
  status: string;
  filename: string;
  chunks: number;
  characters: number;
}

export function useBackend() {
  const getApiUrl = useCallback(async (): Promise<string> => {
    try {
      return await invoke<string>('get_api_url');
    } catch {
      return API_BASE;
    }
  }, []);

  const health = useCallback(async (): Promise<HealthResponse> => {
    const res = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(5000) });
    if (!res.ok) throw new Error(`Health check failed: ${res.status}`);
    return res.json();
  }, []);

  const query = useCallback(async (question: string, mode: string = 'standard'): Promise<QueryResponse> => {
    const res = await fetch(`${API_BASE}/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, mode }),
      signal: AbortSignal.timeout(60000),
    });
    if (!res.ok) throw new Error(`Query failed: ${res.status}`);
    return res.json();
  }, []);

  const upload = useCallback(async (file: File): Promise<UploadResponse> => {
    const form = new FormData();
    form.append('file', file);
    const res = await fetch(`${API_BASE}/upload`, {
      method: 'POST',
      body: form,
      signal: AbortSignal.timeout(120000),
    });
    if (!res.ok) throw new Error(`Upload failed: ${res.status}`);
    return res.json();
  }, []);

  const stats = useCallback(async (): Promise<StatsResponse> => {
    const res = await fetch(`${API_BASE}/stats`, { signal: AbortSignal.timeout(5000) });
    if (!res.ok) throw new Error(`Stats failed: ${res.status}`);
    return res.json();
  }, []);

  const askOllama = useCallback(async (prompt: string): Promise<string> => {
    const result = await query(prompt);
    return result.answer;
  }, [query]);

  return { health, query, upload, stats, askOllama, getApiUrl };
}
