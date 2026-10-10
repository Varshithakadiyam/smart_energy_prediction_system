/**
 * api.js
 * Centralized API client service for Smart Energy Dashboard.
 * Connects to FastAPI backend via configurable VITE_API_URL environment variable.
 */

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

async function fetchJson(endpoint, options = {}) {
  const url = `${API_BASE}${endpoint}`;
  try {
    const isFormData = options.body instanceof FormData;
    const defaultHeaders = isFormData ? {} : { 'Content-Type': 'application/json' };

    const res = await fetch(url, {
      ...options,
      headers: {
        ...defaultHeaders,
        ...(options.headers || {})
      }
    });

    if (!res.ok) {
      let errorMsg = `Server error ${res.status}`;
      try {
        const errData = await res.json();
        if (errData && errData.detail) {
          errorMsg = typeof errData.detail === 'string' ? errData.detail : JSON.stringify(errData.detail);
        }
      } catch (e) {
        // use default status
      }
      throw new Error(errorMsg);
    }

    return await res.json();
  } catch (err) {
    console.error(`API Error [${endpoint}]:`, err);
    throw err;
  }
}

export const api = {
  getHealth: () => fetchJson('/health'),
  getModelInfo: () => fetchJson('/model-info'),
  predictNextHour: (data = null) => fetchJson('/predict', {
    method: 'POST',
    body: JSON.stringify(data ? { data } : {})
  }),
  predictNext24Hours: (data = null) => fetchJson('/predict-next-24-hours', {
    method: 'POST',
    body: JSON.stringify(data ? { data } : {})
  }),
  getAnalytics: (rangeType = 'all', data = null) => fetchJson('/analytics', {
    method: 'POST',
    body: JSON.stringify({ range_type: rangeType, ...(data ? { data } : {}) })
  }),
  getHistoricalData: () => fetchJson('/analytics/historical'),
  detectAnomalies: (data = null) => fetchJson('/anomaly-detection', {
    method: 'POST',
    body: JSON.stringify(data ? { data } : {})
  }),
  getRecommendations: (predictions = null) => fetchJson('/recommendations', {
    method: 'POST',
    body: JSON.stringify(predictions ? { predictions } : {})
  }),
  // User-Specific Upload & Forecast Endpoints
  getTemplateUrl: () => `${API_BASE}/user-forecast/template`,
  uploadUserCsv: (file) => {
    const formData = new FormData();
    formData.append('file', file);
    return fetchJson('/user-forecast/upload', {
      method: 'POST',
      body: formData
    });
  },
  predictUserForecast: (sessionId, horizon = '24h') => fetchJson('/user-forecast/predict', {
    method: 'POST',
    body: JSON.stringify({ session_id: sessionId, horizon })
  }),
  resetUserSession: (sessionId) => fetchJson(`/user-forecast/session/${sessionId}`, {
    method: 'DELETE'
  })
};
