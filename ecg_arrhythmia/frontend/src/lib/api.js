'use client';

/**
 * API Client for ECG Arrhythmia Detection System
 * Integrates with FastAPI backend endpoints
 */

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

/**
 * Health check endpoint
 * @returns {Promise<{status: string, model_loaded: boolean, device: string}>}
 */
export async function healthCheck() {
  const response = await fetch(`${API_BASE}/health`);
  if (!response.ok) {
    throw new Error('Backend API is not available');
  }
  return response.json();
}

/**
 * Get class information
 * @returns {Promise<{classes: Array, referral_threshold: number}>}
 */
export async function getClassInfo() {
  const response = await fetch(`${API_BASE}/class_info`);
  if (!response.ok) {
    throw new Error('Failed to fetch class info');
  }
  return response.json();
}

/**
 * Get model information
 * @returns {Promise<Object>}
 */
export async function getModelInfo() {
  const response = await fetch(`${API_BASE}/model_info`);
  if (!response.ok) {
    throw new Error('Failed to fetch model info');
  }
  return response.json();
}

/**
 * Predict arrhythmia for a single ECG beat
 * @param {number[]} signal - ECG signal (187 samples)
 * @returns {Promise<Object>}
 */
export async function predictECG(signal) {
  const response = await fetch(`${API_BASE}/predict`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ signal }),
  });
  
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Prediction failed');
  }
  
  return response.json();
}

/**
 * Predict with Grad-CAM explanation
 * @param {number[]} signal - ECG signal (187 samples)
 * @param {boolean} includeVisualization - Whether to include base64 visualization
 * @returns {Promise<Object>}
 */
export async function predictWithExplanation(signal, includeVisualization = true) {
  const response = await fetch(
    `${API_BASE}/predict_with_explanation?include_visualization=${includeVisualization}`,
    {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ signal }),
    }
  );
  
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Prediction with explanation failed');
  }
  
  return response.json();
}

/**
 * Batch prediction for multiple beats
 * @param {number[][]} signals - Array of ECG signals
 * @returns {Promise<Object>}
 */
export async function predictBatch(signals) {
  const response = await fetch(`${API_BASE}/predict_batch`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ signals }),
  });
  
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Batch prediction failed');
  }
  
  return response.json();
}

/**
 * Parse CSV file containing ECG data
 * @param {File} file - CSV file
 * @returns {Promise<{signals: number[][], samplingRate: number, signalLength: number}>}
 */
export async function parseECGFile(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    
    reader.onload = (event) => {
      try {
        const text = event.target.result;
        const lines = text.trim().split('\n');
        const signals = [];
        
        for (const line of lines) {
          const values = line.split(',').map(v => parseFloat(v.trim()));
          // Filter out NaN values and validate
          if (values.length > 0 && !values.some(isNaN)) {
            signals.push(values);
          }
        }
        
        if (signals.length === 0) {
          reject(new Error('No valid ECG data found in file'));
          return;
        }
        
        // Determine signal length (should be 187 for MIT-BIH dataset)
        const signalLength = signals[0].length;
        
        resolve({
          signals,
          samplingRate: 125, // Default for MIT-BIH Kaggle dataset
          signalLength,
          beatCount: signals.length,
        });
      } catch (error) {
        reject(new Error('Failed to parse ECG file: ' + error.message));
      }
    };
    
    reader.onerror = () => reject(new Error('Failed to read file'));
    reader.readAsText(file);
  });
}

/**
 * Get decision status based on confidence and referral
 * @param {number} confidence - Confidence score (0-1)
 * @param {boolean} needsReferral - Whether referral is needed
 * @returns {{status: string, color: string, icon: string}}
 */
export function getDecisionStatus(confidence, needsReferral) {
  if (needsReferral || confidence < 0.7) {
    return {
      status: 'Refer to Cardiologist',
      color: 'red',
      icon: '❌',
      bgClass: 'bg-red-50',
      textClass: 'text-red-700',
      borderClass: 'border-red-200',
    };
  } else if (confidence < 0.85) {
    return {
      status: 'Monitor',
      color: 'yellow',
      icon: '⚠️',
      bgClass: 'bg-amber-50',
      textClass: 'text-amber-700',
      borderClass: 'border-amber-200',
    };
  } else {
    return {
      status: 'Auto-Classified',
      color: 'green',
      icon: '✅',
      bgClass: 'bg-emerald-50',
      textClass: 'text-emerald-700',
      borderClass: 'border-emerald-200',
    };
  }
}

/**
 * Format confidence as percentage
 * @param {number} confidence - Confidence score (0-1)
 * @returns {string}
 */
export function formatConfidence(confidence) {
  return `${(confidence * 100).toFixed(1)}%`;
}

/**
 * Save prediction to history in localStorage
 * @param {Object} prediction - Prediction result
 */
export function savePredictionToHistory(prediction) {
  const history = getPredictionHistory();
  const entry = {
    ...prediction,
    id: Date.now(),
    timestamp: new Date().toISOString(),
  };
  history.unshift(entry);
  // Keep only last 100 entries
  const trimmed = history.slice(0, 100);
  localStorage.setItem('ecg_prediction_history', JSON.stringify(trimmed));
  return entry;
}

/**
 * Get prediction history from localStorage
 * @returns {Array}
 */
export function getPredictionHistory() {
  if (typeof window === 'undefined') return [];
  const stored = localStorage.getItem('ecg_prediction_history');
  return stored ? JSON.parse(stored) : [];
}

/**
 * Clear prediction history
 */
export function clearPredictionHistory() {
  localStorage.removeItem('ecg_prediction_history');
}
