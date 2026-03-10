'use client';

import { useEffect, useRef, useState } from 'react';
import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend,
    Filler,
} from 'chart.js';
import { Line } from 'react-chartjs-2';

// Register Chart.js components
ChartJS.register(
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend,
    Filler
);

/**
 * ECGPlot Component
 * Visualizes ECG waveform with Chart.js
 * Supports zoom, pan, and R-peak highlighting
 */
export default function ECGPlot({
    signal,
    samplingRate = 125,
    heatmap = null,
    showHeatmap = false,
    rPeaks = [],
    title = 'ECG Waveform',
    height = 300
}) {
    const chartRef = useRef(null);
    const [viewMode, setViewMode] = useState('raw'); // 'raw' or 'cleaned'

    // Generate time axis in milliseconds
    const timeAxis = signal
        ? signal.map((_, i) => ((i / samplingRate) * 1000).toFixed(0))
        : [];

    // Process signal based on view mode
    const processedSignal = signal ? (
        viewMode === 'cleaned' ? cleanSignal(signal) : signal
    ) : [];

    // Chart configuration
    const chartData = {
        labels: timeAxis,
        datasets: [
            // Main ECG signal
            {
                label: 'ECG Signal',
                data: processedSignal,
                borderColor: '#1E40AF',
                backgroundColor: 'transparent',
                borderWidth: 1.5,
                pointRadius: 0,
                tension: 0.1,
                fill: false,
            },
            // Heatmap overlay (if enabled)
            ...(showHeatmap && heatmap ? [{
                label: 'Grad-CAM',
                data: processedSignal.map((val, i) =>
                    heatmap[i] ? val : null
                ),
                borderColor: 'rgba(220, 38, 38, 0.8)',
                backgroundColor: (context) => {
                    if (!heatmap || !context.raw) return 'transparent';
                    const index = context.dataIndex;
                    const intensity = heatmap[index] || 0;
                    return `rgba(220, 38, 38, ${intensity * 0.4})`;
                },
                borderWidth: 2,
                pointRadius: 0,
                fill: true,
                tension: 0.1,
            }] : []),
            // R-peaks markers
            ...(rPeaks.length > 0 ? [{
                label: 'R-Peaks',
                data: timeAxis.map((_, i) =>
                    rPeaks.includes(i) ? processedSignal[i] : null
                ),
                borderColor: '#059669',
                backgroundColor: '#059669',
                borderWidth: 0,
                pointRadius: 6,
                pointStyle: 'triangle',
                showLine: false,
            }] : []),
        ],
    };

    const chartOptions = {
        responsive: true,
        maintainAspectRatio: false,
        animation: {
            duration: 0, // Disable animations for medical data
        },
        interaction: {
            mode: 'index',
            intersect: false,
        },
        plugins: {
            legend: {
                display: true,
                position: 'top',
                labels: {
                    font: {
                        family: 'Inter, system-ui, sans-serif',
                        size: 12,
                    },
                    usePointStyle: true,
                    padding: 20,
                },
            },
            title: {
                display: true,
                text: title,
                font: {
                    family: 'Inter, system-ui, sans-serif',
                    size: 14,
                    weight: '600',
                },
                color: '#111827',
                padding: {
                    bottom: 16,
                },
            },
            tooltip: {
                backgroundColor: 'rgba(17, 24, 39, 0.9)',
                titleFont: {
                    family: 'Inter, system-ui, sans-serif',
                    size: 12,
                },
                bodyFont: {
                    family: 'Inter, system-ui, sans-serif',
                    size: 11,
                },
                padding: 12,
                cornerRadius: 8,
                callbacks: {
                    title: (items) => `Time: ${items[0].label} ms`,
                    label: (context) => {
                        if (context.dataset.label === 'ECG Signal') {
                            return `Amplitude: ${context.raw?.toFixed(4) || 'N/A'}`;
                        }
                        if (context.dataset.label === 'Grad-CAM') {
                            const intensity = heatmap?.[context.dataIndex] || 0;
                            return `Importance: ${(intensity * 100).toFixed(1)}%`;
                        }
                        return context.dataset.label;
                    },
                },
            },
        },
        scales: {
            x: {
                title: {
                    display: true,
                    text: 'Time (ms)',
                    font: {
                        family: 'Inter, system-ui, sans-serif',
                        size: 12,
                        weight: '500',
                    },
                    color: '#6B7280',
                },
                ticks: {
                    font: {
                        family: 'Inter, system-ui, sans-serif',
                        size: 10,
                    },
                    color: '#9CA3AF',
                    maxTicksLimit: 10,
                },
                grid: {
                    color: '#F3F4F6',
                },
            },
            y: {
                title: {
                    display: true,
                    text: 'Amplitude (mV)',
                    font: {
                        family: 'Inter, system-ui, sans-serif',
                        size: 12,
                        weight: '500',
                    },
                    color: '#6B7280',
                },
                ticks: {
                    font: {
                        family: 'Inter, system-ui, sans-serif',
                        size: 10,
                    },
                    color: '#9CA3AF',
                },
                grid: {
                    color: '#F3F4F6',
                },
            },
        },
    };

    // Simple signal cleaning (baseline wander removal)
    function cleanSignal(rawSignal) {
        if (!rawSignal || rawSignal.length === 0) return rawSignal;

        // Simple moving average for baseline
        const windowSize = 25;
        const baseline = [];

        for (let i = 0; i < rawSignal.length; i++) {
            const start = Math.max(0, i - windowSize);
            const end = Math.min(rawSignal.length, i + windowSize);
            const window = rawSignal.slice(start, end);
            baseline.push(window.reduce((a, b) => a + b, 0) / window.length);
        }

        // Subtract baseline
        return rawSignal.map((val, i) => val - baseline[i] + 0.5);
    }

    if (!signal || signal.length === 0) {
        return (
            <div
                className="flex items-center justify-center bg-gray-50 border border-gray-200 rounded-xl"
                style={{ height }}
            >
                <div className="text-center text-gray-500">
                    <svg className="w-16 h-16 mx-auto mb-4 text-gray-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                    </svg>
                    <p className="text-sm font-medium">No ECG data to display</p>
                    <p className="text-xs text-gray-400 mt-1">Upload a file to visualize the waveform</p>
                </div>
            </div>
        );
    }

    return (
        <div className="bg-white border border-gray-200 rounded-xl p-4">
            {/* Controls */}
            <div className="flex flex-wrap items-center justify-between gap-4 mb-4">
                <div className="flex items-center space-x-2">
                    <span className="text-sm font-medium text-gray-700">View:</span>
                    <div className="flex rounded-lg overflow-hidden border border-gray-200">
                        <button
                            onClick={() => setViewMode('raw')}
                            className={`px-3 py-1.5 text-sm font-medium transition-colors ${viewMode === 'raw'
                                    ? 'bg-blue-600 text-white'
                                    : 'bg-white text-gray-600 hover:bg-gray-50'
                                }`}
                        >
                            Raw ECG
                        </button>
                        <button
                            onClick={() => setViewMode('cleaned')}
                            className={`px-3 py-1.5 text-sm font-medium transition-colors ${viewMode === 'cleaned'
                                    ? 'bg-blue-600 text-white'
                                    : 'bg-white text-gray-600 hover:bg-gray-50'
                                }`}
                        >
                            Cleaned ECG
                        </button>
                    </div>
                </div>

                {/* Signal info badges */}
                <div className="flex items-center space-x-3">
                    <span className="px-2 py-1 bg-gray-100 rounded text-xs text-gray-600">
                        {signal.length} samples
                    </span>
                    <span className="px-2 py-1 bg-gray-100 rounded text-xs text-gray-600">
                        {samplingRate} Hz
                    </span>
                    <span className="px-2 py-1 bg-gray-100 rounded text-xs text-gray-600">
                        {((signal.length / samplingRate) * 1000).toFixed(0)} ms
                    </span>
                </div>
            </div>

            {/* Chart */}
            <div style={{ height }}>
                <Line ref={chartRef} data={chartData} options={chartOptions} />
            </div>

            {/* Chart legend explanation */}
            {showHeatmap && heatmap && (
                <div className="mt-4 p-3 bg-amber-50 border border-amber-200 rounded-lg">
                    <p className="text-sm text-amber-800">
                        <span className="font-semibold">💡 Grad-CAM Overlay:</span> Red-highlighted regions show areas that most influenced the model's prediction.
                    </p>
                </div>
            )}
        </div>
    );
}
