'use client';

import { useState } from 'react';

/**
 * GradCamOverlay Component
 * Displays Grad-CAM visualization for model explainability
 */
export default function GradCamOverlay({
    visualizationBase64 = null,
    heatmap = null,
    keyRegions = [],
    isLoading = false,
    onToggle = null,
    showOverlay = true
}) {
    const [isExpanded, setIsExpanded] = useState(false);

    if (isLoading) {
        return (
            <div className="bg-white border border-gray-200 rounded-xl p-6">
                <div className="animate-pulse">
                    <div className="h-6 bg-gray-200 rounded w-1/3 mb-4"></div>
                    <div className="h-48 bg-gray-200 rounded w-full"></div>
                </div>
            </div>
        );
    }

    if (!visualizationBase64 && !heatmap) {
        return (
            <div className="bg-gray-50 border border-gray-200 rounded-xl p-6 text-center">
                <svg className="w-16 h-16 mx-auto mb-4 text-gray-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                </svg>
                <p className="text-gray-500 font-medium">No explanation available</p>
                <p className="text-gray-400 text-sm mt-1">Analyze an ECG with explanation enabled</p>
            </div>
        );
    }

    return (
        <div className="bg-white border border-gray-200 rounded-xl overflow-hidden">
            {/* Header */}
            <div className="px-6 py-4 bg-gray-50 border-b border-gray-200 flex items-center justify-between">
                <div>
                    <h3 className="text-lg font-semibold text-gray-900">Model Explanation</h3>
                    <p className="text-sm text-gray-500">Grad-CAM Visualization</p>
                </div>

                {/* Toggle Switch */}
                {onToggle && (
                    <div className="flex items-center space-x-3">
                        <span className="text-sm text-gray-600">Show Overlay</span>
                        <button
                            onClick={() => onToggle(!showOverlay)}
                            className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${showOverlay ? 'bg-blue-600' : 'bg-gray-300'
                                }`}
                        >
                            <span
                                className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${showOverlay ? 'translate-x-6' : 'translate-x-1'
                                    }`}
                            />
                        </button>
                    </div>
                )}
            </div>

            <div className="p-6">
                {/* Explanation Tooltip */}
                <div className="mb-4 p-3 bg-blue-50 border border-blue-200 rounded-lg">
                    <div className="flex items-start">
                        <svg className="w-5 h-5 text-blue-500 mt-0.5 mr-3 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                        <p className="text-sm text-blue-700">
                            <span className="font-semibold">Highlighted regions</span> show the parts of the ECG waveform that most influenced the model's prediction. Warmer colors indicate higher importance.
                        </p>
                    </div>
                </div>

                {/* Visualization Image */}
                {visualizationBase64 && (
                    <div
                        className={`relative rounded-lg overflow-hidden border border-gray-200 cursor-pointer transition-all ${isExpanded ? 'fixed inset-4 z-50 bg-white p-4' : ''
                            }`}
                        onClick={() => setIsExpanded(!isExpanded)}
                    >
                        {isExpanded && (
                            <button
                                className="absolute top-2 right-2 z-10 p-2 bg-white rounded-full shadow-lg hover:bg-gray-100"
                                onClick={(e) => {
                                    e.stopPropagation();
                                    setIsExpanded(false);
                                }}
                            >
                                <svg className="w-5 h-5 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                                </svg>
                            </button>
                        )}
                        <img
                            src={`data:image/png;base64,${visualizationBase64}`}
                            alt="Grad-CAM Visualization"
                            className={`w-full ${isExpanded ? 'h-full object-contain' : 'h-auto'}`}
                        />
                        {!isExpanded && (
                            <div className="absolute bottom-2 right-2 px-2 py-1 bg-black bg-opacity-50 text-white text-xs rounded">
                                Click to expand
                            </div>
                        )}
                    </div>
                )}

                {/* Key Regions */}
                {keyRegions && keyRegions.length > 0 && (
                    <div className="mt-6">
                        <h4 className="text-sm font-semibold text-gray-700 mb-3">Key Regions of Interest</h4>
                        <div className="space-y-2">
                            {keyRegions.map((region, index) => (
                                <div
                                    key={index}
                                    className="flex items-center justify-between p-3 bg-gray-50 rounded-lg"
                                >
                                    <div>
                                        <span className="text-sm font-medium text-gray-800">
                                            Region {index + 1}
                                        </span>
                                        {region.start_ms !== undefined && region.end_ms !== undefined && (
                                            <span className="text-sm text-gray-500 ml-2">
                                                ({region.start_ms.toFixed(0)} - {region.end_ms.toFixed(0)} ms)
                                            </span>
                                        )}
                                    </div>
                                    {region.importance !== undefined && (
                                        <div className="flex items-center">
                                            <div className="w-20 bg-gray-200 rounded-full h-2 mr-2">
                                                <div
                                                    className="bg-red-500 h-2 rounded-full"
                                                    style={{ width: `${region.importance * 100}%` }}
                                                ></div>
                                            </div>
                                            <span className="text-sm text-gray-600">
                                                {(region.importance * 100).toFixed(0)}%
                                            </span>
                                        </div>
                                    )}
                                </div>
                            ))}
                        </div>
                    </div>
                )}

                {/* Color Legend */}
                <div className="mt-6 pt-4 border-t border-gray-200">
                    <h4 className="text-sm font-semibold text-gray-700 mb-3">Importance Scale</h4>
                    <div className="flex items-center space-x-2">
                        <span className="text-xs text-gray-500">Low</span>
                        <div className="flex-1 h-3 rounded-full bg-gradient-to-r from-blue-200 via-yellow-300 via-orange-400 to-red-500"></div>
                        <span className="text-xs text-gray-500">High</span>
                    </div>
                </div>
            </div>
        </div>
    );
}
