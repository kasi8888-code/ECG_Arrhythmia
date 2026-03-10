'use client';

import { useState, useEffect } from 'react';
import HistoryTable from '@/components/HistoryTable';
import PredictionCard from '@/components/PredictionCard';
import GradCamOverlay from '@/components/GradCamOverlay';
import { getPredictionHistory, clearPredictionHistory } from '@/lib/api';

/**
 * History Page
 * View and manage prediction history
 */
export default function HistoryPage() {
    const [history, setHistory] = useState([]);
    const [selectedPrediction, setSelectedPrediction] = useState(null);
    const [showClearConfirm, setShowClearConfirm] = useState(false);

    // Load history on mount
    useEffect(() => {
        setHistory(getPredictionHistory());
    }, []);

    // Handle view prediction details
    const handleViewPrediction = (prediction) => {
        setSelectedPrediction(prediction);
    };

    // Handle clear history
    const handleClearHistory = () => {
        if (showClearConfirm) {
            clearPredictionHistory();
            setHistory([]);
            setSelectedPrediction(null);
            setShowClearConfirm(false);
        } else {
            setShowClearConfirm(true);
        }
    };

    // Close detail view
    const handleCloseDetail = () => {
        setSelectedPrediction(null);
    };

    return (
        <div className="space-y-8">
            {/* Page Header */}
            <header className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                <div>
                    <h1 className="text-2xl font-bold text-gray-900">Prediction History</h1>
                    <p className="text-gray-600 mt-1">
                        View and manage your past ECG predictions
                    </p>
                </div>

                {history.length > 0 && (
                    <div className="flex items-center gap-3">
                        {showClearConfirm ? (
                            <>
                                <span className="text-sm text-red-600">Are you sure?</span>
                                <button
                                    onClick={handleClearHistory}
                                    className="px-4 py-2 bg-red-600 text-white text-sm font-medium rounded-lg hover:bg-red-700"
                                >
                                    Yes, Clear All
                                </button>
                                <button
                                    onClick={() => setShowClearConfirm(false)}
                                    className="px-4 py-2 bg-gray-200 text-gray-700 text-sm font-medium rounded-lg hover:bg-gray-300"
                                >
                                    Cancel
                                </button>
                            </>
                        ) : (
                            <button
                                onClick={handleClearHistory}
                                className="px-4 py-2 border border-red-200 text-red-600 text-sm font-medium rounded-lg hover:bg-red-50"
                            >
                                Clear History
                            </button>
                        )}
                    </div>
                )}
            </header>

            {/* Stats Summary */}
            {history.length > 0 && (
                <section className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div className="bg-white p-4 rounded-xl border border-gray-200">
                        <p className="text-sm text-gray-500">Total Predictions</p>
                        <p className="text-2xl font-bold text-gray-900">{history.length}</p>
                    </div>
                    <div className="bg-white p-4 rounded-xl border border-gray-200">
                        <p className="text-sm text-gray-500">Auto-Classified</p>
                        <p className="text-2xl font-bold text-emerald-600">
                            {history.filter(h => h.confidence >= 0.85 && !h.needs_referral).length}
                        </p>
                    </div>
                    <div className="bg-white p-4 rounded-xl border border-gray-200">
                        <p className="text-sm text-gray-500">Monitor</p>
                        <p className="text-2xl font-bold text-amber-600">
                            {history.filter(h => h.confidence >= 0.7 && h.confidence < 0.85).length}
                        </p>
                    </div>
                    <div className="bg-white p-4 rounded-xl border border-gray-200">
                        <p className="text-sm text-gray-500">Referrals</p>
                        <p className="text-2xl font-bold text-red-600">
                            {history.filter(h => h.needs_referral || h.confidence < 0.7).length}
                        </p>
                    </div>
                </section>
            )}

            {/* History Table */}
            <section>
                <HistoryTable
                    history={history}
                    onViewPrediction={handleViewPrediction}
                    onClearHistory={handleClearHistory}
                />
            </section>

            {/* Detail Modal */}
            {selectedPrediction && (
                <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
                    <div className="bg-white rounded-2xl shadow-2xl max-w-4xl w-full max-h-[90vh] overflow-y-auto">
                        {/* Modal Header */}
                        <div className="px-6 py-4 border-b border-gray-200 flex items-center justify-between sticky top-0 bg-white">
                            <div>
                                <h2 className="text-xl font-semibold text-gray-900">Prediction Details</h2>
                                <p className="text-sm text-gray-500">
                                    {new Date(selectedPrediction.timestamp).toLocaleString()}
                                </p>
                            </div>
                            <button
                                onClick={handleCloseDetail}
                                className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
                            >
                                <svg className="w-5 h-5 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                                </svg>
                            </button>
                        </div>

                        {/* Modal Content */}
                        <div className="p-6 grid grid-cols-1 lg:grid-cols-2 gap-6">
                            <PredictionCard prediction={selectedPrediction} />
                            <GradCamOverlay
                                visualizationBase64={selectedPrediction.visualization_base64}
                                heatmap={selectedPrediction.heatmap}
                                keyRegions={selectedPrediction.key_regions}
                            />
                        </div>
                    </div>
                </div>
            )}

            {/* Data Persistence Info */}
            <section className="bg-gray-50 border border-gray-200 rounded-xl p-4 text-sm text-gray-600">
                <p>
                    💾 <strong>Note:</strong> Prediction history is stored locally in your browser.
                    Data will persist across sessions but is not synced across devices.
                </p>
            </section>
        </div>
    );
}
