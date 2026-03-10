'use client';

import { useState, useMemo } from 'react';
import { getDecisionStatus, formatConfidence } from '../lib/api';

/**
 * HistoryTable Component
 * Displays prediction history with sorting and filtering
 */
export default function HistoryTable({
    history = [],
    onViewPrediction = null,
    onClearHistory = null
}) {
    const [sortField, setSortField] = useState('timestamp');
    const [sortDirection, setSortDirection] = useState('desc');
    const [searchQuery, setSearchQuery] = useState('');
    const [filterDecision, setFilterDecision] = useState('all');

    // Filter and sort history
    const filteredHistory = useMemo(() => {
        let filtered = [...history];

        // Apply search filter
        if (searchQuery) {
            const query = searchQuery.toLowerCase();
            filtered = filtered.filter(item =>
                item.prediction_name?.toLowerCase().includes(query) ||
                item.referral_reason?.toLowerCase().includes(query)
            );
        }

        // Apply decision filter
        if (filterDecision !== 'all') {
            filtered = filtered.filter(item => {
                const decision = getDecisionStatus(item.confidence, item.needs_referral);
                if (filterDecision === 'auto') return decision.status === 'Auto-Classified';
                if (filterDecision === 'monitor') return decision.status === 'Monitor';
                if (filterDecision === 'refer') return decision.status === 'Refer to Cardiologist';
                return true;
            });
        }

        // Apply sorting
        filtered.sort((a, b) => {
            let aVal, bVal;

            if (sortField === 'timestamp') {
                aVal = new Date(a.timestamp).getTime();
                bVal = new Date(b.timestamp).getTime();
            } else if (sortField === 'confidence') {
                aVal = a.confidence;
                bVal = b.confidence;
            } else if (sortField === 'prediction') {
                aVal = a.prediction_name || '';
                bVal = b.prediction_name || '';
            }

            if (sortDirection === 'asc') {
                return aVal > bVal ? 1 : -1;
            }
            return aVal < bVal ? 1 : -1;
        });

        return filtered;
    }, [history, sortField, sortDirection, searchQuery, filterDecision]);

    // Handle sort
    const handleSort = (field) => {
        if (sortField === field) {
            setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
        } else {
            setSortField(field);
            setSortDirection('desc');
        }
    };

    // Export to CSV
    const exportToCSV = () => {
        const headers = ['Timestamp', 'Prediction', 'Confidence', 'Decision', 'Needs Referral'];
        const rows = filteredHistory.map(item => {
            const decision = getDecisionStatus(item.confidence, item.needs_referral);
            return [
                new Date(item.timestamp).toLocaleString(),
                item.prediction_name,
                formatConfidence(item.confidence),
                decision.status,
                item.needs_referral ? 'Yes' : 'No'
            ];
        });

        const csv = [headers, ...rows].map(row => row.join(',')).join('\n');
        const blob = new Blob([csv], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `ecg_predictions_${new Date().toISOString().split('T')[0]}.csv`;
        a.click();
        URL.revokeObjectURL(url);
    };

    // Sort icon
    const SortIcon = ({ field }) => {
        if (sortField !== field) {
            return (
                <svg className="w-4 h-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16V4m0 0L3 8m4-4l4 4m6 0v12m0 0l4-4m-4 4l-4-4" />
                </svg>
            );
        }
        return sortDirection === 'asc' ? (
            <svg className="w-4 h-4 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 15l7-7 7 7" />
            </svg>
        ) : (
            <svg className="w-4 h-4 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
        );
    };

    if (history.length === 0) {
        return (
            <div className="bg-gray-50 border border-gray-200 rounded-xl p-12 text-center">
                <svg className="w-20 h-20 mx-auto mb-6 text-gray-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                </svg>
                <h3 className="text-lg font-semibold text-gray-700 mb-2">No Predictions Yet</h3>
                <p className="text-gray-500">Your prediction history will appear here after analyzing ECG files.</p>
            </div>
        );
    }

    return (
        <div className="bg-white border border-gray-200 rounded-xl overflow-hidden">
            {/* Header with filters */}
            <div className="px-6 py-4 bg-gray-50 border-b border-gray-200">
                <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                    <h3 className="text-lg font-semibold text-gray-900">
                        Prediction History
                        <span className="ml-2 text-sm font-normal text-gray-500">
                            ({filteredHistory.length} of {history.length})
                        </span>
                    </h3>

                    <div className="flex flex-wrap items-center gap-3">
                        {/* Search */}
                        <div className="relative">
                            <input
                                type="text"
                                placeholder="Search predictions..."
                                value={searchQuery}
                                onChange={(e) => setSearchQuery(e.target.value)}
                                className="pl-9 pr-4 py-2 text-sm border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                            />
                            <svg className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                            </svg>
                        </div>

                        {/* Decision filter */}
                        <select
                            value={filterDecision}
                            onChange={(e) => setFilterDecision(e.target.value)}
                            className="px-3 py-2 text-sm border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                        >
                            <option value="all">All Decisions</option>
                            <option value="auto">✅ Auto-Classified</option>
                            <option value="monitor">⚠️ Monitor</option>
                            <option value="refer">❌ Refer</option>
                        </select>

                        {/* Export button */}
                        <button
                            onClick={exportToCSV}
                            className="px-3 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-50"
                        >
                            Export CSV
                        </button>

                        {/* Clear button */}
                        {onClearHistory && (
                            <button
                                onClick={onClearHistory}
                                className="px-3 py-2 text-sm font-medium text-red-600 bg-white border border-red-200 rounded-lg hover:bg-red-50"
                            >
                                Clear All
                            </button>
                        )}
                    </div>
                </div>
            </div>

            {/* Table */}
            <div className="overflow-x-auto">
                <table className="w-full">
                    <thead className="bg-gray-50 border-b border-gray-200">
                        <tr>
                            <th
                                className="px-6 py-3 text-left text-xs font-semibold text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100"
                                onClick={() => handleSort('timestamp')}
                            >
                                <div className="flex items-center space-x-1">
                                    <span>Timestamp</span>
                                    <SortIcon field="timestamp" />
                                </div>
                            </th>
                            <th
                                className="px-6 py-3 text-left text-xs font-semibold text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100"
                                onClick={() => handleSort('prediction')}
                            >
                                <div className="flex items-center space-x-1">
                                    <span>Prediction</span>
                                    <SortIcon field="prediction" />
                                </div>
                            </th>
                            <th
                                className="px-6 py-3 text-left text-xs font-semibold text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100"
                                onClick={() => handleSort('confidence')}
                            >
                                <div className="flex items-center space-x-1">
                                    <span>Confidence</span>
                                    <SortIcon field="confidence" />
                                </div>
                            </th>
                            <th className="px-6 py-3 text-left text-xs font-semibold text-gray-500 uppercase tracking-wider">
                                Decision
                            </th>
                            <th className="px-6 py-3 text-right text-xs font-semibold text-gray-500 uppercase tracking-wider">
                                Actions
                            </th>
                        </tr>
                    </thead>
                    <tbody className="divide-y divide-gray-200">
                        {filteredHistory.map((item, index) => {
                            const decision = getDecisionStatus(item.confidence, item.needs_referral);
                            return (
                                <tr
                                    key={item.id || index}
                                    className="hover:bg-gray-50 transition-colors"
                                >
                                    <td className="px-6 py-4 whitespace-nowrap">
                                        <div className="text-sm text-gray-900">
                                            {new Date(item.timestamp).toLocaleDateString()}
                                        </div>
                                        <div className="text-xs text-gray-500">
                                            {new Date(item.timestamp).toLocaleTimeString()}
                                        </div>
                                    </td>
                                    <td className="px-6 py-4 whitespace-nowrap">
                                        <span className="text-sm font-medium text-gray-900">
                                            {item.prediction_name}
                                        </span>
                                    </td>
                                    <td className="px-6 py-4 whitespace-nowrap">
                                        <div className="flex items-center">
                                            <div className="w-16 bg-gray-200 rounded-full h-2 mr-2">
                                                <div
                                                    className={`h-2 rounded-full ${item.confidence >= 0.85
                                                            ? 'bg-emerald-500'
                                                            : item.confidence >= 0.7
                                                                ? 'bg-amber-500'
                                                                : 'bg-red-500'
                                                        }`}
                                                    style={{ width: `${item.confidence * 100}%` }}
                                                ></div>
                                            </div>
                                            <span className="text-sm text-gray-600">
                                                {formatConfidence(item.confidence)}
                                            </span>
                                        </div>
                                    </td>
                                    <td className="px-6 py-4 whitespace-nowrap">
                                        <span className={`inline-flex items-center px-2.5 py-1 rounded-full text-xs font-medium ${decision.bgClass} ${decision.textClass}`}>
                                            <span className="mr-1">{decision.icon}</span>
                                            {decision.status}
                                        </span>
                                    </td>
                                    <td className="px-6 py-4 whitespace-nowrap text-right">
                                        {onViewPrediction && (
                                            <button
                                                onClick={() => onViewPrediction(item)}
                                                className="text-blue-600 hover:text-blue-800 text-sm font-medium"
                                            >
                                                View Details
                                            </button>
                                        )}
                                    </td>
                                </tr>
                            );
                        })}
                    </tbody>
                </table>
            </div>

            {/* Empty filtered state */}
            {filteredHistory.length === 0 && (
                <div className="p-8 text-center">
                    <p className="text-gray-500">No predictions match your filters.</p>
                </div>
            )}
        </div>
    );
}
