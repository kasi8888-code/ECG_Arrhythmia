'use client';

import { getDecisionStatus, formatConfidence } from '../lib/api';

/**
 * PredictionCard Component
 * Displays AI prediction results with confidence and decision status
 */
export default function PredictionCard({ prediction, isLoading = false }) {
    if (isLoading) {
        return (
            <div className="bg-white border border-gray-200 rounded-xl p-6">
                <div className="animate-pulse">
                    <div className="h-6 bg-gray-200 rounded w-1/3 mb-4"></div>
                    <div className="h-10 bg-gray-200 rounded w-2/3 mb-6"></div>
                    <div className="space-y-3">
                        <div className="h-4 bg-gray-200 rounded w-full"></div>
                        <div className="h-4 bg-gray-200 rounded w-5/6"></div>
                        <div className="h-4 bg-gray-200 rounded w-4/6"></div>
                    </div>
                </div>
            </div>
        );
    }

    if (!prediction) {
        return (
            <div className="bg-gray-50 border border-gray-200 rounded-xl p-6 text-center">
                <svg className="w-16 h-16 mx-auto mb-4 text-gray-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                </svg>
                <p className="text-gray-500 font-medium">No prediction yet</p>
                <p className="text-gray-400 text-sm mt-1">Upload and analyze an ECG file</p>
            </div>
        );
    }

    const {
        prediction: predClass,
        prediction_name: predName,
        confidence,
        probabilities,
        needs_referral: needsReferral,
        referral_reason: referralReason,
    } = prediction;

    const decision = getDecisionStatus(confidence, needsReferral);

    return (
        <div className="bg-white border border-gray-200 rounded-xl overflow-hidden">
            {/* Header */}
            <div className="px-6 py-4 bg-gray-50 border-b border-gray-200">
                <h3 className="text-lg font-semibold text-gray-900">AI Prediction Result</h3>
            </div>

            <div className="p-6">
                {/* Main Prediction */}
                <div className="mb-6">
                    <p className="text-sm text-gray-500 uppercase tracking-wide mb-2">Predicted Arrhythmia Type</p>
                    <h4 className="text-2xl font-bold text-gray-900">{predName}</h4>
                </div>

                {/* Confidence Score */}
                <div className="mb-6">
                    <div className="flex items-center justify-between mb-2">
                        <p className="text-sm text-gray-500 uppercase tracking-wide">Confidence Score</p>
                        <p className="text-2xl font-bold text-blue-600">{formatConfidence(confidence)}</p>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-3 overflow-hidden">
                        <div
                            className={`h-full rounded-full transition-all duration-500 ${confidence >= 0.85
                                    ? 'bg-emerald-500'
                                    : confidence >= 0.7
                                        ? 'bg-amber-500'
                                        : 'bg-red-500'
                                }`}
                            style={{ width: `${confidence * 100}%` }}
                        ></div>
                    </div>
                    <div className="flex justify-between mt-1 text-xs text-gray-400">
                        <span>0%</span>
                        <span>50%</span>
                        <span>100%</span>
                    </div>
                </div>

                {/* Decision Status */}
                <div className={`p-4 rounded-lg border ${decision.bgClass} ${decision.borderClass} mb-6`}>
                    <div className="flex items-center">
                        <span className="text-2xl mr-3">{decision.icon}</span>
                        <div>
                            <p className={`font-semibold ${decision.textClass}`}>{decision.status}</p>
                            {referralReason && (
                                <p className="text-sm text-gray-600 mt-1">{referralReason}</p>
                            )}
                        </div>
                    </div>
                </div>

                {/* Class Probabilities */}
                {probabilities && Object.keys(probabilities).length > 0 && (
                    <div>
                        <p className="text-sm text-gray-500 uppercase tracking-wide mb-3">Class Probabilities</p>
                        <div className="space-y-2">
                            {Object.entries(probabilities)
                                .sort(([, a], [, b]) => b - a)
                                .map(([className, prob]) => (
                                    <div key={className} className="flex items-center">
                                        <span className="w-32 text-sm text-gray-700 truncate" title={className}>
                                            {className}
                                        </span>
                                        <div className="flex-1 mx-3 bg-gray-100 rounded-full h-2 overflow-hidden">
                                            <div
                                                className={`h-full rounded-full ${className === predName ? 'bg-blue-500' : 'bg-gray-300'
                                                    }`}
                                                style={{ width: `${prob * 100}%` }}
                                            ></div>
                                        </div>
                                        <span className="text-sm text-gray-500 w-14 text-right">
                                            {(prob * 100).toFixed(1)}%
                                        </span>
                                    </div>
                                ))}
                        </div>
                    </div>
                )}
            </div>

            {/* Footer */}
            <div className="px-6 py-3 bg-gray-50 border-t border-gray-200">
                <p className="text-xs text-gray-500">
                    ⚠️ This is an AI-assisted analysis. Always consult a qualified cardiologist for medical decisions.
                </p>
            </div>
        </div>
    );
}
