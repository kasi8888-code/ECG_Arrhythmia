'use client';

import { useState, useEffect } from 'react';
import { getModelInfo, getClassInfo } from '@/lib/api';

/**
 * About Model Page
 * Detailed information about the AI model and system
 */
export default function AboutPage() {
    const [modelInfo, setModelInfo] = useState(null);
    const [classInfo, setClassInfo] = useState(null);
    const [isLoading, setIsLoading] = useState(true);

    useEffect(() => {
        Promise.all([
            getModelInfo().catch(() => null),
            getClassInfo().catch(() => null),
        ]).then(([model, classes]) => {
            setModelInfo(model);
            setClassInfo(classes);
            setIsLoading(false);
        });
    }, []);

    return (
        <div className="space-y-8">
            {/* Page Header */}
            <header>
                <h1 className="text-2xl font-bold text-gray-900">About the Model</h1>
                <p className="text-gray-600 mt-1">
                    Technical details about the ECG Arrhythmia Detection System
                </p>
            </header>

            {/* Model Overview */}
            <section className="bg-white rounded-xl border border-gray-200 p-6">
                <h2 className="text-lg font-semibold text-gray-900 mb-4">Model Overview</h2>
                <div className="prose prose-blue max-w-none text-gray-700">
                    <p>
                        This system uses a <strong>Hybrid Fusion Architecture</strong> that combines
                        deep learning with traditional signal processing for robust arrhythmia detection.
                    </p>
                    <p>
                        The model processes ECG beats from the MIT-BIH Arrhythmia Database and classifies
                        them according to the <strong>AAMI EC57 standard</strong> into 5 classes.
                    </p>
                </div>
            </section>

            {/* Architecture Diagram */}
            <section className="bg-white rounded-xl border border-gray-200 p-6">
                <h2 className="text-lg font-semibold text-gray-900 mb-4">Model Architecture</h2>
                <div className="bg-gray-50 rounded-lg p-6 font-mono text-sm overflow-x-auto">
                    <pre className="text-gray-700 whitespace-pre">
                        {`Input Beat (187 samples) ────────────────────┐
                                             │
    ┌────────────────────────────────────────┤
    │                                        │
    ▼                                        ▼
┌─────────────────┐                 ┌─────────────────┐
│  1D CNN Backbone│                 │ Feature Extractor│
│  (4 Conv Blocks)│                 │ (25 Features)    │
│  32→64→128→256  │                 │ Morphological,   │
│                 │                 │ Statistical,     │
│                 │                 │ Wavelet, etc.    │
└────────┬────────┘                 └────────┬────────┘
         │                                   │
         │ Global Avg Pool                   │ MLP
         │                                   │
         └─────────────┬─────────────────────┘
                       │
                       ▼
              ┌────────────────┐
              │  Fusion Layer  │
              │   (128 → 64)   │
              └───────┬────────┘
                      │
                      ▼
              ┌────────────────┐
              │  Classifier    │
              │  (5 classes)   │
              └───────┬────────┘
                      │
                      ▼
              ┌────────────────┐
              │  Temperature   │
              │   Scaling      │
              └───────┬────────┘
                      │
                      ▼
         ┌────────────┴────────────┐
         │                         │
         ▼                         ▼
┌─────────────────┐       ┌─────────────────┐
│   Prediction    │       │  Grad-CAM       │
│   + Confidence  │       │  Explanation    │
└─────────────────┘       └─────────────────┘`}
                    </pre>
                </div>
            </section>

            {/* Class Information */}
            <section className="bg-white rounded-xl border border-gray-200 p-6">
                <h2 className="text-lg font-semibold text-gray-900 mb-4">Arrhythmia Classes</h2>
                <p className="text-gray-600 mb-4">
                    The model classifies ECG beats into 5 categories following the AAMI EC57 standard:
                </p>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
                    {[
                        { name: 'Normal (N)', desc: 'Normal sinus rhythm beats', color: 'bg-emerald-100 border-emerald-300 text-emerald-800' },
                        { name: 'Supraventricular (S)', desc: 'Atrial premature, nodal escape, etc.', color: 'bg-blue-100 border-blue-300 text-blue-800' },
                        { name: 'Ventricular (V)', desc: 'Ventricular premature, escape beats', color: 'bg-red-100 border-red-300 text-red-800' },
                        { name: 'Fusion (F)', desc: 'Fusion of ventricular and normal', color: 'bg-purple-100 border-purple-300 text-purple-800' },
                        { name: 'Unknown (Q)', desc: 'Paced, unclassifiable beats', color: 'bg-gray-100 border-gray-300 text-gray-800' },
                    ].map((cls) => (
                        <div key={cls.name} className={`p-4 rounded-lg border ${cls.color}`}>
                            <p className="font-semibold">{cls.name}</p>
                            <p className="text-sm mt-1 opacity-80">{cls.desc}</p>
                        </div>
                    ))}
                </div>
            </section>

            {/* Technical Specifications */}
            <section className="bg-white rounded-xl border border-gray-200 p-6">
                <h2 className="text-lg font-semibold text-gray-900 mb-4">Technical Specifications</h2>
                {isLoading ? (
                    <div className="animate-pulse space-y-3">
                        <div className="h-4 bg-gray-200 rounded w-3/4"></div>
                        <div className="h-4 bg-gray-200 rounded w-1/2"></div>
                        <div className="h-4 bg-gray-200 rounded w-2/3"></div>
                    </div>
                ) : (
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        <div>
                            <h3 className="font-medium text-gray-800 mb-3">Signal Processing</h3>
                            <table className="w-full text-sm">
                                <tbody className="divide-y divide-gray-100">
                                    <tr>
                                        <td className="py-2 text-gray-500">Sampling Rate</td>
                                        <td className="py-2 font-medium text-gray-900">{modelInfo?.sampling_rate || 125} Hz</td>
                                    </tr>
                                    <tr>
                                        <td className="py-2 text-gray-500">Beat Length</td>
                                        <td className="py-2 font-medium text-gray-900">{modelInfo?.beat_length || 187} samples</td>
                                    </tr>
                                    <tr>
                                        <td className="py-2 text-gray-500">Duration</td>
                                        <td className="py-2 font-medium text-gray-900">~1.5 seconds</td>
                                    </tr>
                                </tbody>
                            </table>
                        </div>
                        <div>
                            <h3 className="font-medium text-gray-800 mb-3">Model Configuration</h3>
                            <table className="w-full text-sm">
                                <tbody className="divide-y divide-gray-100">
                                    <tr>
                                        <td className="py-2 text-gray-500">CNN Filters</td>
                                        <td className="py-2 font-medium text-gray-900">
                                            {modelInfo?.features?.cnn_filters?.join(' → ') || '32 → 64 → 128 → 256'}
                                        </td>
                                    </tr>
                                    <tr>
                                        <td className="py-2 text-gray-500">Engineered Features</td>
                                        <td className="py-2 font-medium text-gray-900">{modelInfo?.features?.num_engineered_features || 25}</td>
                                    </tr>
                                    <tr>
                                        <td className="py-2 text-gray-500">Referral Threshold</td>
                                        <td className="py-2 font-medium text-gray-900">{((modelInfo?.referral_threshold || 0.7) * 100).toFixed(0)}%</td>
                                    </tr>
                                </tbody>
                            </table>
                        </div>
                    </div>
                )}
            </section>

            {/* Engineered Features */}
            <section className="bg-white rounded-xl border border-gray-200 p-6">
                <h2 className="text-lg font-semibold text-gray-900 mb-4">Engineered Features (25 total)</h2>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    <FeatureGroup
                        title="Morphological (7)"
                        features={['R-peak amplitude', 'Q/S amplitudes', 'QRS duration', 'R-wave slopes', 'Beat area']}
                    />
                    <FeatureGroup
                        title="Statistical (6)"
                        features={['Mean', 'Std deviation', 'Skewness', 'Kurtosis', 'Max', 'Min']}
                    />
                    <FeatureGroup
                        title="Frequency Domain (4)"
                        features={['Dominant frequency', 'Spectral centroid', 'Spectral spread', 'Total power']}
                    />
                    <FeatureGroup
                        title="Wavelet (6)"
                        features={['Energy at 4 levels', 'Entropy', 'Max coefficient']}
                    />
                    <FeatureGroup
                        title="Nonlinear (2)"
                        features={['Sample entropy approximation', 'Zero crossings']}
                    />
                </div>
            </section>

            {/* Referral Mechanism */}
            <section className="bg-white rounded-xl border border-gray-200 p-6">
                <h2 className="text-lg font-semibold text-gray-900 mb-4">Confidence-Based Referral</h2>
                <p className="text-gray-600 mb-4">
                    The system uses calibrated confidence scores to determine the appropriate action:
                </p>
                <div className="space-y-3">
                    <div className="flex items-center p-4 bg-emerald-50 border border-emerald-200 rounded-lg">
                        <span className="text-2xl mr-4">✅</span>
                        <div>
                            <p className="font-semibold text-emerald-800">Auto-Classified (≥85% confidence)</p>
                            <p className="text-sm text-emerald-700">High confidence prediction, suitable for automated processing</p>
                        </div>
                    </div>
                    <div className="flex items-center p-4 bg-amber-50 border border-amber-200 rounded-lg">
                        <span className="text-2xl mr-4">⚠️</span>
                        <div>
                            <p className="font-semibold text-amber-800">Monitor (70-85% confidence)</p>
                            <p className="text-sm text-amber-700">Moderate confidence, recommend additional review</p>
                        </div>
                    </div>
                    <div className="flex items-center p-4 bg-red-50 border border-red-200 rounded-lg">
                        <span className="text-2xl mr-4">❌</span>
                        <div>
                            <p className="font-semibold text-red-800">Refer to Cardiologist (&lt;70% confidence)</p>
                            <p className="text-sm text-red-700">Low confidence or ambiguous prediction, requires expert review</p>
                        </div>
                    </div>
                </div>
            </section>

            {/* Explainability */}
            <section className="bg-white rounded-xl border border-gray-200 p-6">
                <h2 className="text-lg font-semibold text-gray-900 mb-4">Explainability (Grad-CAM)</h2>
                <p className="text-gray-600 mb-4">
                    Every prediction includes a Grad-CAM visualization showing which parts of the ECG
                    waveform were most important for the model's decision. This helps clinicians:
                </p>
                <ul className="list-disc list-inside text-gray-600 space-y-2">
                    <li>Understand the model's reasoning</li>
                    <li>Validate that the model focuses on clinically relevant features</li>
                    <li>Identify potential edge cases or anomalies</li>
                    <li>Build trust in the AI-assisted diagnosis</li>
                </ul>
            </section>

            {/* Performance Metrics */}
            <section className="bg-white rounded-xl border border-gray-200 p-6">
                <h2 className="text-lg font-semibold text-gray-900 mb-4">Expected Performance</h2>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <MetricCard label="Overall Accuracy" value="~97%" />
                    <MetricCard label="Weighted F1" value="~0.95" />
                    <MetricCard label="Normal (N) F1" value="~0.99" />
                    <MetricCard label="Ventricular (V) F1" value="~0.95" />
                </div>
                <p className="text-sm text-gray-500 mt-4">
                    * Performance measured on patient-wise split test set to prevent data leakage
                </p>
            </section>

            {/* Disclaimer */}
            <section className="bg-amber-50 border border-amber-200 rounded-xl p-6">
                <h2 className="text-lg font-semibold text-amber-900 mb-3">⚠️ Important Disclaimer</h2>
                <div className="text-amber-800 space-y-2 text-sm">
                    <p>
                        This system is designed for <strong>research and educational purposes</strong>.
                        It is NOT a certified medical device and should NOT be used for actual clinical diagnosis.
                    </p>
                    <p>
                        Always consult a qualified healthcare professional for medical advice and diagnosis.
                        The predictions made by this system should be used as a supplementary tool only.
                    </p>
                </div>
            </section>

            {/* Citation */}
            <section className="bg-gray-50 rounded-xl p-6">
                <h2 className="text-lg font-semibold text-gray-900 mb-3">Citation</h2>
                <p className="text-gray-600 text-sm mb-2">If using the MIT-BIH dataset:</p>
                <div className="bg-white p-4 rounded-lg border border-gray-200 font-mono text-xs text-gray-700">
                    Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database.
                    IEEE Eng in Med and Biol 20(3):45-50 (May-June 2001).
                </div>
            </section>
        </div>
    );
}

/**
 * Feature Group Component
 */
function FeatureGroup({ title, features }) {
    return (
        <div className="bg-gray-50 rounded-lg p-4">
            <h4 className="font-medium text-gray-800 mb-2">{title}</h4>
            <ul className="text-sm text-gray-600 space-y-1">
                {features.map((f, i) => (
                    <li key={i}>• {f}</li>
                ))}
            </ul>
        </div>
    );
}

/**
 * Metric Card Component
 */
function MetricCard({ label, value }) {
    return (
        <div className="bg-gray-50 rounded-lg p-4 text-center">
            <p className="text-2xl font-bold text-blue-600">{value}</p>
            <p className="text-sm text-gray-600 mt-1">{label}</p>
        </div>
    );
}
