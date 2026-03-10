'use client';

import { useState } from 'react';
import UploadECG from '@/components/UploadECG';
import ECGPlot from '@/components/ECGPlot';
import PredictionCard from '@/components/PredictionCard';
import GradCamOverlay from '@/components/GradCamOverlay';
import { predictWithExplanation, savePredictionToHistory } from '@/lib/api';

/**
 * Upload ECG Page
 * Main analysis interface with file upload, visualization, and prediction
 */
export default function UploadPage() {
    const [fileData, setFileData] = useState(null);
    const [selectedBeatIndex, setSelectedBeatIndex] = useState(0);
    const [isAnalyzing, setIsAnalyzing] = useState(false);
    const [prediction, setPrediction] = useState(null);
    const [showGradCam, setShowGradCam] = useState(true);
    const [error, setError] = useState(null);

    // Handle file loaded
    const handleFileLoaded = (data) => {
        setFileData(data);
        setSelectedBeatIndex(0);
        setPrediction(null);
        setError(null);
    };

    // Handle analyze button
    const handleAnalyze = async (data) => {
        setIsAnalyzing(true);
        setError(null);
        setPrediction(null);

        try {
            // Get the selected beat signal
            let signal = data.signals[selectedBeatIndex];

            // If signal has 188 values, the last one is the label - remove it
            if (signal.length === 188) {
                signal = signal.slice(0, 187);
            }

            // Validate signal length
            if (signal.length !== 187) {
                throw new Error(`Expected 187 samples, got ${signal.length}. Please ensure your ECG data is in MIT-BIH format.`);
            }

            // Call API with explanation
            const result = await predictWithExplanation(signal, true);

            // Save to history
            savePredictionToHistory(result);

            setPrediction(result);
        } catch (err) {
            console.error('Analysis failed:', err);
            setError(err.message || 'Analysis failed. Please check if the backend API is running.');
        } finally {
            setIsAnalyzing(false);
        }
    };

    // Get current beat signal
    const currentSignal = fileData?.signals?.[selectedBeatIndex]?.slice(0, 187) || null;

    return (
        <div className="space-y-8">
            {/* Page Header */}
            <header>
                <h1 className="text-2xl font-bold text-gray-900">Upload ECG</h1>
                <p className="text-gray-600 mt-1">
                    Upload your ECG data file for AI-assisted arrhythmia analysis
                </p>
            </header>

            {/* Upload Section */}
            <section className="bg-white rounded-xl border border-gray-200 p-6">
                <h2 className="text-lg font-semibold text-gray-900 mb-4">
                    Step 1: Upload ECG File
                </h2>
                <UploadECG
                    onFileLoaded={handleFileLoaded}
                    onAnalyze={handleAnalyze}
                    isAnalyzing={isAnalyzing}
                />
            </section>

            {/* Beat Selector (if multiple beats) */}
            {fileData && fileData.beatCount > 1 && (
                <section className="bg-white rounded-xl border border-gray-200 p-6">
                    <h2 className="text-lg font-semibold text-gray-900 mb-4">
                        Step 2: Select Beat to Analyze
                    </h2>
                    <div className="flex flex-wrap items-center gap-4">
                        <div className="flex-1">
                            <label className="block text-sm font-medium text-gray-700 mb-2">
                                Beat Index (0 to {fileData.beatCount - 1})
                            </label>
                            <input
                                type="range"
                                min="0"
                                max={fileData.beatCount - 1}
                                value={selectedBeatIndex}
                                onChange={(e) => {
                                    setSelectedBeatIndex(parseInt(e.target.value));
                                    setPrediction(null);
                                }}
                                className="w-full"
                            />
                        </div>
                        <div className="flex items-center gap-2">
                            <input
                                type="number"
                                min="0"
                                max={fileData.beatCount - 1}
                                value={selectedBeatIndex}
                                onChange={(e) => {
                                    const val = parseInt(e.target.value);
                                    if (val >= 0 && val < fileData.beatCount) {
                                        setSelectedBeatIndex(val);
                                        setPrediction(null);
                                    }
                                }}
                                className="w-20 px-3 py-2 border border-gray-300 rounded-lg text-center"
                            />
                            <span className="text-sm text-gray-500">
                                of {fileData.beatCount} beats
                            </span>
                        </div>
                    </div>
                </section>
            )}

            {/* ECG Visualization */}
            {currentSignal && (
                <section>
                    <h2 className="text-lg font-semibold text-gray-900 mb-4">
                        ECG Waveform Visualization
                    </h2>
                    <ECGPlot
                        signal={currentSignal}
                        samplingRate={fileData?.samplingRate || 125}
                        heatmap={prediction?.heatmap}
                        showHeatmap={showGradCam && prediction?.heatmap}
                        title={`Beat ${selectedBeatIndex + 1} of ${fileData?.beatCount || 1}`}
                        height={350}
                    />
                </section>
            )}

            {/* Error Display */}
            {error && (
                <div className="p-4 bg-red-50 border border-red-200 rounded-xl">
                    <div className="flex items-start">
                        <svg className="w-5 h-5 text-red-500 mt-0.5 mr-3 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                        <div>
                            <h4 className="text-red-700 font-medium">Analysis Error</h4>
                            <p className="text-red-600 text-sm mt-1">{error}</p>
                            <p className="text-red-500 text-sm mt-2">
                                Make sure the backend API is running: <code className="bg-red-100 px-1 rounded">uvicorn api:app --reload</code>
                            </p>
                        </div>
                    </div>
                </div>
            )}

            {/* Analysis Results */}
            {(prediction || isAnalyzing) && (
                <section className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    {/* Prediction Card */}
                    <div>
                        <h2 className="text-lg font-semibold text-gray-900 mb-4">
                            AI Prediction
                        </h2>
                        <PredictionCard prediction={prediction} isLoading={isAnalyzing} />
                    </div>

                    {/* Grad-CAM Explanation */}
                    <div>
                        <div className="flex items-center justify-between mb-4">
                            <h2 className="text-lg font-semibold text-gray-900">
                                Model Explanation
                            </h2>
                        </div>
                        <GradCamOverlay
                            visualizationBase64={prediction?.visualization_base64}
                            heatmap={prediction?.heatmap}
                            keyRegions={prediction?.key_regions}
                            isLoading={isAnalyzing}
                            onToggle={setShowGradCam}
                            showOverlay={showGradCam}
                        />
                    </div>
                </section>
            )}

            {/* Instructions */}
            <section className="bg-blue-50 border border-blue-200 rounded-xl p-6">
                <h3 className="text-lg font-semibold text-blue-900 mb-3">📋 Instructions</h3>
                <ol className="list-decimal list-inside space-y-2 text-blue-800 text-sm">
                    <li>Upload a CSV file containing ECG beat data (MIT-BIH format preferred)</li>
                    <li>Each row should contain 187 samples representing one heartbeat</li>
                    <li>If your file contains multiple beats, select which beat to analyze</li>
                    <li>Click "Analyze ECG" to get the AI prediction</li>
                    <li>Review the prediction, confidence score, and Grad-CAM explanation</li>
                </ol>
                <div className="mt-4 p-3 bg-blue-100 rounded-lg">
                    <p className="text-sm text-blue-700">
                        <strong>Sample Data:</strong> You can test with data from the
                        <a href="https://www.kaggle.com/datasets/shayanfazeli/heartbeat"
                            target="_blank"
                            rel="noopener noreferrer"
                            className="underline ml-1">
                            Kaggle ECG Heartbeat Dataset
                        </a>
                    </p>
                </div>
            </section>
        </div>
    );
}
