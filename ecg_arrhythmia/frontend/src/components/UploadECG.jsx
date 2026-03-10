'use client';

import { useState, useCallback } from 'react';

/**
 * UploadECG Component
 * Drag-and-drop interface for uploading ECG CSV files
 */
export default function UploadECG({ onFileLoaded, onAnalyze, isAnalyzing = false }) {
    const [dragActive, setDragActive] = useState(false);
    const [file, setFile] = useState(null);
    const [fileInfo, setFileInfo] = useState(null);
    const [error, setError] = useState(null);
    const [isLoading, setIsLoading] = useState(false);

    // Handle drag events
    const handleDrag = useCallback((e) => {
        e.preventDefault();
        e.stopPropagation();
        if (e.type === 'dragenter' || e.type === 'dragover') {
            setDragActive(true);
        } else if (e.type === 'dragleave') {
            setDragActive(false);
        }
    }, []);

    // Validate and parse CSV file
    const parseCSVFile = async (file) => {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();

            reader.onload = (event) => {
                try {
                    const text = event.target.result;
                    const lines = text.trim().split('\n');
                    const signals = [];

                    for (const line of lines) {
                        // Skip empty lines
                        if (!line.trim()) continue;

                        const values = line.split(',').map(v => parseFloat(v.trim()));
                        // Validate numeric values
                        if (values.length > 0 && !values.some(isNaN)) {
                            signals.push(values);
                        }
                    }

                    if (signals.length === 0) {
                        reject(new Error('No valid ECG data found in file'));
                        return;
                    }

                    const signalLength = signals[0].length;

                    // Validate signal length (should be 187 for MIT-BIH or 188 with label)
                    if (signalLength < 100) {
                        reject(new Error(`Signal too short: ${signalLength} samples. Expected at least 100 samples.`));
                        return;
                    }

                    resolve({
                        signals,
                        samplingRate: 125,
                        signalLength,
                        beatCount: signals.length,
                    });
                } catch (error) {
                    reject(new Error('Failed to parse CSV: ' + error.message));
                }
            };

            reader.onerror = () => reject(new Error('Failed to read file'));
            reader.readAsText(file);
        });
    };

    // Handle file drop
    const handleDrop = useCallback(async (e) => {
        e.preventDefault();
        e.stopPropagation();
        setDragActive(false);
        setError(null);

        const droppedFile = e.dataTransfer?.files?.[0];
        if (!droppedFile) return;

        await processFile(droppedFile);
    }, []);

    // Handle file input change
    const handleFileChange = async (e) => {
        const selectedFile = e.target.files?.[0];
        if (!selectedFile) return;

        await processFile(selectedFile);
    };

    // Process the uploaded file
    const processFile = async (uploadedFile) => {
        setError(null);
        setIsLoading(true);

        // Validate file type
        if (!uploadedFile.name.endsWith('.csv')) {
            setError('Please upload a CSV file');
            setIsLoading(false);
            return;
        }

        try {
            const parsed = await parseCSVFile(uploadedFile);
            setFile(uploadedFile);
            setFileInfo(parsed);

            if (onFileLoaded) {
                onFileLoaded(parsed);
            }
        } catch (err) {
            setError(err.message);
            setFile(null);
            setFileInfo(null);
        } finally {
            setIsLoading(false);
        }
    };

    // Handle analyze button click
    const handleAnalyze = () => {
        if (fileInfo && onAnalyze) {
            onAnalyze(fileInfo);
        }
    };

    // Clear the uploaded file
    const handleClear = () => {
        setFile(null);
        setFileInfo(null);
        setError(null);
    };

    return (
        <div className="w-full">
            {/* Drop Zone */}
            <div
                className={`relative border-2 border-dashed rounded-xl p-8 transition-all duration-200 ${dragActive
                        ? 'border-blue-500 bg-blue-50'
                        : error
                            ? 'border-red-300 bg-red-50'
                            : file
                                ? 'border-emerald-300 bg-emerald-50'
                                : 'border-gray-300 bg-gray-50 hover:border-blue-400 hover:bg-blue-50'
                    }`}
                onDragEnter={handleDrag}
                onDragLeave={handleDrag}
                onDragOver={handleDrag}
                onDrop={handleDrop}
            >
                <input
                    type="file"
                    accept=".csv"
                    onChange={handleFileChange}
                    className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                    disabled={isLoading || isAnalyzing}
                />

                <div className="text-center">
                    {isLoading ? (
                        <div className="flex flex-col items-center">
                            <div className="w-12 h-12 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mb-4"></div>
                            <p className="text-gray-600 font-medium">Processing file...</p>
                        </div>
                    ) : file ? (
                        <div className="flex flex-col items-center">
                            <div className="w-16 h-16 bg-emerald-100 rounded-full flex items-center justify-center mb-4">
                                <svg className="w-8 h-8 text-emerald-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                                </svg>
                            </div>
                            <p className="text-emerald-700 font-semibold text-lg mb-1">{file.name}</p>
                            <p className="text-gray-500 text-sm">File loaded successfully</p>
                        </div>
                    ) : (
                        <div className="flex flex-col items-center">
                            <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mb-4">
                                <svg className="w-8 h-8 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                                </svg>
                            </div>
                            <p className="text-gray-700 font-semibold text-lg mb-1">Drop ECG file here</p>
                            <p className="text-gray-500 text-sm">or click to browse</p>
                            <p className="text-gray-400 text-xs mt-2">Supports CSV files (MIT-BIH format)</p>
                        </div>
                    )}
                </div>
            </div>

            {/* Error Message */}
            {error && (
                <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg">
                    <div className="flex items-start">
                        <svg className="w-5 h-5 text-red-500 mt-0.5 mr-3 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                        <div>
                            <p className="text-red-700 font-medium">Upload Error</p>
                            <p className="text-red-600 text-sm mt-1">{error}</p>
                        </div>
                    </div>
                </div>
            )}

            {/* File Info */}
            {fileInfo && (
                <div className="mt-4 p-4 bg-white border border-gray-200 rounded-lg">
                    <h4 className="text-sm font-semibold text-gray-700 mb-3">File Information</h4>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                        <div className="bg-gray-50 rounded-lg p-3">
                            <p className="text-xs text-gray-500 uppercase tracking-wide">Beats</p>
                            <p className="text-lg font-semibold text-gray-900">{fileInfo.beatCount}</p>
                        </div>
                        <div className="bg-gray-50 rounded-lg p-3">
                            <p className="text-xs text-gray-500 uppercase tracking-wide">Samples/Beat</p>
                            <p className="text-lg font-semibold text-gray-900">{fileInfo.signalLength}</p>
                        </div>
                        <div className="bg-gray-50 rounded-lg p-3">
                            <p className="text-xs text-gray-500 uppercase tracking-wide">Sampling Rate</p>
                            <p className="text-lg font-semibold text-gray-900">{fileInfo.samplingRate} Hz</p>
                        </div>
                        <div className="bg-gray-50 rounded-lg p-3">
                            <p className="text-xs text-gray-500 uppercase tracking-wide">Duration/Beat</p>
                            <p className="text-lg font-semibold text-gray-900">
                                {((fileInfo.signalLength / fileInfo.samplingRate) * 1000).toFixed(0)} ms
                            </p>
                        </div>
                    </div>
                </div>
            )}

            {/* Action Buttons */}
            {fileInfo && (
                <div className="mt-4 flex flex-col sm:flex-row gap-3">
                    <button
                        onClick={handleAnalyze}
                        disabled={isAnalyzing}
                        className={`flex-1 py-3 px-6 rounded-lg font-semibold text-white transition-all duration-200 ${isAnalyzing
                                ? 'bg-blue-400 cursor-not-allowed'
                                : 'bg-blue-600 hover:bg-blue-700 active:bg-blue-800 shadow-md hover:shadow-lg'
                            }`}
                    >
                        {isAnalyzing ? (
                            <span className="flex items-center justify-center">
                                <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" fill="none" viewBox="0 0 24 24">
                                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                </svg>
                                Analyzing...
                            </span>
                        ) : (
                            <span className="flex items-center justify-center">
                                <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4" />
                                </svg>
                                Analyze ECG
                            </span>
                        )}
                    </button>
                    <button
                        onClick={handleClear}
                        disabled={isAnalyzing}
                        className="py-3 px-6 rounded-lg font-medium text-gray-700 bg-gray-100 hover:bg-gray-200 transition-colors duration-200"
                    >
                        Clear
                    </button>
                </div>
            )}
        </div>
    );
}
