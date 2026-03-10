'use client';

import Link from 'next/link';
import { useEffect, useState } from 'react';
import { getPredictionHistory, healthCheck } from '@/lib/api';

/**
 * Dashboard Page
 * Landing page with system overview and quick actions
 */
export default function Dashboard() {
  const [stats, setStats] = useState({
    totalPredictions: 0,
    todayPredictions: 0,
    avgConfidence: 0,
    referralRate: 0,
  });
  const [apiStatus, setApiStatus] = useState({ checking: true, online: false });
  const [recentPredictions, setRecentPredictions] = useState([]);

  useEffect(() => {
    // Load history stats
    const history = getPredictionHistory();
    const today = new Date().toDateString();
    const todayPredictions = history.filter(
      (p) => new Date(p.timestamp).toDateString() === today
    );
    const avgConf = history.length > 0
      ? history.reduce((sum, p) => sum + p.confidence, 0) / history.length
      : 0;
    const referrals = history.filter((p) => p.needs_referral).length;

    setStats({
      totalPredictions: history.length,
      todayPredictions: todayPredictions.length,
      avgConfidence: avgConf,
      referralRate: history.length > 0 ? referrals / history.length : 0,
    });
    setRecentPredictions(history.slice(0, 5));

    // Check API status
    healthCheck()
      .then(() => setApiStatus({ checking: false, online: true }))
      .catch(() => setApiStatus({ checking: false, online: false }));
  }, []);

  return (
    <div className="space-y-8">
      {/* Hero Section */}
      <section className="bg-white rounded-2xl shadow-sm border border-gray-200 overflow-hidden">
        <div className="bg-gradient-to-r from-blue-600 to-blue-700 px-8 py-12">
          <div className="max-w-3xl">
            <h1 className="text-3xl md:text-4xl font-bold text-white mb-4">
              ECG Arrhythmia Detection System
            </h1>
            <p className="text-blue-100 text-lg mb-6">
              AI-assisted cardiac rhythm analysis powered by deep learning.
              Upload ECG data, visualize waveforms, and receive predictions with
              confidence-based decision support.
            </p>
            <div className="flex flex-wrap gap-4">
              <Link
                href="/upload"
                className="inline-flex items-center px-6 py-3 bg-white text-blue-600 font-semibold rounded-lg hover:bg-blue-50 transition-colors shadow-md"
              >
                <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                </svg>
                Upload ECG
              </Link>
              <Link
                href="/about"
                className="inline-flex items-center px-6 py-3 bg-blue-500 text-white font-semibold rounded-lg hover:bg-blue-400 transition-colors"
              >
                <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                Learn More
              </Link>
            </div>
          </div>
        </div>

        {/* API Status Banner */}
        <div className={`px-8 py-3 ${apiStatus.checking
            ? 'bg-gray-100'
            : apiStatus.online
              ? 'bg-emerald-50'
              : 'bg-amber-50'
          }`}>
          <div className="flex items-center">
            {apiStatus.checking ? (
              <>
                <div className="w-3 h-3 bg-gray-400 rounded-full animate-pulse mr-3"></div>
                <span className="text-sm text-gray-600">Checking backend status...</span>
              </>
            ) : apiStatus.online ? (
              <>
                <div className="w-3 h-3 bg-emerald-500 rounded-full mr-3"></div>
                <span className="text-sm text-emerald-700">Backend API is online and ready</span>
              </>
            ) : (
              <>
                <div className="w-3 h-3 bg-amber-500 rounded-full mr-3"></div>
                <span className="text-sm text-amber-700">
                  Backend API is offline. Start with: <code className="bg-amber-100 px-1 rounded">uvicorn api:app --reload</code>
                </span>
              </>
            )}
          </div>
        </div>
      </section>

      {/* Stats Grid */}
      <section className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          title="Total Predictions"
          value={stats.totalPredictions}
          icon="📊"
          color="blue"
        />
        <StatCard
          title="Today's Analyses"
          value={stats.todayPredictions}
          icon="📅"
          color="green"
        />
        <StatCard
          title="Avg. Confidence"
          value={`${(stats.avgConfidence * 100).toFixed(1)}%`}
          icon="🎯"
          color="purple"
        />
        <StatCard
          title="Referral Rate"
          value={`${(stats.referralRate * 100).toFixed(1)}%`}
          icon="⚠️"
          color="amber"
        />
      </section>

      {/* Features Grid */}
      <section>
        <h2 className="text-xl font-semibold text-gray-900 mb-4">System Capabilities</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <FeatureCard
            icon="🔬"
            title="5-Class AAMI Classification"
            description="Classifies ECG beats into Normal, Supraventricular, Ventricular, Fusion, and Unknown categories."
          />
          <FeatureCard
            icon="📈"
            title="Confidence-Based Decisions"
            description="High-confidence predictions are auto-classified; low-confidence cases are flagged for cardiologist review."
          />
          <FeatureCard
            icon="🧠"
            title="Explainable AI"
            description="Grad-CAM visualizations show which parts of the ECG influenced the model's prediction."
          />
        </div>
      </section>

      {/* Recent Predictions */}
      {recentPredictions.length > 0 && (
        <section>
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-semibold text-gray-900">Recent Predictions</h2>
            <Link
              href="/history"
              className="text-blue-600 hover:text-blue-800 text-sm font-medium"
            >
              View All →
            </Link>
          </div>
          <div className="bg-white rounded-xl border border-gray-200 divide-y divide-gray-100">
            {recentPredictions.map((pred, idx) => (
              <div key={pred.id || idx} className="px-6 py-4 flex items-center justify-between">
                <div>
                  <p className="font-medium text-gray-900">{pred.prediction_name}</p>
                  <p className="text-sm text-gray-500">
                    {new Date(pred.timestamp).toLocaleString()}
                  </p>
                </div>
                <div className="text-right">
                  <p className={`font-semibold ${pred.confidence >= 0.85
                      ? 'text-emerald-600'
                      : pred.confidence >= 0.7
                        ? 'text-amber-600'
                        : 'text-red-600'
                    }`}>
                    {(pred.confidence * 100).toFixed(1)}%
                  </p>
                  <p className="text-xs text-gray-500">Confidence</p>
                </div>
              </div>
            ))}
          </div>
        </section>
      )}
    </div>
  );
}

/**
 * Stat Card Component
 */
function StatCard({ title, value, icon, color }) {
  const colorClasses = {
    blue: 'bg-blue-50 border-blue-200',
    green: 'bg-emerald-50 border-emerald-200',
    purple: 'bg-purple-50 border-purple-200',
    amber: 'bg-amber-50 border-amber-200',
  };

  return (
    <div className={`p-6 rounded-xl border ${colorClasses[color]}`}>
      <div className="flex items-center justify-between mb-2">
        <span className="text-2xl">{icon}</span>
      </div>
      <p className="text-3xl font-bold text-gray-900">{value}</p>
      <p className="text-sm text-gray-600 mt-1">{title}</p>
    </div>
  );
}

/**
 * Feature Card Component
 */
function FeatureCard({ icon, title, description }) {
  return (
    <div className="bg-white p-6 rounded-xl border border-gray-200 hover:shadow-md transition-shadow">
      <span className="text-3xl mb-4 block">{icon}</span>
      <h3 className="text-lg font-semibold text-gray-900 mb-2">{title}</h3>
      <p className="text-gray-600 text-sm">{description}</p>
    </div>
  );
}
