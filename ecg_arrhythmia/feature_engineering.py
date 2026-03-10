"""
Feature Engineering for ECG Arrhythmia Detection
Extracts HRV and morphological features from ECG beats
"""
import numpy as np
from scipy import signal, stats
from scipy.fft import fft
import pywt
from typing import Dict, List, Tuple, Optional
import warnings

import config


class ECGFeatureExtractor:
    """
    Extract engineered features from ECG beat waveforms.
    
    Features include:
    1. Morphological features (amplitude, intervals, slopes)
    2. Statistical features (mean, std, skewness, kurtosis)
    3. Frequency domain features (power spectral density)
    4. Wavelet features (multi-resolution analysis)
    5. HRV-like features (for beat-level analysis)
    """
    
    def __init__(self, sampling_rate: int = 125):
        """
        Args:
            sampling_rate: Sampling rate in Hz
        """
        self.fs = sampling_rate
        self.feature_names = []
        self._build_feature_names()
    
    def _build_feature_names(self):
        """Build list of feature names for interpretability."""
        self.feature_names = [
            # Morphological (7)
            'r_peak_amplitude',
            'q_amplitude',
            's_amplitude',
            'qrs_duration',
            'r_slope_up',
            'r_slope_down',
            'beat_area',
            
            # Statistical (6)
            'mean',
            'std',
            'skewness',
            'kurtosis',
            'max_val',
            'min_val',
            
            # Frequency domain (4)
            'dominant_freq',
            'spectral_centroid',
            'spectral_spread',
            'total_power',
            
            # Wavelet (6)
            'wavelet_energy_1',
            'wavelet_energy_2',
            'wavelet_energy_3',
            'wavelet_energy_4',
            'wavelet_entropy',
            'wavelet_max_coef',
            
            # Nonlinear (2)
            'sample_entropy_approx',
            'zero_crossings',
        ]
    
    def extract_morphological_features(self, beat: np.ndarray) -> Dict[str, float]:
        """
        Extract morphological features from a single beat.
        
        Args:
            beat: 1D array of beat samples
        
        Returns:
            Dictionary of morphological features
        """
        features = {}
        
        # Find R-peak (assumed to be near center and highest point)
        center = len(beat) // 2
        search_window = len(beat) // 4
        r_region = beat[center - search_window:center + search_window]
        r_idx_local = np.argmax(np.abs(r_region))
        r_idx = center - search_window + r_idx_local
        
        # R-peak amplitude
        features['r_peak_amplitude'] = beat[r_idx]
        
        # Q and S waves (local minima around R)
        q_region = beat[max(0, r_idx - 20):r_idx]
        s_region = beat[r_idx:min(len(beat), r_idx + 20)]
        
        features['q_amplitude'] = np.min(q_region) if len(q_region) > 0 else 0
        features['s_amplitude'] = np.min(s_region) if len(s_region) > 0 else 0
        
        # QRS duration (approximate)
        threshold = 0.3 * features['r_peak_amplitude']
        above_thresh = np.where(np.abs(beat) > abs(threshold))[0]
        if len(above_thresh) > 1:
            features['qrs_duration'] = (above_thresh[-1] - above_thresh[0]) / self.fs
        else:
            features['qrs_duration'] = 0
        
        # Slopes
        if r_idx > 0:
            features['r_slope_up'] = (beat[r_idx] - beat[max(0, r_idx - 5)]) / 5
        else:
            features['r_slope_up'] = 0
            
        if r_idx < len(beat) - 1:
            features['r_slope_down'] = (beat[min(len(beat)-1, r_idx + 5)] - beat[r_idx]) / 5
        else:
            features['r_slope_down'] = 0
        
        # Beat area (absolute integral) - use trapezoid for NumPy 2.x compatibility
        try:
            features['beat_area'] = np.trapezoid(np.abs(beat))
        except AttributeError:
            features['beat_area'] = np.trapz(np.abs(beat))
        
        return features
    
    def extract_statistical_features(self, beat: np.ndarray) -> Dict[str, float]:
        """
        Extract statistical features from a beat.
        """
        features = {}
        
        features['mean'] = np.mean(beat)
        features['std'] = np.std(beat)
        features['skewness'] = stats.skew(beat)
        features['kurtosis'] = stats.kurtosis(beat)
        features['max_val'] = np.max(beat)
        features['min_val'] = np.min(beat)
        
        return features
    
    def extract_frequency_features(self, beat: np.ndarray) -> Dict[str, float]:
        """
        Extract frequency domain features using FFT.
        """
        features = {}
        
        # Compute FFT
        n = len(beat)
        fft_vals = np.abs(fft(beat))[:n // 2]
        freqs = np.fft.fftfreq(n, 1/self.fs)[:n // 2]
        
        # Avoid division by zero
        total_power = np.sum(fft_vals**2)
        if total_power == 0:
            total_power = 1e-10
        
        # Dominant frequency
        features['dominant_freq'] = freqs[np.argmax(fft_vals)]
        
        # Spectral centroid
        features['spectral_centroid'] = np.sum(freqs * fft_vals**2) / total_power
        
        # Spectral spread
        features['spectral_spread'] = np.sqrt(
            np.sum((freqs - features['spectral_centroid'])**2 * fft_vals**2) / total_power
        )
        
        # Total power
        features['total_power'] = total_power
        
        return features
    
    def extract_wavelet_features(self, beat: np.ndarray) -> Dict[str, float]:
        """
        Extract wavelet-based features using discrete wavelet transform.
        """
        features = {}
        
        # Perform 4-level DWT using Daubechies wavelet
        try:
            coeffs = pywt.wavedec(beat, 'db4', level=4)
            
            # Energy at each level
            energies = []
            for i, c in enumerate(coeffs[1:5]):  # Detail coefficients
                energy = np.sum(c**2)
                features[f'wavelet_energy_{i+1}'] = energy
                energies.append(energy)
            
            # Wavelet entropy
            total_energy = sum(energies) + 1e-10
            probs = np.array(energies) / total_energy
            probs = probs[probs > 0]
            features['wavelet_entropy'] = -np.sum(probs * np.log2(probs + 1e-10))
            
            # Maximum coefficient
            all_coeffs = np.concatenate([c for c in coeffs])
            features['wavelet_max_coef'] = np.max(np.abs(all_coeffs))
            
        except Exception as e:
            # Fallback values
            for i in range(4):
                features[f'wavelet_energy_{i+1}'] = 0
            features['wavelet_entropy'] = 0
            features['wavelet_max_coef'] = 0
        
        return features
    
    def extract_nonlinear_features(self, beat: np.ndarray) -> Dict[str, float]:
        """
        Extract nonlinear/complexity features.
        """
        features = {}
        
        # Approximate sample entropy (simplified version)
        # Full implementation would use neurokit2.entropy_sample
        diff = np.diff(beat)
        features['sample_entropy_approx'] = np.std(diff) / (np.std(beat) + 1e-10)
        
        # Zero crossings
        zero_crossings = np.where(np.diff(np.signbit(beat)))[0]
        features['zero_crossings'] = len(zero_crossings)
        
        return features
    
    def extract_all_features(self, beat: np.ndarray) -> np.ndarray:
        """
        Extract all features from a single beat.
        
        Args:
            beat: 1D array of beat samples
        
        Returns:
            1D array of all features
        """
        all_features = {}
        
        all_features.update(self.extract_morphological_features(beat))
        all_features.update(self.extract_statistical_features(beat))
        all_features.update(self.extract_frequency_features(beat))
        all_features.update(self.extract_wavelet_features(beat))
        all_features.update(self.extract_nonlinear_features(beat))
        
        # Return as array in consistent order
        feature_vector = np.array([all_features[name] for name in self.feature_names])
        
        return feature_vector
    
    def extract_batch_features(self, beats: np.ndarray, verbose: bool = True) -> np.ndarray:
        """
        Extract features from multiple beats.
        
        Args:
            beats: 2D array (n_samples, beat_length)
            verbose: Print progress
        
        Returns:
            2D array (n_samples, n_features)
        """
        n_samples = len(beats)
        n_features = len(self.feature_names)
        
        features = np.zeros((n_samples, n_features), dtype=np.float32)
        
        for i, beat in enumerate(beats):
            if verbose and (i + 1) % 10000 == 0:
                print(f"Extracting features: {i+1}/{n_samples} ({100*(i+1)/n_samples:.1f}%)")
            
            features[i] = self.extract_all_features(beat)
        
        # Handle NaN/Inf values
        features = np.nan_to_num(features, nan=0, posinf=0, neginf=0)
        
        if verbose:
            print(f"Extracted {n_features} features from {n_samples} beats")
        
        return features
    
    def normalize_features(
        self,
        features: np.ndarray,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Z-score normalize features.
        
        Args:
            features: Feature array
            mean: Pre-computed mean (for val/test normalization)
            std: Pre-computed std
        
        Returns:
            normalized_features, mean, std
        """
        if mean is None:
            mean = np.mean(features, axis=0)
        if std is None:
            std = np.std(features, axis=0)
        
        # Avoid division by zero
        std = np.where(std == 0, 1, std)
        
        normalized = (features - mean) / std
        
        return normalized, mean, std


def extract_hrv_features_from_rr_intervals(rr_intervals: np.ndarray) -> Dict[str, float]:
    """
    Extract HRV features from RR intervals.
    
    Note: For beat-level classification, we don't have true RR intervals.
    This function is for when you have continuous recordings with R-peak detection.
    
    Args:
        rr_intervals: Array of RR intervals in milliseconds
    
    Returns:
        Dictionary of HRV features
    """
    features = {}
    
    if len(rr_intervals) < 2:
        return {
            'rmssd': 0, 'sdnn': 0, 'pnn50': 0,
            'mean_hr': 0, 'std_hr': 0
        }
    
    # Time domain features
    diff_rr = np.diff(rr_intervals)
    
    # RMSSD - Root Mean Square of Successive Differences
    features['rmssd'] = np.sqrt(np.mean(diff_rr**2))
    
    # SDNN - Standard Deviation of NN intervals
    features['sdnn'] = np.std(rr_intervals)
    
    # pNN50 - Percentage of successive differences > 50ms
    features['pnn50'] = 100 * np.sum(np.abs(diff_rr) > 50) / len(diff_rr)
    
    # Heart rate statistics
    hr = 60000 / rr_intervals  # Convert to BPM
    features['mean_hr'] = np.mean(hr)
    features['std_hr'] = np.std(hr)
    
    return features


if __name__ == "__main__":
    # Test feature extraction
    print("Testing feature extraction...")
    
    # Create synthetic beat
    t = np.linspace(0, 1, 187)
    synthetic_beat = 0.5 * np.sin(2 * np.pi * 5 * t) + 0.3 * np.exp(-((t - 0.5)**2) / 0.01)
    
    extractor = ECGFeatureExtractor(sampling_rate=125)
    
    print(f"\nNumber of features: {len(extractor.feature_names)}")
    print(f"Feature names: {extractor.feature_names}")
    
    features = extractor.extract_all_features(synthetic_beat)
    
    print(f"\nExtracted features:")
    for name, value in zip(extractor.feature_names, features):
        print(f"  {name}: {value:.4f}")
