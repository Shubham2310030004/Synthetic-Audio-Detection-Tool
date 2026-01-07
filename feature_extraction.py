"""Audio Feature Extraction Module

Extracts MFCC (Mel-frequency cepstral coefficients) and spectral features
from audio signals for synthetic audio detection.
"""

import numpy as np
import librosa
from scipy import signal
from scipy.fftpack import fft
import warnings
warnings.filterwarnings('ignore')

class AudioFeatureExtractor:
    """Extracts audio features for deep-fake detection."""
    
    def __init__(self, sr=16000, n_mfcc=13):
        self.sr = sr
        self.n_mfcc = n_mfcc
    
    def extract_mfcc(self, audio, n_mfcc=None):
        """Extract MFCC features from audio."""
        if n_mfcc is None:
            n_mfcc = self.n_mfcc
        mfcc = librosa.feature.mfcc(y=audio, sr=self.sr, n_mfcc=n_mfcc)
        return mfcc
    
    def extract_spectral_features(self, audio):
        """Extract spectral features from audio."""
        # Spectral centroid
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sr)
        # Spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sr)
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio)
        return spectral_centroid, spectral_rolloff, zcr
    
    def extract_chroma_features(self, audio):
        """Extract chroma features."""
        chroma = librosa.feature.chroma_stft(y=audio, sr=self.sr)
        return chroma
    
    def extract_temporal_features(self, audio):
        """Extract temporal features."""
        rms = librosa.feature.rms(y=audio)
        return rms
    
    def extract_all_features(self, audio):
        """Extract all audio features."""
        mfcc = self.extract_mfcc(audio)
        spec_cent, spec_roll, zcr = self.extract_spectral_features(audio)
        chroma = self.extract_chroma_features(audio)
        rms = self.extract_temporal_features(audio)
        
        features = {
            'mfcc': mfcc,
            'spectral_centroid': spec_cent,
            'spectral_rolloff': spec_roll,
            'zero_crossing_rate': zcr,
            'chroma': chroma,
            'rms': rms
        }
        return features
    
    def normalize_features(self, features):
        """Normalize feature vectors."""
        normalized = {}
        for key, feat in features.items():
            if feat is not None:
                mean = np.mean(feat, axis=1, keepdims=True)
                std = np.std(feat, axis=1, keepdims=True) + 1e-8
                normalized[key] = (feat - mean) / std
        return normalized
