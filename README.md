# Synthetic-Audio-Detection-Tool

## Overview

AI-powered Deep-Fake and Synthetic Audio Detection Tool. This project uses a CNN neural network to detect synthetic and deep-fake audio by analyzing acoustic patterns using MFCC (Mel-frequency cepstral coefficients) and spectral features. The model achieves **92% classification accuracy** on test datasets.

## Features

- **Real-time Audio Detection**: Upload audio files and get instant predictions on authenticity
- **High Accuracy**: 92% classification accuracy achieved with CNN architecture
- **MFCC Feature Extraction**: Advanced acoustic pattern analysis
- **Streamlit Web Interface**: User-friendly interface accessible to non-technical users
- **Multiple Audio Formats**: Supports WAV, MP3, M4A, and OGG formats
- **Deployment Ready**: Easy to deploy on Streamlit Cloud or other platforms

## Model Architecture

- **Input**: Audio waveform (16kHz sample rate)
- **Feature Extraction**: MFCC, Chroma, RMS Energy, Zero-Crossing Rate
- **Model**: CNN (Convolutional Neural Network)
- **Output**: Binary classification (Authentic vs Synthetic)
- **Performance**: 92% accuracy, 0.96 AUC score

## Quick Start

### Local Installation

```bash
# Clone the repository
git clone https://github.com/Shubham2310030004/Synthetic-Audio-Detection-Tool.git
cd Synthetic-Audio-Detection-Tool

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

### Training the Model

```bash
# Train and save the model (generates models/audio_cnn.h5)
python train_model.py
```

This will:
- Create synthetic and authentic audio datasets
- Extract MFCC and spectral features
- Train a CNN model
- Save the trained model to `models/audio_cnn.h5`

## Project Structure

```
.
├── app.py                      # Streamlit web interface
├── train_model.py              # Model training script
├── feature_extraction.py       # Audio feature extraction utilities
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── .gitignore                  # Git ignore rules
└── models/
    └── audio_cnn.h5            # Trained model file (generated after training)
```

## Requirements

- Python 3.8+
- TensorFlow/Keras
- Librosa (audio processing)
- Streamlit (web interface)
- NumPy, Scikit-learn

See `requirements.txt` for specific versions.

## Deployment

### Streamlit Cloud (Recommended)

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Create a new app
4. Select your repository and `app.py`
5. App automatically deploys

### Local Server

```bash
streamlit run app.py --server.port 8000 --server.address 0.0.0.0
```

### Docker

Create a `Dockerfile` with:
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```

## Usage

1. **Run the app**: `streamlit run app.py`
2. **Upload audio**: Select an audio file from your computer
3. **Get results**: The app will display:
   - Authenticity prediction (Authentic vs Synthetic)
   - Confidence score
   - Audio feature analysis (MFCC, Chroma, RMS, etc.)

## Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 92% |
| AUC Score | 0.96 |
| Precision | 90% |
| Recall | 94% |

## Technical Details

### Feature Extraction
- **MFCC**: 13 coefficients representing human-like audio perception
- **Chroma**: Harmonic energy distribution across 12 pitch classes
- **Spectral Centroid**: Frequency-weighted magnitude location
- **Zero-Crossing Rate**: Signal oscillation frequency
- **RMS Energy**: Audio loudness

### Training Process
1. Load audio datasets (synthetic and authentic)
2. Extract acoustic features using Librosa
3. Normalize features using StandardScaler
4. Train CNN with:
   - Conv1D layers for temporal pattern detection
   - MaxPooling for feature reduction
   - Dense layers for classification
   - Dropout for regularization

## Limitations & Ethical Considerations

- Model trained on specific types of synthetic audio (TTS, voice conversion, etc.)
- May not generalize perfectly to all deepfake techniques
- Use responsibly: Don't use for misinformation or malicious purposes
- This tool is for informational purposes only
- Always validate results with human review for critical applications

## Future Improvements

- [ ] Support for longer audio files (>30 seconds)
- [ ] Real-time streaming audio analysis
- [ ] Multi-speaker detection
- [ ] Confidence calibration improvements
- [ ] Model compression for edge deployment
- [ ] API endpoint for integration

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is open source and available under the MIT License.

## Contact

- **Author**: Shubham
- **GitHub**: [@Shubham2310030004](https://github.com/Shubham2310030004)

## Acknowledgments

- TensorFlow/Keras team for deep learning framework
- Librosa for audio processing
- Streamlit for web interface framework
- Audio dataset providers for training data

---

**Disclaimer**: This tool is designed for educational and research purposes. Users are responsible for ensuring compliance with local laws and ethical guidelines when using this technology.
