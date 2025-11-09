from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import wave
import os
import tempfile

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)

# Global classifier variable
classifier = None

def load_classifier():
    """Lazy load the classifier on first use"""
    global classifier
    if classifier is None:
        print("Loading emotion classifier...")
        from speechbrain.inference.classifiers import EncoderClassifier
        classifier = EncoderClassifier.from_hparams(
            source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
            savedir="tmpdir_emotion"
        )
        print("Classifier loaded!")
    return classifier
import torchaudio
import torch
def emotion_recognition(file_path):
    """Classify emotion from audio file."""
    try:
        clf = load_classifier()

        # Load any supported audio (wav, mp3, ogg, etc.)
        signal, fs = torchaudio.load(file_path)

        # Convert to mono if needed
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)

        # Resample to 16 kHz if required by model
        if fs != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=fs, new_freq=16000)
            signal = resampler(signal)
            fs = 16000

        # Classify directly from tensor (no need for wave.open)
        out_prob, score, index, text_lab = clf.classify_batch(signal)
        return text_lab[0], float(score[0])

    except Exception as e:
        raise Exception(f"Error processing audio: {str(e)}")

# Serve the React app
@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    if os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "model_loaded": classifier is not None}), 200

@app.route('/api/classify', methods=['POST'])
def classify_audio():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    
    audio_file = request.files['audio']
    
    if audio_file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
        temp_path = temp_file.name
        audio_file.save(temp_path)
    
    try:
        emotion, confidence = emotion_recognition(temp_path)
        return jsonify({
            "emotion": emotion,
            "confidence": confidence
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.unlink(temp_path)

if __name__ == '__main__':
    # Railway sets PORT environment variable
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)