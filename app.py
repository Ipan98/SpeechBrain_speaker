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

def emotion_recognition(file_path):
    """Classify emotion from audio file"""
    try:
        clf = load_classifier()
        with wave.open(file_path, 'rb') as wav_file:
            frame_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            frame_count = wav_file.getnframes()
            segment_length = 10 * frame_rate

            # Process first segment (or entire file if shorter)
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_file_name = temp_file.name
                
                with wave.open(temp_file_name, 'wb') as new_wav_file:
                    new_wav_file.setframerate(frame_rate)
                    new_wav_file.setnchannels(channels)
                    new_wav_file.setsampwidth(sample_width)
                    segment = wav_file.readframes(min(segment_length, frame_count))
                    new_wav_file.writeframes(segment)

                out_prob, score, index, text_lab = clf.classify_file(temp_file_name)
                
                # Clean up temp file
                os.unlink(temp_file_name)
                
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