from flask import Flask, request, jsonify
from flask_cors import CORS
import wave
import os
import tempfile
from speechbrain.pretrained.interfaces import foreign_class

app = Flask(__name__)
CORS(app)

# Initialize the classifier
print("Loading emotion classifier...")
classifier = foreign_class(
    source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
    pymodule_file="custom_interface.py",
    classname="CustomEncoderWav2vec2Classifier"
)
print("Classifier loaded!")

def emotion_recognition(file_path):
    """Classify emotion from audio file"""
    try:
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

                out_prob, score, index, text_lab = classifier.classify_file(temp_file_name)
                
                # Clean up temp file
                os.unlink(temp_file_name)
                
                return text_lab[0], float(score[0])
    except Exception as e:
        raise Exception(f"Error processing audio: {str(e)}")

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"}), 200

@app.route('/classify', methods=['POST'])
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
    port = 8082
    app.run(host='0.0.0.0', port=port, debug=False)