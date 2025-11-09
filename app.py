from flask import Flask, render_template, request, jsonify
import os
import wave
import librosa
import soundfile as sf
from werkzeug.utils import secure_filename
from speechbrain.pretrained.interfaces import foreign_class

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'wav', 'mp3', 'm4a', 'ogg', 'flac'}

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the emotion classifier once at startup
print("Loading emotion recognition model...")
classifier = foreign_class(
    source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
    pymodule_file="custom_interface.py",
    classname="CustomEncoderWav2vec2Classifier"
)
print("Model loaded successfully!")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_audio(input_path, output_path):
    """Convert any audio format to 16kHz mono WAV"""
    try:
        y, sr = librosa.load(input_path, sr=16000, mono=True)
        sf.write(output_path, y, sr)
        return True
    except Exception as e:
        print(f"Error preprocessing audio: {e}")
        return False

def emotion_recognition(file_path):
    """Analyze emotion from audio file"""
    try:
        with wave.open(file_path, 'rb') as wav_file:
            frame_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            frame_count = wav_file.getnframes()
            segment_length = 10 * frame_rate

            # Process first 10 seconds
            temp_file = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_segment.wav')
            
            with wave.open(temp_file, 'wb') as new_wav_file:
                new_wav_file.setframerate(frame_rate)
                new_wav_file.setnchannels(channels)
                new_wav_file.setsampwidth(sample_width)
                segment = wav_file.readframes(min(segment_length, frame_count))
                new_wav_file.writeframes(segment)

            out_prob, score, index, text_lab = classifier.classify_file(temp_file)
            
            # Clean up temp file
            if os.path.exists(temp_file):
                os.remove(temp_file)
            
            return text_lab[0], float(score[0])
    except Exception as e:
        print(f"Error in emotion recognition: {e}")
        return None, None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        file = request.files['audio']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file format'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(input_path)
        
        # Preprocess to 16kHz mono WAV
        processed_path = os.path.join(app.config['UPLOAD_FOLDER'], 'processed.wav')
        if not preprocess_audio(input_path, processed_path):
            return jsonify({'error': 'Failed to process audio file'}), 500
        
        # Analyze emotion
        emotion, confidence = emotion_recognition(processed_path)
        
        # Clean up files
        if os.path.exists(input_path):
            os.remove(input_path)
        if os.path.exists(processed_path):
            os.remove(processed_path)
        
        if emotion is None:
            return jsonify({'error': 'Failed to analyze emotion'}), 500
        
        return jsonify({
            'emotion': emotion,
            'confidence': confidence
        })
    
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)