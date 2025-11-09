from flask import Flask, render_template, request, jsonify
import os
import librosa
import soundfile as sf
import torch
from werkzeug.utils import secure_filename
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'wav', 'mp3', 'm4a', 'ogg', 'flac'}

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the emotion classifier once at startup
print("Loading emotion recognition model...")
model_name = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)
print("Model loaded successfully!")

# Emotion labels
emotion_labels = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_audio(input_path):
    """Load and preprocess audio to 16kHz mono"""
    try:
        speech, sr = librosa.load(input_path, sr=16000, mono=True)
        # Take first 10 seconds if longer
        max_length = 16000 * 10
        if len(speech) > max_length:
            speech = speech[:max_length]
        return speech
    except Exception as e:
        print(f"Error preprocessing audio: {e}")
        return None

def predict_emotion(audio_data):
    """Analyze emotion from audio data"""
    try:
        # Prepare input
        inputs = processor(audio_data, sampling_rate=16000, return_tensors="pt", padding=True)
        
        # Get prediction
        with torch.no_grad():
            logits = model(**inputs).logits
        
        # Get probabilities
        probs = torch.nn.functional.softmax(logits, dim=-1)
        predicted_id = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][predicted_id].item()
        
        emotion = emotion_labels[predicted_id]
        
        return emotion, confidence
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
        
        # Preprocess audio
        audio_data = preprocess_audio(input_path)
        
        # Clean up uploaded file
        if os.path.exists(input_path):
            os.remove(input_path)
        
        if audio_data is None:
            return jsonify({'error': 'Failed to process audio file'}), 500
        
        # Analyze emotion
        emotion, confidence = predict_emotion(audio_data)
        
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