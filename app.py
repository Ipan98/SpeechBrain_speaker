import io
import os
import tempfile
from typing import Dict

import torch
import torchaudio
from flask import Flask, jsonify, render_template, request
from speechbrain.pretrained import EncoderClassifier

app = Flask(__name__)

MODEL_SOURCE = "speechbrain/emotion-recognition-wav2vec2-IEMOCAP"
MODEL_SAVE_DIR = os.path.join(
    os.path.dirname(__file__), "pretrained_models", "emotion-recognition-wav2vec2-IEMOCAP"
)

def load_model() -> EncoderClassifier:
    """Loads the SpeechBrain emotion recognition model."""
    return EncoderClassifier.from_hparams(source=MODEL_SOURCE, savedir=MODEL_SAVE_DIR)

classifier = load_model()

EMOTION_LABELS: Dict[int, str] = {
    0: "neutral",
    1: "angry",
    2: "sad",
    3: "happy",
}

def prepare_audio(file_like: io.BytesIO) -> torch.Tensor:
    """Loads the uploaded audio and prepares it for inference."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(file_like.read())
        tmp.flush()
        tmp_path = tmp.name

    signal, sample_rate = torchaudio.load(tmp_path)
    os.unlink(tmp_path)

    if signal.ndim > 1 and signal.shape[0] > 1:
        signal = torch.mean(signal, dim=0, keepdim=True)

    target_sample_rate = 16000
    if sample_rate != target_sample_rate:
        signal = torchaudio.functional.resample(signal, sample_rate, target_sample_rate)

    return signal

def detect_emotion(audio_tensor: torch.Tensor) -> Dict[str, float]:
    """Runs inference on the provided audio tensor and returns label probabilities."""
    with torch.no_grad():
        out_prob, _, _, _ = classifier.classify_batch(audio_tensor)

    probabilities = out_prob.squeeze(0)

    result = {}
    for index, probability in enumerate(probabilities.tolist()):
        label = EMOTION_LABELS.get(index, f"label_{index}")
        result[label] = probability

    return result

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    audio_file = request.files["audio"]
    audio_bytes = io.BytesIO(audio_file.read())

    if audio_bytes.getbuffer().nbytes == 0:
        return jsonify({"error": "Empty audio file"}), 400

    audio_bytes.seek(0)

    try:
        audio_tensor = prepare_audio(audio_bytes)
        emotion_probs = detect_emotion(audio_tensor)
    except Exception as exc:  # pragma: no cover - defensive fallback
        app.logger.exception("Failed to analyse audio")
        return jsonify({"error": str(exc)}), 500

    predicted_emotion = max(emotion_probs, key=emotion_probs.get)

    return jsonify({
        "predicted_emotion": predicted_emotion,
        "probabilities": emotion_probs,
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    debug_flag = os.environ.get("FLASK_DEBUG", "").lower() in {"1", "true", "yes"}
    app.run(host="0.0.0.0", port=port, debug=debug_flag)
