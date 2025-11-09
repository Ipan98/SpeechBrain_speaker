import os
import sys
import types
import streamlit as st

# -----------------------------------------------------------
# ✅ Torchaudio patch before SpeechBrain imports
# -----------------------------------------------------------
try:
    import torchaudio
    import soundfile  # ensures the backend exists

    # Monkey-patch list_audio_backends to always return ["soundfile"]
    if not hasattr(torchaudio, "list_audio_backends") or not torchaudio.list_audio_backends():
        def _fake_backends():
            return ["soundfile"]
        torchaudio.list_audio_backends = _fake_backends

    torchaudio.set_audio_backend("soundfile")
    st.write("✅ Torchaudio backend set to:", torchaudio.get_audio_backend())
except Exception as e:
    st.warning(f"⚠️ Torchaudio backend init failed: {e}")

# -----------------------------------------------------------
# Now import SpeechBrain (after backend is safe)
# -----------------------------------------------------------
from speechbrain.pretrained.interfaces import foreign_class
import wave
import tempfile
import numpy as np
import torch

# -----------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------
st.set_page_config(page_title="🎧 Emotion Detection", layout="centered")
st.title("🎙️ Speech Emotion Detection with SpeechBrain")
st.markdown("""
Upload a voice sample to detect the **emotion** expressed in speech.  
Model: [`speechbrain/emotion-recognition-wav2vec2-IEMOCAP`](https://huggingface.co/speechbrain/emotion-recognition-wav2vec2-IEMOCAP)
""")

@st.cache_resource
def load_classifier():
    return foreign_class(
        source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
        pymodule_file="custom_interface.py",
        classname="CustomEncoderWav2vec2Classifier"
    )

classifier = load_classifier()

def emotion_recognition(file_name):
    with wave.open(file_name, 'rb') as wav_file:
        frame_rate = wav_file.getframerate()
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        frame_count = wav_file.getnframes()
        segment_length = 10 * frame_rate
        for i in range(0, frame_count, segment_length):
            temp_file_name = 'temp.wav'
            with wave.open(temp_file_name, 'wb') as new_wav_file:
                new_wav_file.setframerate(frame_rate)
                new_wav_file.setnchannels(channels)
                new_wav_file.setsampwidth(sample_width)
                new_wav_file.writeframes(wav_file.readframes(segment_length))
            out_prob, score, index, text_lab = classifier.classify_file(temp_file_name)
            return text_lab[0]

st.subheader("🎧 Upload an Audio File (WAV/MP3/OGG)")
audio_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "ogg"])

if audio_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
        tmpfile.write(audio_file.read())
        tmp_path = tmpfile.name
    st.audio(tmp_path)
    st.info("🔍 Analyzing emotion...")
    try:
        emotion = emotion_recognition(tmp_path)
        st.success(f"🧠 **Detected Emotion:** {emotion}")
    except Exception as e:
        st.error(f"❌ Error during emotion recognition: {e}")
    os.remove(tmp_path)

st.markdown("---")
st.caption("Built with ❤️ using Streamlit + SpeechBrain")
