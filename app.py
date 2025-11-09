import os
import wave
import tempfile
import numpy as np
import torch
import torchaudio
import streamlit as st
import sounddevice as sd
from speechbrain.pretrained.interfaces import foreign_class

# -----------------------------------------------------------
# ✅ Torchaudio backend fix for Streamlit Cloud
# -----------------------------------------------------------
os.environ["TORCHAUDIO_USE_SOUNDFILE_LEGACY_INTERFACE"] = "1"
try:
    torchaudio.set_audio_backend("soundfile")
except Exception as e:
    st.warning(f"⚠️ Could not set torchaudio backend: {e}")

# -----------------------------------------------------------
# Streamlit setup
# -----------------------------------------------------------
st.set_page_config(page_title="🎧 Emotion Detection", layout="centered")
st.title("🎙️ Speech Emotion Detection with SpeechBrain")
st.markdown("""
Upload or record a voice sample — the model will detect the **emotion** expressed in speech.  
This app uses [`speechbrain/emotion-recognition-wav2vec2-IEMOCAP`](https://huggingface.co/speechbrain/emotion-recognition-wav2vec2-IEMOCAP)
with your custom `custom_interface.py`.
""")

# -----------------------------------------------------------
# Load model (cached for performance)
# -----------------------------------------------------------
@st.cache_resource
def load_classifier():
    return foreign_class(
        source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
        pymodule_file="custom_interface.py",
        classname="CustomEncoderWav2vec2Classifier"
    )

classifier = load_classifier()

# -----------------------------------------------------------
# Emotion recognition logic
# -----------------------------------------------------------
def emotion_recognition(file_name):
    with wave.open(file_name, 'rb') as wav_file:
        frame_rate = wav_file.getframerate()
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        frame_count = wav_file.getnframes()
        segment_length = 10 * frame_rate  # 10-second chunks

        for i in range(0, frame_count, segment_length):
            temp_file_name = 'temp.wav'
            with wave.open(temp_file_name, 'wb') as new_wav_file:
                new_wav_file.setframerate(frame_rate)
                new_wav_file.setnchannels(channels)
                new_wav_file.setsampwidth(sample_width)
                segment = wav_file.readframes(segment_length)
                new_wav_file.writeframes(segment)

            out_prob, score, index, text_lab = classifier.classify_file(temp_file_name)
            return text_lab[0]

# -----------------------------------------------------------
# Upload or record audio
# -----------------------------------------------------------
st.subheader("🎧 Upload or Record Audio")

col1, col2 = st.columns(2)
audio_file = None

# Upload
with col1:
    uploaded = st.file_uploader("Upload an audio file (WAV/MP3/OGG)", type=["wav", "mp3", "ogg"])
    if uploaded:
        audio_file = uploaded

# Record
with col2:
    duration = st.slider("Record duration (seconds):", 3, 10, 5)
    if st.button("🎤 Record"):
        st.info("Recording... please speak clearly 🎙️")
        fs = 16000
        audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
        sd.wait()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            torchaudio.save(tmp.name, torch.tensor(audio.T), fs)
            audio_file = open(tmp.name, "rb")
        st.success("Recording finished!")

# -----------------------------------------------------------
# Run inference
# -----------------------------------------------------------
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
