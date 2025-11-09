import streamlit as st
import wave
import os
import tempfile
import torch
import torchaudio
import numpy as np
import sounddevice as sd
from speechbrain.pretrained.interfaces import foreign_class

# -------------------------------
# Streamlit UI setup
# -------------------------------
st.set_page_config(page_title="Emotion Detection 🎧", layout="centered")
st.title("🎙️ Emotion Detection with SpeechBrain (Custom Interface)")
st.markdown("""
Upload or record an audio clip — the app will detect your emotion using your custom SpeechBrain model.

Model: **speechbrain/emotion-recognition-wav2vec2-IEMOCAP**  
Interface: **custom_interface.py**
""")

# -------------------------------
# Load SpeechBrain custom model
# -------------------------------
@st.cache_resource
def load_classifier():
    return foreign_class(
        source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
        pymodule_file="custom_interface.py",
        classname="CustomEncoderWav2vec2Classifier"
    )

classifier = load_classifier()


# -------------------------------
# Function for emotion recognition
# -------------------------------
def emotion_recognition(file_name):
    with wave.open(file_name, 'rb') as wav_file:
        frame_rate = wav_file.getframerate()
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        frame_count = wav_file.getnframes()
        segment_length = 10 * frame_rate

        # Loop through segments of audio
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


# -------------------------------
# Upload or record section
# -------------------------------
st.subheader("🎧 Upload or Record Audio")

col1, col2 = st.columns(2)

# Upload
with col1:
    uploaded_file = st.file_uploader("Upload WAV file", type=["wav", "mp3", "ogg"])

# Record
with col2:
    duration = st.slider("Record duration (seconds):", 3, 10, 5)
    if st.button("🎤 Record"):
        st.info("Recording...")
        fs = 16000
        audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
        sd.wait()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            torchaudio.save(tmp.name, torch.tensor(audio.T), fs)
            uploaded_file = open(tmp.name, "rb")
        st.success("Recording finished!")


# -------------------------------
# Run emotion detection
# -------------------------------
if uploaded_file:
    # Save uploaded/recorded audio
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
        tmpfile.write(uploaded_file.read())
        tmp_path = tmpfile.name

    st.audio(tmp_path)

    st.info("Analyzing emotion...")
    emotion = emotion_recognition(tmp_path)
    st.success(f"🧠 **Detected Emotion:** {emotion}")

    os.remove(tmp_path)

st.markdown("---")
st.caption("Built with SpeechBrain + Streamlit | Custom Interface Loader")
