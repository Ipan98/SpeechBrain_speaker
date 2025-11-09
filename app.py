import os
import wave
import tempfile
import numpy as np
import torch
import torchaudio
import streamlit as st
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
Upload a voice sample to detect the **emotion** expressed in speech.  
Model: [`speechbrain/emotion-recognition-wav2vec2-IEMOCAP`](https://huggingface.co/speechbrain/emotion-recognition-wav2vec2-IEMOCAP)
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
# Upload section
# -----------------------------------------------------------
st.subheader("🎧 Upload an Audio File (WAV/MP3/OGG)")
audio_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "ogg"])

# -----------------------------------------------------------
# (Optional) Microphone recording — local only
# -----------------------------------------------------------
if os.environ.get("STREAMLIT_RUNTIME_ENV") != "cloud":
    try:
        import sounddevice as sd
        st.markdown("### 🎤 Record Your Voice (local use only)")
        duration = st.slider("Duration (seconds):", 3, 10, 5)
        if st.button("Record"):
            st.info("Recording... 🎙️")
            fs = 16000
            audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
            sd.wait()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                torchaudio.save(tmp.name, torch.tensor(audio.T), fs)
                audio_file = open(tmp.name, "rb")
            st.success("Recording finished!")
    except Exception as e:
        st.warning(f"Microphone not available: {e}")

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
