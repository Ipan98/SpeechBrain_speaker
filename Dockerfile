FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    ffmpeg \ 
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .
COPY static ./static
COPY custom_interface.py .
# Pre-download the SpeechBrain model during build to speed up startup
RUN python -c "from speechbrain.pretrained.interfaces import foreign_class; \
    classifier = foreign_class(source='speechbrain/emotion-recognition-wav2vec2-IEMOCAP', \
    pymodule_file='custom_interface.py', classname='CustomEncoderWav2vec2Classifier')" || true

# Expose port
EXPOSE 5000

# Run the application with gunicorn for production
CMD gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 2 --timeout 300 --preload app:app