# Speech Emotion Web App

This project provides a Flask web application that records speech through the browser and uses a SpeechBrain model to estimate the speaker's emotion.

## Features

- Browser-based audio recording with start/stop controls.
- Server-side emotion recognition powered by SpeechBrain's wav2vec2 model.
- Displays the predicted emotion and the model's confidence for each supported label.

## Requirements

Install Python dependencies:

```bash
pip install -r requirements.txt
```

> **Note:** The SpeechBrain model requires PyTorch and torchaudio. If you encounter installation issues, ensure you are using a Python environment with access to the appropriate wheels (CPU or GPU) for your platform.

## Running the app

Start the Flask development server:

```bash
flask --app app run --debug
```

Then open your browser at http://127.0.0.1:5000/ and allow microphone access when prompted.

## Deploying on Railway

1. [Create a Railway account](https://railway.app/) and install the [Railway CLI](https://docs.railway.app/guides/cli). Log in from your terminal:
   ```bash
   railway login
   ```
2. Initialise the project in this repository (only required the first time):
   ```bash
   railway init
   ```
3. Push the code to a GitHub repository and connect it to a new **Service** on Railway, or use `railway up` to deploy directly from the CLI.
4. Railway automatically installs the dependencies from `requirements.txt`. The included `Procfile` and `railway.json` configure Gunicorn (`gunicorn app:app`) as the start command so no further start command configuration is needed.
5. Set environment variables as required:
   - `PORT` is provided by Railway automatically—`app.py` reads it so no manual change is necessary.
   - Optional: `FLASK_DEBUG=1` to enable debug logging (not recommended for production).
6. (Recommended) Attach a persistent volume to the service so the SpeechBrain model download is cached between deploys. Without it, the model will be re-downloaded on each cold start, which increases the first-request latency.
7. Trigger a deploy. When the deployment completes, open the generated Railway domain to access the application.

## Emotion labels

The underlying model outputs the following labels:

- `neutral`
- `angry`
- `sad`
- `happy`

The app returns the label with the highest probability along with the complete probability distribution.

