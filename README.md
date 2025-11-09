# Emotion Recognition Web App - Railway Deployment

A full-stack application for recording audio and classifying emotions using SpeechBrain's emotion recognition model.

## Project Structure

```
emotion-recognition/
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── app.py
│   └── railway.json
├── frontend/
│   ├── Dockerfile
│   ├── nginx.conf
│   ├── package.json
│   ├── public/
│   │   └── index.html
│   └── src/
│       ├── App.js
│       └── index.js
└── README.md
```

## Setup Instructions

### 1. Create the project structure

```bash
mkdir -p emotion-recognition/backend emotion-recognition/frontend/src emotion-recognition/frontend/public
cd emotion-recognition
```

### 2. Create Backend Files

Place the following files in `backend/`:
- `app.py` (Flask backend)
- `Dockerfile` (Backend Docker config)
- `requirements.txt` (Python dependencies)
- `railway.json` (Railway configuration)

### 3. Create Frontend Files

**frontend/public/index.html:**
```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <meta name="theme-color" content="#000000" />
  <meta name="description" content="Emotion Recognition App" />
  <title>Emotion Recognition</title>
  <script src="https://cdn.tailwindcss.com"></script>
</head>
<body>
  <noscript>You need to enable JavaScript to run this app.</noscript>
  <div id="root"></div>
</body>
</html>
```

**frontend/src/index.js:**
```javascript
import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
```

**frontend/src/App.js:**
- Use the React component code provided

Place additional files in `frontend/`:
- `Dockerfile`
- `nginx.conf`
- `package.json`
- `railway.json`

## Deployment on Railway

### Method 1: Deploy from GitHub (Recommended)

1. **Push your code to GitHub:**
```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin YOUR_GITHUB_REPO_URL
git push -u origin main
```

2. **Deploy on Railway:**
   - Go to [railway.app](https://railway.app)
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Choose your repository
   - Railway will detect both services automatically

3. **Configure Services:**
   
   **Backend Service:**
   - Railway will auto-detect the Dockerfile in `backend/`
   - Add environment variable:
     - `PORT`: 5000 (Railway sets this automatically)
   - Note the generated URL (e.g., `https://your-backend.railway.app`)

   **Frontend Service:**
   - Railway will auto-detect the Dockerfile in `frontend/`
   - Add environment variable:
     - `REACT_APP_API_URL`: Your backend URL from above
   - Railway will generate a public URL

4. **Update Frontend Environment:**
   - In Railway dashboard, go to Frontend service
   - Add variable: `REACT_APP_API_URL=https://your-backend.railway.app`
   - Redeploy if needed

### Method 2: Deploy with Railway CLI

1. **Install Railway CLI:**
```bash
npm install -g @railway/cli
```

2. **Login:**
```bash
railway login
```

3. **Deploy Backend:**
```bash
cd backend
railway init
railway up
```

4. **Deploy Frontend:**
```bash
cd ../frontend
railway init
railway up
```

5. **Link services and set environment variables in Railway dashboard**

## Running Locally

### Using Docker

**Backend:**
```bash
cd backend
docker build -t emotion-backend .
docker run -p 5000:5000 emotion-backend
```

**Frontend:**
```bash
cd frontend
docker build -t emotion-frontend .
docker run -p 80:80 emotion-frontend
```

### Without Docker

**Backend:**
```bash
cd backend
pip install -r requirements.txt
python app.py
```

**Frontend:**
```bash
cd frontend
npm install
npm start
```

## Railway Configuration Files

### backend/railway.json
```json
{
  "$schema": "https://railway.app/railway.schema.json",
  "build": {
    "builder": "DOCKERFILE",
    "dockerfilePath": "Dockerfile"
  },
  "deploy": {
    "startCommand": "python app.py",
    "healthcheckPath": "/health",
    "healthcheckTimeout": 100,
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 10
  }
}
```

### frontend/railway.json
```json
{
  "$schema": "https://railway.app/railway.schema.json",
  "build": {
    "builder": "DOCKERFILE",
    "dockerfilePath": "Dockerfile"
  },
  "deploy": {
    "startCommand": "nginx -g 'daemon off;'",
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 10
  }
}
```

## Environment Variables

### Backend
- `PORT`: Automatically set by Railway (default: 5000)
- `FLASK_ENV`: Set to `production`

### Frontend
- `REACT_APP_API_URL`: Your Railway backend URL (e.g., `https://your-backend.railway.app`)

## API Endpoints

### Backend

**GET /health**
- Health check endpoint
- Returns: `{"status": "healthy"}`

**POST /classify**
- Classifies emotion from audio
- Body: FormData with `audio` file (WAV format)
- Returns:
```json
{
  "emotion": "hap",
  "confidence": 0.85
}
```

## Features

- **Audio Recording**: Record directly from browser microphone
- **Emotion Classification**: Detects 5 emotions (Happy, Sad, Angry, Neutral, Fearful)
- **Real-time Analysis**: Fast classification using SpeechBrain model
- **Beautiful UI**: Modern, responsive design with Tailwind CSS
- **Railway-Optimized**: Configured for easy Railway deployment

## Emotion Labels

- `hap`: Happy
- `sad`: Sad
- `ang`: Angry
- `neu`: Neutral
- `fea`: Fearful

## Troubleshooting

### Railway Deployment Issues

**Build fails:**
- Check Railway build logs
- Ensure Dockerfile paths are correct
- Verify requirements.txt has all dependencies

**Backend timeout on first request:**
- The first request downloads the SpeechBrain model (~100MB)
- Increase Railway timeout if needed
- Model is cached after first download

**CORS errors:**
- Verify `REACT_APP_API_URL` in frontend environment variables
- Ensure it matches your backend Railway URL
- Check Flask-CORS is installed

**Frontend can't connect to backend:**
- Verify backend is deployed and running
- Check `REACT_APP_API_URL` environment variable
- Ensure backend URL is correct (no trailing slash)

### General Issues

**Microphone not working:**
- Railway provides HTTPS by default (required for microphone)
- Check browser permissions

**Audio format issues:**
- Browser must support MediaRecorder API
- WAV format is used

## Cost Optimization on Railway

- Railway offers $5 free credit monthly
- Backend model download happens once, then cached
- Consider using Railway's sleep mode for development

## Tech Stack

**Backend:**
- Flask (Python web framework)
- SpeechBrain (Emotion recognition)
- PyTorch (ML backend)

**Frontend:**
- React (UI framework)
- Tailwind CSS (Styling)
- MediaRecorder API (Audio recording)

**Deployment:**
- Railway (Hosting platform)
- Docker (Containerization)
- Nginx (Frontend server)

## Next Steps

1. Push code to GitHub
2. Connect repository to Railway
3. Configure environment variables
4. Access your deployed app!

Your app will be available at:
- Frontend: `https://your-app.railway.app`
- Backend: `https://your-backend.railway.app`