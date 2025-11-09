# Emotion Recognition Web App - Railway Deployment Guide

## 📁 Project Structure

```
emotion-recognition-app/
├── app.py                 # Main Flask application
├── templates/
│   └── index.html        # Frontend interface
├── uploads/              # Temporary audio storage (created automatically)
├── requirements.txt      # Python dependencies
├── Procfile             # Railway process configuration
├── railway.json         # Railway deployment config
└── README.md            # Project documentation
```

## 🚀 Deployment Steps

### 1. Prepare Your Project

1. Create a new directory for your project:
```bash
mkdir emotion-recognition-app
cd emotion-recognition-app
```

2. Create the required files:
   - `app.py` - Main Flask application
   - `requirements.txt` - Dependencies
   - `Procfile` - Process configuration
   - `railway.json` - Railway configuration

3. Create a `templates` folder and add `index.html`:
```bash
mkdir templates
```

### 2. Initialize Git Repository

```bash
git init
git add .
git commit -m "Initial commit: Emotion recognition app"
```

### 3. Deploy to Railway

#### Option A: Using Railway CLI

1. Install Railway CLI:
```bash
npm install -g @railway/cli
```

2. Login to Railway:
```bash
railway login
```

3. Initialize and deploy:
```bash
railway init
railway up
```

#### Option B: Using Railway Dashboard

1. Go to [railway.app](https://railway.app)
2. Click "New Project" → "Deploy from GitHub repo"
3. Connect your GitHub account and select your repository
4. Railway will automatically detect the configuration and deploy

### 4. Configure Environment (if needed)

Railway will automatically set the `PORT` environment variable. No additional configuration needed for basic deployment.

## 🔧 Important Configuration Details

### Memory Requirements
- The SpeechBrain model requires significant memory (~2GB)
- Recommended Railway plan: **Pro** (for adequate resources)
- The app uses 2 workers with 120s timeout for handling large audio files

### Audio File Limits
- Max upload size: 16MB
- Supported formats: WAV, MP3, M4A, OGG, FLAC
- Files are automatically converted to 16kHz mono WAV

### Model Loading
- The emotion recognition model loads at startup (~30-60 seconds)
- First request may be slower as the model initializes
- Subsequent requests are faster

## 🧪 Testing Your Deployment

Once deployed, test your app:

1. **Health Check**:
   - Visit `https://your-app.railway.app/health`
   - Should return: `{"status": "healthy"}`

2. **Record Audio**:
   - Click "Record" tab
   - Allow microphone access
   - Click the microphone button to start/stop recording
   - Emotion analysis happens automatically

3. **Upload Audio**:
   - Click "Upload" tab
   - Upload a WAV/MP3/M4A file
   - Click "Analyze Emotion"

## 🐛 Troubleshooting

### Model Loading Issues
If the model fails to load:
- Check Railway logs: `railway logs`
- Ensure you're on a plan with sufficient memory (>2GB)
- The model downloads from HuggingFace on first run

### Timeout Errors
If requests timeout:
- Increase timeout in `Procfile`: `--timeout 180`
- Reduce audio length (use first 10 seconds)

### Memory Issues
If app crashes due to memory:
- Upgrade to Railway Pro plan
- Reduce number of workers in `Procfile`: `--workers 1`

### Audio Upload Issues
- Ensure file is under 16MB
- Check file format is supported
- Try converting to WAV format first

## 📊 Expected Emotions

The model recognizes these emotions from IEMOCAP dataset:
- **neu** - Neutral
- **hap** - Happy
- **sad** - Sad
- **ang** - Angry
- **exc** - Excited

## 🔒 Security Notes

- No authentication implemented (add if needed for production)
- Uploaded files are deleted after processing
- No data is stored permanently
- HTTPS is provided by Railway by default

## 💰 Cost Estimates

Railway pricing (as of 2024):
- **Hobby Plan**: $5/month (limited resources, may not be sufficient)
- **Pro Plan**: Pay-as-you-go (recommended for ML models)
- Typical cost with Pro: $10-20/month for moderate usage

## 🔄 Updating Your App

To deploy updates:

```bash
git add .
git commit -m "Update description"
git push
```

Railway automatically redeploys on push to main branch.

## 📝 Environment Variables (Optional)

You can add these in Railway dashboard if needed:
- `MAX_CONTENT_LENGTH`: Maximum upload size (default: 16MB)
- `UPLOAD_FOLDER`: Custom upload directory (default: uploads)

## 🆘 Getting Help

- Railway Docs: https://docs.railway.app
- SpeechBrain Docs: https://speechbrain.github.io
- Check Railway logs: `railway logs`
- Monitor in Railway dashboard: https://railway.app/dashboard

## ✅ Success Checklist

- [ ] Git repository initialized
- [ ] All files committed
- [ ] Railway project created
- [ ] App deployed successfully
- [ ] Health endpoint responds
- [ ] Can record audio via microphone
- [ ] Can upload audio files
- [ ] Emotion analysis works
- [ ] Results display correctly

---

**Note**: First deployment takes 5-10 minutes due to model download. Subsequent deployments are faster.