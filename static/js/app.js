const startBtn = document.getElementById('start-btn');
const stopBtn = document.getElementById('stop-btn');
const statusText = document.getElementById('status-text');
const emotionText = document.getElementById('emotion');
const probabilitiesTable = document.getElementById('probabilities');
const probabilitiesBody = probabilitiesTable.querySelector('tbody');

let audioContext;
let mediaStream;
let processor;
let sourceNode;
let chunks = [];
let desiredSampleRate = 16000;

function updateStatus(message) {
  statusText.textContent = message;
}

function resetUI() {
  emotionText.textContent = 'No prediction yet';
  probabilitiesTable.classList.add('hidden');
  probabilitiesBody.innerHTML = '';
}

function flattenChunks(chunksArray) {
  const length = chunksArray.reduce((sum, chunk) => sum + chunk.length, 0);
  const result = new Float32Array(length);
  let offset = 0;
  chunksArray.forEach((chunk) => {
    result.set(chunk, offset);
    offset += chunk.length;
  });
  return result;
}

function floatTo16BitPCM(float32Array) {
  const buffer = new ArrayBuffer(float32Array.length * 2);
  const view = new DataView(buffer);
  let offset = 0;
  for (let i = 0; i < float32Array.length; i += 1) {
    const s = Math.max(-1, Math.min(1, float32Array[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    offset += 2;
  }
  return buffer;
}

function writeWavHeader(view, sampleRate, dataLength) {
  const blockAlign = 2;
  const byteRate = sampleRate * blockAlign;

  view.setUint32(0, 0x52494646, false); // 'RIFF'
  view.setUint32(4, 36 + dataLength, true);
  view.setUint32(8, 0x57415645, false); // 'WAVE'
  view.setUint32(12, 0x666d7420, false); // 'fmt '
  view.setUint32(16, 16, true); // Subchunk1Size
  view.setUint16(20, 1, true); // AudioFormat (PCM)
  view.setUint16(22, 1, true); // NumChannels
  view.setUint32(24, sampleRate, true); // SampleRate
  view.setUint32(28, byteRate, true); // ByteRate
  view.setUint16(32, blockAlign, true); // BlockAlign
  view.setUint16(34, 16, true); // BitsPerSample
  view.setUint32(36, 0x64617461, false); // 'data'
  view.setUint32(40, dataLength, true);
}

function encodeWAV(channels, sampleRate) {
  const samples = flattenChunks(channels);
  const pcmBuffer = floatTo16BitPCM(samples);
  const wavBuffer = new ArrayBuffer(44 + pcmBuffer.byteLength);
  const view = new DataView(wavBuffer);

  writeWavHeader(view, sampleRate, pcmBuffer.byteLength);
  const dataBytes = new Uint8Array(wavBuffer, 44);
  dataBytes.set(new Uint8Array(pcmBuffer));

  return new Blob([new Uint8Array(wavBuffer)], { type: 'audio/wav' });
}

async function uploadRecording(blob) {
  const formData = new FormData();
  formData.append('audio', blob, 'recording.wav');

  updateStatus('Uploading audio…');

  try {
    const response = await fetch('/predict', {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'Failed to analyse audio');
    }

    const result = await response.json();
    emotionText.textContent = result.predicted_emotion;

    probabilitiesBody.innerHTML = '';
    Object.entries(result.probabilities)
      .sort(([, a], [, b]) => b - a)
      .forEach(([label, probability]) => {
        const row = document.createElement('tr');
        const emotionCell = document.createElement('td');
        const probabilityCell = document.createElement('td');
        emotionCell.textContent = label;
        probabilityCell.textContent = `${(probability * 100).toFixed(2)}%`;
        row.appendChild(emotionCell);
        row.appendChild(probabilityCell);
        probabilitiesBody.appendChild(row);
      });

    probabilitiesTable.classList.remove('hidden');
    updateStatus('Analysis complete');
  } catch (error) {
    console.error(error);
    updateStatus(error.message);
  }
}

async function startRecording() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    updateStatus('Microphone access is not supported in this browser.');
    return;
  }

  resetUI();
  updateStatus('Requesting microphone access…');

  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    audioContext = new AudioContext({ sampleRate: desiredSampleRate });
    desiredSampleRate = audioContext.sampleRate;
    sourceNode = audioContext.createMediaStreamSource(mediaStream);
    processor = audioContext.createScriptProcessor(4096, 1, 1);

    chunks = [];

    processor.onaudioprocess = (event) => {
      if (!mediaStream) return;
      const channelData = event.inputBuffer.getChannelData(0);
      chunks.push(new Float32Array(channelData));
    };

    sourceNode.connect(processor);
    processor.connect(audioContext.destination);

    updateStatus('Recording…');
    startBtn.disabled = true;
    stopBtn.disabled = false;
  } catch (error) {
    console.error(error);
    updateStatus('Microphone permission denied or unavailable.');
  }
}

function stopRecording() {
  if (!mediaStream || !processor || !audioContext) {
    return;
  }

  mediaStream.getTracks().forEach((track) => track.stop());
  mediaStream = null;

  processor.disconnect();
  if (sourceNode) {
    sourceNode.disconnect();
    sourceNode = null;
  }
  processor.onaudioprocess = null;
  processor = null;

  if (chunks.length === 0) {
    updateStatus('No audio captured. Try recording again.');
    if (audioContext) {
      audioContext.close();
      audioContext = null;
    }
    startBtn.disabled = false;
    stopBtn.disabled = true;
    return;
  }

  const wavBlob = encodeWAV(chunks, audioContext.sampleRate);
  audioContext.close();
  audioContext = null;
  chunks = [];

  updateStatus('Processing audio…');
  startBtn.disabled = false;
  stopBtn.disabled = true;

  uploadRecording(wavBlob);
}

startBtn.addEventListener('click', startRecording);
stopBtn.addEventListener('click', stopRecording);

