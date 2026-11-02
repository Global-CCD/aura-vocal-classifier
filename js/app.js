/**
 * @file app.js
 * Production-Hardened UI Orchestrator.
 *
 * Implements:
 * 1. Transform matrix reset before DPI canvas scaling to prevent compounding zoom.
 * 2. Dynamic AudioContext sampleRate propagation to Worklet.
 * 3. Bidirectional zero-allocation telemetry buffer recycling.
 */

const STATE = {
  audioCtx: null,
  workletNode: null,
  mlWorker: null,
  isAudioActive: false,
  mlVadEnabled: false,   // STRICT DEFAULT: DISABLED
  currentTheme: 'light', // STRICT DEFAULT: LIGHT
  cachedMagnitudes: null,
  frameCount: 0,
  lastFpsCheck: performance.now(),
  fps: 60
};

const DOM = {
  btnAudioInit: document.getElementById('btn-audio-init'),
  toggleML: document.getElementById('toggle-ml'),
  toggleTheme: document.getElementById('toggle-theme'),
  metricEngine: document.getElementById('metric-engine'),
  metricClass: document.getElementById('metric-class'),
  metricMlProb: document.getElementById('metric-ml-prob'),
  metricRms: document.getElementById('metric-rms'),
  metricZcr: document.getElementById('metric-zcr'),
  metricSfm: document.getElementById('metric-sfm'),
  metricTilt: document.getElementById('metric-tilt'),
  metricIbpc: document.getElementById('metric-ibpc'),
  metricKurt: document.getElementById('metric-kurt'),
  metricFormants: document.getElementById('metric-formants'),
  metricFps: document.getElementById('metric-fps'),
  viewport: document.getElementById('viewport'),
  canvas: document.getElementById('spectral-canvas')
};

const canvasCtx = DOM.canvas.getContext('2d', { alpha: false });

const CLASSIFICATION_LABELS = [
  'NOISE',
  'VOICED_SPEECH',
  'WHISPER_UNVOICED',
  'TRANSIENT_FRAGMENT',
  'SYNTHETIC_VOCODER'
];

/**
 * Resets transformation matrix before DPR scaling to eliminate zoom drift
 */
function resizeCanvas() {
  const dpr = window.devicePixelRatio || 1;
  const rect = DOM.viewport.getBoundingClientRect();
  DOM.canvas.width = Math.floor(rect.width * dpr);
  DOM.canvas.height = Math.floor(rect.height * dpr);

  // Explicit matrix reset
  canvasCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
}
window.addEventListener('resize', resizeCanvas);
resizeCanvas();

DOM.toggleTheme.addEventListener('change', (e) => {
  STATE.currentTheme = e.target.checked ? 'dark' : 'light';
  if (STATE.currentTheme === 'dark') {
    document.documentElement.setAttribute('data-theme', 'dark');
  } else {
    document.documentElement.removeAttribute('data-theme');
  }
});

function configureMLVAD(enabled) {
  STATE.mlVadEnabled = enabled;

  if (STATE.workletNode) {
    STATE.workletNode.port.postMessage({
      type: 'TOGGLE_ML_VAD',
      enabled: enabled
    });
  }

  if (enabled) {
    DOM.metricMlProb.innerText = 'INITIALIZING...';
    DOM.metricMlProb.style.color = 'var(--text-secondary)';

    if (!STATE.mlWorker) {
      STATE.mlWorker = new Worker('js/inference-worker.js');
      STATE.mlWorker.postMessage({ type: 'INIT' });

      STATE.mlWorker.onmessage = (event) => {
        const { type, probability, isSpeech, error, recycledBuffer } = event.data;

        // Return recycled buffer to AudioWorklet pool
        if (recycledBuffer && STATE.workletNode) {
          STATE.workletNode.port.postMessage(
            { type: 'RECYCLE_DECIM', buffer: recycledBuffer },
            [recycledBuffer]
          );
        }

        if (type === 'WORKER_READY') {
          DOM.metricMlProb.innerText = 'ONLINE (0.0%)';
        } else if (type === 'VAD_RESULT') {
          DOM.metricMlProb.innerText = `${(probability * 100).toFixed(1)}%`;
          DOM.metricMlProb.style.color = isSpeech ? 'var(--status-active-fg)' : 'var(--text-primary)';
        } else if (type === 'VAD_ERROR') {
          console.error('[ML Worker Failure]', error);
          DOM.metricMlProb.innerText = 'ERROR';
          DOM.metricMlProb.style.color = 'var(--status-alert-fg)';
        }
      };
    }
  } else {
    DOM.metricMlProb.innerText = 'DISABLED';
    DOM.metricMlProb.style.color = 'var(--text-muted)';
  }
}

DOM.toggleML.addEventListener('change', (e) => {
  configureMLVAD(e.target.checked);
});

async function bootAudioEngine() {
  try {
    STATE.audioCtx = new (window.AudioContext || window.webkitAudioContext)({
      latencyHint: 'interactive'
    });

    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false
      }
    });

    await STATE.audioCtx.audioWorklet.addModule('js/spectral-processor.js');

    const source = STATE.audioCtx.createMediaStreamSource(stream);
    STATE.workletNode = new AudioWorkletNode(STATE.audioCtx, 'spectral-engine-processor', {
      numberOfInputs: 1,
      numberOfOutputs: 1,
      outputChannelCount: [1]
    });

    STATE.workletNode.port.onmessage = (event) => {
      const { type, buffer } = event.data;

      if (type === 'DSP_TELEMETRY' && buffer) {
        const data = new Float32Array(buffer);
        updateTelemetryUI(data);

        // Keep local reference to magnitudes for rendering
        if (!STATE.cachedMagnitudes || STATE.cachedMagnitudes.length !== data.length - 10) {
          STATE.cachedMagnitudes = new Float32Array(data.length - 10);
        }
        STATE.cachedMagnitudes.set(data.subarray(10));

        // Recycle telemetry buffer back to worklet thread
        STATE.workletNode.port.postMessage(
          { type: 'RECYCLE_TELEMETRY', buffer: buffer },
          [buffer]
        );
      } else if (type === 'AUDIO_CHUNK_16K' && buffer && STATE.mlWorker && STATE.mlVadEnabled) {
        STATE.mlWorker.postMessage({ type: 'PROCESS_CHUNK', buffer }, [buffer]);
      }
    };

    source.connect(STATE.workletNode);

    const sinkGain = STATE.audioCtx.createGain();
    sinkGain.gain.value = 0.0;
    STATE.workletNode.connect(sinkGain);
    sinkGain.connect(STATE.audioCtx.destination);

    DOM.btnAudioInit.disabled = true;
    DOM.btnAudioInit.innerText = 'Engine Running';
    DOM.metricEngine.className = 'badge badge-active';
    DOM.metricEngine.innerText = `ONLINE (${(STATE.audioCtx.sampleRate / 1000).toFixed(1)} kHz)`;
    STATE.isAudioActive = true;

    requestAnimationFrame(renderVisualizer);
  } catch (err) {
    console.error('[Audio Initialization Failure]', err);
    DOM.metricEngine.className = 'badge badge-alert';
    DOM.metricEngine.innerText = 'FAILED';
  }
}

DOM.btnAudioInit.addEventListener('click', () => {
  if (!STATE.isAudioActive) bootAudioEngine();
});

function updateTelemetryUI(data) {
  const classId = Math.round(data[0]);
  const classification = CLASSIFICATION_LABELS[classId] || 'NOISE';
  const isSpeech = Boolean(data[9]);

  DOM.metricRms.innerText = `${data[1].toFixed(1)} dBFS`;
  DOM.metricZcr.innerText = data[2].toFixed(4);
  DOM.metricSfm.innerText = data[3].toFixed(4);
  DOM.metricTilt.innerText = `${data[4].toFixed(2)} dB/oct`;
  DOM.metricIbpc.innerText = `${data[5].toFixed(2)} rad`;
  DOM.metricKurt.innerText = data[6].toFixed(2);
  DOM.metricFormants.innerText = (data[7] > 0) ? `${Math.round(data[7])} / ${Math.round(data[8])} Hz` : '-- / -- Hz';

  DOM.metricClass.innerText = classification;
  if (classification === 'SYNTHETIC_VOCODER') {
    DOM.metricClass.style.color = 'var(--status-alert-fg)';
  } else if (isSpeech) {
    DOM.metricClass.style.color = 'var(--status-active-fg)';
  } else {
    DOM.metricClass.style.color = 'var(--text-secondary)';
  }
}

function renderVisualizer() {
  if (!STATE.isAudioActive) return;

  STATE.frameCount++;
  const now = performance.now();
  if (now - STATE.lastFpsCheck >= 1000) {
    STATE.fps = STATE.frameCount;
    STATE.frameCount = 0;
    STATE.lastFpsCheck = now;
    DOM.metricFps.innerText = `${STATE.fps} FPS`;
  }

  const rect = DOM.viewport.getBoundingClientRect();
  const width = rect.width;
  const height = rect.height;

  const styles = getComputedStyle(document.documentElement);
  canvasCtx.fillStyle = styles.getPropertyValue('--canvas-bg').trim();
  canvasCtx.fillRect(0, 0, width, height);

  canvasCtx.strokeStyle = styles.getPropertyValue('--canvas-grid').trim();
  canvasCtx.lineWidth = 1;
  for (let y = 0; y < height; y += 65) {
    canvasCtx.beginPath();
    canvasCtx.moveTo(0, y);
    canvasCtx.lineTo(width, y);
    canvasCtx.stroke();
  }

  if (STATE.cachedMagnitudes) {
    const bins = STATE.cachedMagnitudes.length;
    const barWidth = width / bins;
    canvasCtx.fillStyle = styles.getPropertyValue('--canvas-bar').trim();

    for (let i = 0; i < bins; i++) {
      const mag = STATE.cachedMagnitudes[i];
      const normalizedHeight = Math.min(height, (Math.log10(mag + 1.0) / 3.0) * height);
      canvasCtx.fillRect(i * barWidth, height - normalizedHeight, Math.max(1, barWidth - 0.5), normalizedHeight);
    }
  }

  requestAnimationFrame(renderVisualizer);
}
