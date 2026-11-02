/**
 * @file inference-worker.js
 * Production-Hardened Neural VAD Worker.
 *
 * Enforces local path resolution for WebAssembly binaries, bypasses CDN
 * cross-origin blocks under strict COEP, and maintains recycled buffer transfers.
 */

// Host ONNX Runtime locally to satisfy Cross-Origin-Embedder-Policy: require-corp
importScripts('../vendor/ort.min.js');

let session = null;
let stateH = null;
let stateC = null;
const STATE_SIZE = 2 * 1 * 64;

let consecutiveSpeech = 0;
let consecutiveSilence = 0;
let hysteresisState = false;

async function bootstrapModel() {
  try {
    ort.env.wasm.wasmPaths = '../vendor/';
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.simd = true;

    session = await ort.InferenceSession.create('../models/silero_vad.onnx', {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all'
    });

    stateH = new Float32Array(STATE_SIZE).fill(0.0);
    stateC = new Float32Array(STATE_SIZE).fill(0.0);

    self.postMessage({ type: 'WORKER_READY', success: true });
  } catch (err) {
    self.postMessage({ type: 'WORKER_READY', success: false, error: err.message });
  }
}

self.onmessage = async (event) => {
  const { type, buffer } = event.data;

  if (type === 'INIT') {
    await bootstrapModel();
    return;
  }

  if (type === 'PROCESS_CHUNK' && session) {
    const pcm = new Float32Array(buffer);

    const inputTensor = new ort.Tensor('float32', pcm, [1, 512]);
    const srTensor = new ort.Tensor('int64', BigInt64Array.from([BigInt(16000)]), [1]);
    const hTensor = new ort.Tensor('float32', stateH, [2, 1, 64]);
    const cTensor = new ort.Tensor('float32', stateC, [2, 1, 64]);

    const feeds = {
      input: inputTensor,
      sr: srTensor,
      h: hTensor,
      c: cTensor
    };

    const startTime = performance.now();

    try {
      const results = await session.run(feeds);
      const rawProb = results.output.data[0];

      stateH.set(results.hn.data);
      stateC.set(results.cn.data);

      // Schmitt-Trigger Hysteresis Filter
      if (rawProb >= 0.65) {
        consecutiveSpeech++;
        consecutiveSilence = 0;
        if (consecutiveSpeech >= 3) hysteresisState = true;
      } else if (rawProb <= 0.35) {
        consecutiveSilence++;
        consecutiveSpeech = 0;
        if (consecutiveSilence >= 4) hysteresisState = false;
      }

      const inferenceTime = performance.now() - startTime;

      // Transfer audio buffer back to worklet for zero-allocation recycling
      self.postMessage({
        type: 'VAD_RESULT',
        probability: rawProb,
        isSpeech: hysteresisState,
        latencyMs: inferenceTime,
        recycledBuffer: buffer
      }, [buffer]);
    } catch (err) {
      self.postMessage({ type: 'VAD_ERROR', error: err.message, recycledBuffer: buffer }, [buffer]);
    }
  }
};
