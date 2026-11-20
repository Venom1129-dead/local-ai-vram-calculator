// Local AI VRAM & Quantization Sizing Engine

const MODEL_PRESETS = {
  llama3_8b: { name: 'Llama 3.1 8B', params: 8.03, layers: 32, kvHeads: 8, headDim: 128, tag: 'llama3.1:8b-instruct' },
  qwen25_7b: { name: 'Qwen 2.5 7B', params: 7.61, layers: 28, kvHeads: 4, headDim: 128, tag: 'qwen2.5:7b-instruct' },
  qwen25_14b: { name: 'Qwen 2.5 14B', params: 14.77, layers: 48, kvHeads: 8, headDim: 128, tag: 'qwen2.5:14b-instruct' },
  qwen25_32b: { name: 'Qwen 2.5 32B', params: 32.76, layers: 64, kvHeads: 8, headDim: 128, tag: 'qwen2.5:32b-instruct' },
  qwen25_72b: { name: 'Qwen 2.5 72B', params: 72.71, layers: 80, kvHeads: 8, headDim: 128, tag: 'qwen2.5:72b-instruct' },
  mistral_12b: { name: 'Mistral NeMo 12B', params: 12.25, layers: 40, kvHeads: 8, headDim: 128, tag: 'mistral-nemo:12b' },
  gemma2_9b: { name: 'Gemma 2 9B', params: 9.24, layers: 42, kvHeads: 8, headDim: 256, tag: 'gemma2:9b' }
};

const QUANT_PRESETS = {
  q2_k: { name: 'Q2_K', bpw: 2.6 },
  q3_k_m: { name: 'Q3_K_M', bpw: 3.4 },
  q4_k_m: { name: 'Q4_K_M', bpw: 4.5 },
  q5_k_m: { name: 'Q5_K_M', bpw: 5.5 },
  q6_k: { name: 'Q6_K', bpw: 6.6 },
  q8_0: { name: 'Q8_0', bpw: 8.5 },
  fp16: { name: 'FP16', bpw: 16.0 }
};

const GPU_PRESETS = {
  '3060_12': { name: 'RTX 3060', vram: 12.0, bandwidth: 360 },
  '4060_8': { name: 'RTX 4060', vram: 8.0, bandwidth: 272 },
  '4070_12': { name: 'RTX 4070', vram: 12.0, bandwidth: 504 },
  '4080_16': { name: 'RTX 4080', vram: 16.0, bandwidth: 717 },
  '3090_24': { name: 'RTX 3090 / 4090', vram: 24.0, bandwidth: 1008 },
  'a100_80': { name: 'NVIDIA A100 / H100', vram: 80.0, bandwidth: 2000 },
  'apple_36': { name: 'Apple M3/M4 Max', vram: 36.0, bandwidth: 300 },
  'apple_64': { name: 'Apple M3/M4 Max', vram: 64.0, bandwidth: 400 },
  'apple_128': { name: 'Apple M2/M4 Ultra', vram: 128.0, bandwidth: 800 }
};

function calculateVRAM() {
  // 1. Get Model Config
  const modelKey = document.getElementById('model-preset').value;
  let paramsB, layers, kvHeads, headDim, modelTag;

  if (modelKey === 'custom') {
    paramsB = parseFloat(document.getElementById('custom-params').value) || 7.0;
    layers = parseInt(document.getElementById('custom-layers').value) || 32;
    kvHeads = parseInt(document.getElementById('custom-kv-heads').value) || 8;
    headDim = parseInt(document.getElementById('custom-head-dim').value) || 128;
    modelTag = 'custom-model';
  } else {
    const m = MODEL_PRESETS[modelKey];
    paramsB = m.params;
    layers = m.layers;
    kvHeads = m.kvHeads;
    headDim = m.headDim;
    modelTag = m.tag;
  }

  // 2. Quantization BPW
  const quantKey = document.getElementById('quant-preset').value;
  const bpw = QUANT_PRESETS[quantKey].bpw;

  // 3. Context & KV Cache
  const contextLen = parseInt(document.getElementById('context-slider').value) || 8192;
  const kvBytesPerElem = parseFloat(document.getElementById('kv-cache-type').value) || 2.0;

  // Update slider label
  const contextK = contextLen >= 1024 ? `${Math.round(contextLen / 1024)}k` : `${contextLen}`;
  document.getElementById('context-val-display').innerText = `${contextLen.toLocaleString()} tokens (${contextK})`;

  // Weights calculation: params * 1e9 * (bpw / 8) / (1024^3)
  const weightsGB = (paramsB * 1e9 * (bpw / 8.0)) / Math.pow(1024, 3);

  // KV Cache calculation: 2 * layers * kvHeads * headDim * contextLen * kvBytesPerElem / (1024^3)
  const kvCacheGB = (2 * layers * kvHeads * headDim * contextLen * kvBytesPerElem) / Math.pow(1024, 3);

  // Fixed CUDA context / PyTorch runtime overhead
  const cudaOverheadGB = 0.8;

  const totalVRAM = weightsGB + kvCacheGB + cudaOverheadGB;

  // 4. GPU Comparison
  const gpuKey = document.getElementById('gpu-preset').value;
  let gpuVRAM, gpuBandwidth, gpuName;

  if (gpuKey === 'custom_gpu') {
    gpuVRAM = parseFloat(document.getElementById('custom-vram').value) || 12.0;
    gpuBandwidth = parseFloat(document.getElementById('custom-bandwidth').value) || 360;
    gpuName = 'Custom GPU';
  } else {
    const g = GPU_PRESETS[gpuKey];
    gpuVRAM = g.vram;
    gpuBandwidth = g.bandwidth;
    gpuName = g.name;
  }

  // Tokens/sec estimate: Bandwidth (GB/s) / Model Weights (GB)
  const tokPerSec = weightsGB > 0 ? (gpuBandwidth / weightsGB).toFixed(1) : '0.0';

  // UI Updates
  document.getElementById('stat-total-vram').innerText = `${totalVRAM.toFixed(2)} GB`;
  document.getElementById('stat-breakdown-subtext').innerText = 
    `Weights: ${weightsGB.toFixed(2)} GB | KV Cache: ${kvCacheGB.toFixed(2)} GB | CUDA: ${cudaOverheadGB} GB`;
  document.getElementById('stat-speed').innerText = `~${tokPerSec} tok/s`;
  document.getElementById('stat-gpu-bandwidth').innerText = `Based on ${gpuBandwidth} GB/s Memory Bandwidth`;

  // KV Cache Warning
  const warnBox = document.getElementById('kv-cache-warning');
  if (kvCacheGB > 6.0) {
    warnBox.style.display = 'block';
    document.getElementById('warn-kv-size').innerText = `${kvCacheGB.toFixed(1)} GB`;
  } else {
    warnBox.style.display = 'none';
  }

  // Compatibility Badge
  const compatBadge = document.getElementById('stat-compat-badge');
  const headroomText = document.getElementById('stat-vram-remaining');
  const headroom = gpuVRAM - totalVRAM;

  if (headroom >= 0) {
    compatBadge.className = 'stat-badge status-fit';
    compatBadge.innerText = `✓ Fits in GPU (${gpuName} ${gpuVRAM}GB)`;
    headroomText.innerText = `${headroom.toFixed(2)} GB Headroom Remaining`;
  } else {
    compatBadge.className = 'stat-badge status-oom';
    compatBadge.innerText = `✗ CUDA OOM Crash (${Math.abs(headroom).toFixed(2)} GB Short!)`;
    headroomText.innerText = `Requires ${Math.abs(headroom).toFixed(2)} GB more VRAM or smaller quant`;
  }

  // Visual Bar Percentage
  const maxBarCapacity = Math.max(gpuVRAM, totalVRAM);
  const weightsPct = (weightsGB / maxBarCapacity) * 100;
  const kvPct = (kvCacheGB / maxBarCapacity) * 100;
  const cudaPct = (cudaOverheadGB / maxBarCapacity) * 100;

  document.getElementById('bar-weights').style.width = `${Math.min(weightsPct, 100)}%`;
  document.getElementById('bar-kv').style.width = `${Math.min(kvPct, 100)}%`;
  document.getElementById('bar-cuda').style.width = `${Math.min(cudaPct, 100)}%`;

  // CLI Command Generation
  const quantSuffix = quantKey === 'fp16' ? '' : `-${quantKey}`;
  const cliCommand = `ollama run ${modelTag}${quantSuffix}`;
  document.getElementById('cli-command').innerText = cliCommand;
}

// Event Listeners
document.getElementById('model-preset').addEventListener('change', e => {
  document.getElementById('custom-model-fields').style.display = e.target.value === 'custom' ? 'block' : 'none';
  calculateVRAM();
});

document.getElementById('gpu-preset').addEventListener('change', e => {
  document.getElementById('custom-gpu-fields').style.display = e.target.value === 'custom_gpu' ? 'block' : 'none';
  calculateVRAM();
});

['quant-preset', 'context-slider', 'kv-cache-type', 'custom-params', 'custom-layers', 'custom-kv-heads', 'custom-head-dim', 'custom-vram', 'custom-bandwidth']
  .forEach(id => {
    const el = document.getElementById(id);
    if (el) el.addEventListener('input', calculateVRAM);
  });

document.getElementById('btn-copy-cli').addEventListener('click', () => {
  const cmd = document.getElementById('cli-command').innerText;
  navigator.clipboard.writeText(cmd).then(() => {
    const btn = document.getElementById('btn-copy-cli');
    btn.innerText = 'Copied!';
    setTimeout(() => btn.innerText = 'Copy', 1500);
  });
});

document.addEventListener('DOMContentLoaded', calculateVRAM);
