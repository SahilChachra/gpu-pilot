// ── vLLM Advisor — Main App JS ─────────────────────────────────────────────

// Live price state — populated by fetchLivePrices()
let LIVE_PRICES = null;       // { gpu_name: { runpod_hr, lambda_hr, vastai_hr }, ... }
let PRICES_ARE_LIVE = false;  // true once we've gotten a successful fetch

document.addEventListener('DOMContentLoaded', () => {
  initNav();
  initGpuFinder();
  initConfigGen();
  initCalculator();
  initParamRef();
  initChat();
  checkApiStatus();
  fetchLivePrices();
});

// ── NAV ───────────────────────────────────────────────────────────────────

function initNav() {
  document.querySelectorAll('.nav-link').forEach(link => {
    link.addEventListener('click', e => {
      e.preventDefault();
      const tab = link.dataset.tab;
      document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(t => { t.classList.add('hidden'); t.classList.remove('active'); });
      link.classList.add('active');
      document.getElementById('tab-' + tab).classList.remove('hidden');
      document.getElementById('tab-' + tab).classList.add('active');
    });
  });
}

// ── API STATUS ────────────────────────────────────────────────────────────

async function checkApiStatus() {
  try {
    const r = await fetch('/api/gpus');
    if (r.ok) {
      document.getElementById('statusDot').className = 'status-dot ok';
      document.getElementById('statusText').textContent = 'Server online';
    }
  } catch {
    document.getElementById('statusDot').className = 'status-dot err';
    document.getElementById('statusText').textContent = 'Server error';
  }
}

// ── LIVE PRICES ───────────────────────────────────────────────────────────

async function fetchLivePrices(manual = false) {
  const priceEl = document.getElementById('priceStatus');
  if (priceEl) priceEl.textContent = manual ? 'Refreshing…' : 'Fetching prices…';

  try {
    const r = await fetch('/api/prices');
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const data = await r.json();

    LIVE_PRICES = data.prices || {};
    PRICES_ARE_LIVE = data.is_live;

    if (priceEl) {
      if (data.is_live) {
        const ago = data.last_updated ? Math.round((Date.now() / 1000 - data.last_updated) / 60) : '?';
        const hits = (data.runpod_hits || 0) + (data.vastai_hits || 0);
        priceEl.innerHTML = `<span class="price-live-badge">● LIVE</span> ${hits} GPUs · ${ago}m ago`;
        if (data.errors && data.errors.length) {
          priceEl.title = 'Errors: ' + data.errors.join('; ');
        }
      } else {
        priceEl.textContent = 'Est. prices (fetching…)';
      }
    }
  } catch (e) {
    if (priceEl) priceEl.textContent = 'Price fetch failed';
  }
}

async function refreshPrices() {
  const btn = document.getElementById('refreshPricesBtn');
  if (btn) { btn.disabled = true; btn.textContent = '↻ Refreshing…'; }

  try {
    const r = await fetch('/api/refresh-prices', { method: 'POST' });
    const data = await r.json();
    LIVE_PRICES = null;  // force re-fetch via /api/prices
    await fetchLivePrices(true);
  } catch (e) {
    const priceEl = document.getElementById('priceStatus');
    if (priceEl) priceEl.textContent = 'Refresh failed';
  } finally {
    if (btn) { btn.disabled = false; btn.textContent = '↻ Refresh Prices'; }
  }
}

// ── SHARED QUANT STATE ────────────────────────────────────────────────────

function makeQuantPicker(containerEl, onChange) {
  let current = 'fp16';
  containerEl.querySelectorAll('.quant-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      containerEl.querySelectorAll('.quant-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      current = btn.dataset.quant;
      if (onChange) onChange(current);
    });
  });
  return { get: () => current };
}

function makePriorityPicker(containerEl) {
  let current = 'balanced';
  containerEl.querySelectorAll('.priority-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      containerEl.querySelectorAll('.priority-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      current = btn.dataset.priority;
    });
  });
  return { get: () => current };
}

// ── GPU FINDER ────────────────────────────────────────────────────────────

function initGpuFinder() {
  const modelSel = document.getElementById('gf-model');
  const customRow = document.getElementById('gf-custom-model-row');
  const quantPicker = makeQuantPicker(document.querySelector('#tab-gpu-finder'), q => {
    updateQuantHint(q, 'gf-quant-hint');
  });

  modelSel.addEventListener('change', () => {
    customRow.classList.toggle('hidden', modelSel.value !== '__custom__');
  });

  // HuggingFace auto-fetch when user types a model ID
  attachHfAutoFetch('gf-custom-model-id', 'gf-custom-params', 'gf-hf-info');

  document.getElementById('gf-submit').addEventListener('click', async () => {
    const modelId = modelSel.value === '__custom__'
      ? document.getElementById('gf-custom-model-id').value.trim()
      : modelSel.value;
    if (!modelId) return showError('gf-results', 'Please select or enter a model.');

    const payload = {
      model_id: modelId,
      quant: quantPicker.get(),
      context_len: document.getElementById('gf-context').value,
      target_batch: document.getElementById('gf-batch').value,
      target_tps: document.getElementById('gf-tps').value,
      num_gpus: document.getElementById('gf-num-gpus').value,
      custom_params_b: document.getElementById('gf-custom-params')?.value || 7,
    };

    showLoading('gf-results');
    const data = await postApi('/api/recommend-gpu', payload);
    if (data && !data.error) {
      saveToHistory(payload, data);
      renderHistory();
    }
    renderGpuResults(data, 'gf-results', quantPicker.get());
  });

  renderHistory();  // show persisted history on load
}

function updateQuantHint(quant, hintId) {
  const hints = {
    fp8: '★ Best on H100/Ada — near fp16 quality with 2× throughput',
    awq: '4-bit weights, fast fused kernels — best quality/speed at 4-bit',
    gptq: '4-bit weights, slightly slower kernels than AWQ',
    int8: '8-bit, broader GPU support (A10G, RTX 3090)',
    fp16: 'Full precision — highest quality, highest VRAM',
    bfloat16: 'Better dynamic range than fp16, same memory — preferred for modern models',
  };
  const el = document.getElementById(hintId);
  if (el) el.textContent = hints[quant] || '';
}

// ── HF MODEL AUTO-FETCH ───────────────────────────────────────────────────

function attachHfAutoFetch(inputId, paramsId, infoId) {
  const input = document.getElementById(inputId);
  const paramsEl = document.getElementById(paramsId);
  const infoEl = document.getElementById(infoId);
  if (!input || !infoEl) return;

  let timer = null;
  input.addEventListener('input', () => {
    clearTimeout(timer);
    const val = input.value.trim();
    infoEl.innerHTML = '';
    if (!val || !val.includes('/')) return;

    infoEl.innerHTML = '<span class="hf-fetching">Fetching model info…</span>';
    timer = setTimeout(async () => {
      try {
        const r = await fetch(`/api/hf-model-info?model_id=${encodeURIComponent(val)}`);
        const d = await r.json();
        if (d.error) {
          infoEl.innerHTML = `<span class="hf-error">⚠ ${escHtml(d.error)}</span>`;
          return;
        }
        if (paramsEl) paramsEl.value = d.params_b;
        const moeTag = d.is_moe ? ` · <span style="color:var(--orange)">⚡ MoE ${d.num_experts} experts</span>` : '';
        infoEl.innerHTML = `<span class="hf-ok">✓ Auto-filled</span>`
          + ` <span class="hf-detail">${escHtml(d.model_type)} · ${d.params_b}B params · ${d.layers} layers · ${d.kv_heads} KV heads${moeTag}</span>`;
      } catch {
        infoEl.innerHTML = '<span class="hf-error">Fetch error — check model ID</span>';
      }
    }, 700);
  });
}

// ── SEARCH HISTORY ────────────────────────────────────────────────────────

const HISTORY_KEY = 'vllm_gf_history';
const HISTORY_MAX = 20;

function saveToHistory(payload, data) {
  const recs = data.recommendations || [];
  const top = recs[0];
  const entry = {
    id: `h_${Date.now()}`,
    ts: Date.now(),
    model_id: payload.model_id,
    quant: payload.quant,
    context_len: payload.context_len,
    num_gpus: payload.num_gpus,
    top_gpu: top?.gpu || '—',
    top_tps: top?.est_tps || 0,
    model_vram_gb: data.model_vram_gb || 0,
    fits_count: recs.filter(r => r.fits).length,
  };
  let hist = getHistory();
  // De-duplicate: remove entry with same model+quant+context if it exists
  hist = hist.filter(h => !(h.model_id === entry.model_id && h.quant === entry.quant && h.context_len === entry.context_len));
  hist.unshift(entry);
  if (hist.length > HISTORY_MAX) hist = hist.slice(0, HISTORY_MAX);
  try { localStorage.setItem(HISTORY_KEY, JSON.stringify(hist)); } catch {}
}

function getHistory() {
  try { return JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]'); } catch { return []; }
}

window.clearHistory = function() {
  localStorage.removeItem(HISTORY_KEY);
  renderHistory();
};

window.deleteHistoryEntry = function(id) {
  const hist = getHistory().filter(h => h.id !== id);
  try { localStorage.setItem(HISTORY_KEY, JSON.stringify(hist)); } catch {}
  renderHistory();
};

window.restoreHistoryEntry = function(id) {
  const entry = getHistory().find(h => h.id === id);
  if (!entry) return;
  const modelSel = document.getElementById('gf-model');
  // Try to match in dropdown, else go custom
  const opt = [...modelSel.options].find(o => o.value === entry.model_id);
  if (opt) {
    modelSel.value = entry.model_id;
    document.getElementById('gf-custom-model-row').classList.add('hidden');
  } else {
    modelSel.value = '__custom__';
    document.getElementById('gf-custom-model-row').classList.remove('hidden');
    document.getElementById('gf-custom-model-id').value = entry.model_id;
  }
  document.getElementById('gf-context').value = entry.context_len;
  document.getElementById('gf-num-gpus').value = entry.num_gpus;
  // Activate correct quant button
  document.querySelectorAll('#tab-gpu-finder .quant-btn').forEach(b => {
    b.classList.toggle('active', b.dataset.quant === entry.quant);
  });
  // Scroll to top of finder and trigger search
  document.getElementById('tab-gpu-finder').scrollIntoView({ behavior: 'smooth' });
  document.getElementById('gf-submit').click();
};

function renderHistory() {
  const el = document.getElementById('gf-history-list');
  if (!el) return;
  const hist = getHistory();
  if (!hist.length) {
    el.innerHTML = '<div class="empty-state" style="font-size:11px">No searches yet — run a query above.</div>';
    return;
  }
  el.innerHTML = hist.map(h => {
    const ago = timeAgo(h.ts);
    const shortModel = h.model_id.split('/').pop();
    const contextK = Math.round(parseInt(h.context_len) / 1024);
    return `<div class="history-item" onclick="restoreHistoryEntry('${h.id}')">
      <div class="history-item-main">
        <span class="history-model">${escHtml(shortModel)}</span>
        <span class="history-badges">
          <span class="history-badge">${escHtml(h.quant)}</span>
          <span class="history-badge">${contextK}K ctx</span>
          ${parseInt(h.num_gpus) > 1 ? `<span class="history-badge">${h.num_gpus}× GPU</span>` : ''}
        </span>
      </div>
      <div class="history-item-sub">
        <span style="color:var(--accent)">${escHtml(h.top_gpu)}</span>
        <span style="color:var(--muted)">${h.top_tps > 0 ? h.top_tps.toLocaleString() + ' tok/s' : ''}</span>
        <span style="color:var(--muted2);margin-left:auto">${ago}</span>
        <button class="history-del-btn" onclick="event.stopPropagation();deleteHistoryEntry('${h.id}')">×</button>
      </div>
    </div>`;
  }).join('');
}

function timeAgo(ts) {
  const s = Math.floor((Date.now() - ts) / 1000);
  if (s < 60) return 'just now';
  if (s < 3600) return `${Math.floor(s/60)}m ago`;
  if (s < 86400) return `${Math.floor(s/3600)}h ago`;
  return `${Math.floor(s/86400)}d ago`;
}

function renderGpuResults(data, containerId, quant) {
  const el = document.getElementById(containerId);
  if (!data || data.error) { showError(containerId, data?.error || 'Failed'); return; }

  const recs = data.recommendations;
  if (!recs.length) { el.innerHTML = '<div class="empty-state">No GPUs matched. Try relaxing requirements.</div>'; return; }

  const modelVram = data.model_vram_gb;
  let html = '';

  // Model summary banner
  if (modelVram) {
    const isMoe = data.is_moe;
    html += `<div style="font-size:11px;color:var(--muted);margin-bottom:14px;padding:10px 14px;background:var(--bg3);border-radius:8px;border:1px solid var(--border2);line-height:1.7;">
      <span style="color:var(--text);font-weight:700">Model weights (${quant}):</span>
      <span style="color:var(--accent);font-weight:800"> ${modelVram} GB</span>
      &nbsp;·&nbsp; KV cache/seq: <span style="color:var(--purple);font-weight:700">${data.kv_per_seq_gb} GB</span>
      ${isMoe ? '&nbsp;·&nbsp;<span style="color:var(--orange);font-weight:700">⚡ MoE — full weights in VRAM, active-param throughput</span>' : ''}
      ${data.model_notes ? `<div style="margin-top:5px;color:var(--muted)">${escHtml(data.model_notes)}</div>` : ''}
    </div>`;
  }

  recs.forEach((r, i) => {
    const isTop = i === 0 && r.fits;
    const cls = isTop ? 'gpu-card top-pick' : r.fits ? 'gpu-card' : 'gpu-card no-fit';
    const tpsColor = r.meets_tps ? 'val-green' : 'val-orange';
    const batchColor = r.meets_batch ? 'val-green' : 'val-red';

    // Pricing row — merge live prices over static estimates if available
    const live = (LIVE_PRICES && LIVE_PRICES[r.gpu]) || {};
    const p = {
      runpod: live.runpod_hr ?? r.pricing?.runpod,
      lambda: live.lambda_hr  ?? r.pricing?.lambda,
      vastai:  live.vastai_hr  ?? r.pricing?.vastai,
    };
    const c = r.cost_per_1m || {};
    const priceBadge = PRICES_ARE_LIVE
      ? `<span class="price-live-badge">LIVE</span>`
      : `<span class="price-est-badge">EST</span>`;
    const priceParts = [];
    if (p.runpod) priceParts.push(`<span style="color:var(--green)">RunPod $${p.runpod.toFixed(2)}</span>`);
    if (p.lambda) priceParts.push(`<span style="color:var(--accent)">Lambda $${p.lambda.toFixed(2)}</span>`);
    if (p.vastai) priceParts.push(`<span style="color:var(--purple)">Vast $${p.vastai.toFixed(2)}</span>`);
    const bestCost = c.runpod || c.lambda || c.vastai;

    html += `<div class="${cls}">
      <div class="gpu-card-header">
        <div>
          <div class="gpu-name">${r.gpu}</div>
          <div style="font-size:10px;color:var(--muted);margin-top:2px">${r.arch || ''} · ${r.tier || ''}</div>
        </div>
        <div style="text-align:right">
          <div style="font-size:11px;font-weight:700">${priceBadge} ${priceParts.join(' <span style="color:var(--muted2)">|</span> ')}/hr</div>
          ${bestCost ? `<div style="font-size:10px;color:var(--muted);margin-top:2px">$${bestCost}/1M tok</div>` : ''}
        </div>
      </div>
      <div class="gpu-stats">
        <div class="gpu-stat">
          <div class="gpu-stat-val ${tpsColor}">${r.est_tps > 0 ? r.est_tps.toLocaleString() : '—'}</div>
          <div class="gpu-stat-label">est. tok/s</div>
        </div>
        <div class="gpu-stat">
          <div class="gpu-stat-val ${batchColor}">${r.max_batch}</div>
          <div class="gpu-stat-label">max seqs</div>
        </div>
        <div class="gpu-stat">
          <div class="gpu-stat-val">${r.vram_gb} GB</div>
          <div class="gpu-stat-label">total VRAM</div>
        </div>
      </div>
      <div style="display:flex;gap:4px;flex-wrap:wrap;align-items:center;margin-top:8px;padding-top:8px;border-top:1px solid var(--border)">
        <span style="font-size:10px;color:var(--muted);flex:1">
          Model: ${r.model_vram_gb}GB · KV budget: ${r.kv_budget_gb}GB
          ${r.min_gpus > 1 ? `· <span style="color:var(--orange)">min ${r.min_gpus}× GPUs</span>` : ''}
        </span>
        ${r.fits ? '<span class="badge badge-green">✓ FITS</span>' : '<span class="badge badge-red">✗ TOO SMALL</span>'}
        ${r.meets_batch ? '<span class="badge badge-blue">BATCH ✓</span>' : '<span class="badge badge-orange">LOW BATCH</span>'}
        ${r.fp8_support ? '<span class="badge badge-purple">FP8</span>' : ''}
        ${r.nvlink ? '<span class="badge badge-blue">NVLink</span>' : ''}
        ${r.mig_slices > 0 ? `<span class="badge badge-orange">MIG×${r.mig_slices}</span>` : ''}
      </div>
      ${r.notes ? `<div style="margin-top:8px;font-size:11px;color:var(--muted);line-height:1.5;border-top:1px solid var(--border);padding-top:8px">${escHtml(r.notes)}</div>` : ''}
    </div>`;
  });

  el.innerHTML = html;
}

// ── CONFIG GENERATOR ──────────────────────────────────────────────────────

function initConfigGen() {
  const modelSel = document.getElementById('cg-model');
  const customRow = document.getElementById('cg-custom-model-row');
  const quantPicker = makeQuantPicker(document.querySelector('#tab-config-gen'));
  const priorityPicker = makePriorityPicker(document.querySelector('#tab-config-gen'));

  modelSel.addEventListener('change', () => {
    customRow.classList.toggle('hidden', modelSel.value !== '__custom__');
  });

  attachHfAutoFetch('cg-custom-model-id', 'cg-custom-params', 'cg-hf-info');

  document.getElementById('cg-submit').addEventListener('click', async () => {
    const modelId = modelSel.value === '__custom__'
      ? document.getElementById('cg-custom-model-id').value.trim()
      : modelSel.value;
    if (!modelId) return showError('cg-results', 'Please select a model.');

    const payload = {
      gpu: document.getElementById('cg-gpu').value,
      model_id: modelId,
      quant: quantPicker.get(),
      context_len: document.getElementById('cg-context').value,
      priority: priorityPicker.get(),
      custom_params_b: document.getElementById('cg-custom-params')?.value || 7,
    };

    showLoading('cg-results');
    const data = await postApi('/api/recommend-config', payload);
    renderConfig(data, 'cg-results', payload);
  });
}

function renderConfig(data, containerId, payload) {
  const el = document.getElementById(containerId);
  if (!data || data.error) { showError(containerId, data?.error || 'Failed'); return; }

  const focus = payload.priority;
  const focusColors = { throughput:'#22d3ee', latency:'#fb923c', memory:'#a78bfa', cost:'#4ade80', balanced:'#22d3ee' };
  const color = focusColors[focus] || '#22d3ee';

  let explanations = {
    throughput: `<strong>--max-num-batched-tokens ${data.max_num_batched_tokens}</strong> packs more prefill tokens per step for better GPU utilization. <strong>--enable-prefix-caching</strong> reuses KV tensors for repeated system prompts. <strong>--scheduler-delay-factor ${data.scheduler_delay_factor}</strong> waits to form larger batches.`,
    latency:    `<strong>--max-num-seqs ${data.max_num_seqs}</strong> limits queueing to reduce wait time. <strong>--max-num-batched-tokens ${data.max_num_batched_tokens}</strong> is lowered to prevent prefill from blocking decode steps. <strong>--enable-prefix-caching</strong> dramatically cuts TTFT for repeated prompts.`,
    cost:       `<strong>--max-num-seqs ${data.max_num_seqs}</strong> maximizes GPU utilization. <strong>--scheduler-delay-factor ${data.scheduler_delay_factor}</strong> forms denser batches for better tokens-per-dollar. <strong>--enable-prefix-caching</strong> reduces compute per request.`,
    balanced:   `Balanced configuration: moderate batch size for reasonable throughput without excessive latency. <strong>--enable-prefix-caching</strong> is always a win.`,
  };

  // Cost table — merge live prices over static estimates
  const liveGpu = (LIVE_PRICES && LIVE_PRICES[payload.gpu]) || {};
  const tpSize = data.tp_size || 1;
  const livePriceRow = (staticDay, liveHr, name) => {
    const hr = liveHr != null ? liveHr * tpSize : (staticDay != null ? staticDay / 24 : null);
    const day = hr != null ? hr * 24 : null;
    const tps = data.est_tps || 0;
    const per1m = (hr && tps > 0) ? Math.round(hr / tps * 1e6 / 3600 * 10000) / 10000 : null;
    return { name, hr, day, per1m, isLive: liveHr != null };
  };
  const costRows = [
    livePriceRow(data.cost_per_day?.runpod, liveGpu.runpod_hr, 'RunPod'),
    livePriceRow(data.cost_per_day?.lambda, liveGpu.lambda_hr, 'Lambda Labs'),
    livePriceRow(data.cost_per_day?.vastai, liveGpu.vastai_hr, 'Vast.ai'),
  ].filter(r => r.day != null);

  let html = `
    <div class="stats-grid">
      <div class="stat-card">
        <div class="stat-card-val" style="color:${color}">${data.est_tps?.toLocaleString() || '—'}</div>
        <div class="stat-card-label">Est. tok/s</div>
      </div>
      <div class="stat-card">
        <div class="stat-card-val" style="color:var(--green)">${data.max_batch_possible}</div>
        <div class="stat-card-label">Max concurrent seqs</div>
      </div>
      <div class="stat-card">
        <div class="stat-card-val" style="color:var(--purple)">${data.model_vram_gb} GB</div>
        <div class="stat-card-label">Model VRAM</div>
      </div>
      <div class="stat-card">
        <div class="stat-card-val" style="color:var(--orange)">${data.tp_size}×</div>
        <div class="stat-card-label">Tensor parallel</div>
      </div>
    </div>`;

  // Provider cost comparison
  if (costRows.length) {
    html += `<div style="margin-bottom:16px">
      <div class="config-label" style="margin-bottom:8px">PROVIDER COST COMPARISON</div>
      <div style="display:grid;grid-template-columns:repeat(${costRows.length},1fr);gap:8px">
        ${costRows.map(r => `
          <div style="background:var(--bg3);border:1px solid var(--border2);border-radius:9px;padding:12px 14px">
            <div style="font-size:10px;color:var(--muted);font-weight:700;letter-spacing:0.06em;margin-bottom:8px">${r.name.toUpperCase()} ${r.isLive ? '<span class="price-live-badge">LIVE</span>' : '<span class="price-est-badge">EST</span>'}</div>
            <div style="font-size:18px;font-weight:800;color:var(--green)">$${r.day.toFixed(2)}<span style="font-size:11px;font-weight:400;color:var(--muted)">/day</span></div>
            ${r.per1m != null ? `<div style="font-size:11px;color:var(--muted);margin-top:4px">$${r.per1m} / 1M tokens</div>` : ''}
          </div>`).join('')}
      </div>
    </div>`;
  }

  if (data.model_notes) {
    html += `<div style="padding:8px 14px;background:rgba(167,139,250,0.06);border:1px solid rgba(167,139,250,0.15);border-radius:8px;margin-bottom:14px;font-size:11px;color:var(--muted)">
      📋 ${escHtml(data.model_notes)}
    </div>`;
  }

  if (data.suggested_quant && data.suggested_quant !== payload.quant) {
    html += `<div style="padding:10px 14px;background:rgba(251,146,60,0.08);border:1px solid rgba(251,146,60,0.2);border-radius:8px;margin-bottom:14px;font-size:12px;color:var(--orange);">
      💡 <strong>Suggestion:</strong> Switch to <code>${data.suggested_quant}</code> on this GPU for better throughput (your GPU supports it natively).
    </div>`;
  }

  html += `<div class="config-block">
    <div class="config-label">VLLM SERVE COMMAND</div>
    <div class="command-box" id="cmd-box">${escHtml(data.command)}<button class="copy-btn" onclick="copyCmd()">copy</button></div>
  </div>`;

  html += `<div class="explanation-box">${explanations[focus] || explanations.balanced}</div>`;

  // Param breakdown
  const params = [
    { k:'tensor-parallel-size', v: data.tp_size },
    { k:'quantization',         v: data.quantization || 'none (fp16)' },
    { k:'max-num-seqs',         v: data.max_num_seqs },
    { k:'max-num-batched-tokens', v: data.max_num_batched_tokens },
    { k:'gpu-memory-utilization', v: data.gpu_memory_utilization },
    { k:'enable-prefix-caching',  v: data.enable_prefix_caching ? 'true' : 'false' },
    { k:'kv-cache-dtype',         v: data.kv_cache_dtype },
    { k:'scheduler-delay-factor', v: data.scheduler_delay_factor },
  ];

  html += `<div style="margin-top:14px">
    <div class="config-label" style="margin-bottom:8px">PARAMETER BREAKDOWN</div>
    ${params.map(p => `
      <div style="display:flex;justify-content:space-between;padding:6px 10px;border-radius:5px;margin-bottom:3px;background:var(--bg3);font-size:11px;">
        <span style="color:var(--muted)">--${p.k}</span>
        <code style="color:${color}">${p.v}</code>
      </div>`).join('')}
  </div>`;

  el.innerHTML = html;
  window._lastCmd = data.command;
}

window.copyCmd = function() {
  if (window._lastCmd) {
    navigator.clipboard.writeText(window._lastCmd.replace(/\\\n\s+/g, ' '));
    document.querySelector('.copy-btn').textContent = '✓ copied';
    setTimeout(() => { const b = document.querySelector('.copy-btn'); if(b) b.textContent = 'copy'; }, 1500);
  }
};

// ── CALCULATOR ────────────────────────────────────────────────────────────

function initCalculator() {
  const utilSlider = document.getElementById('calc-util');
  const utilVal = document.getElementById('calc-util-val');
  utilSlider.addEventListener('input', () => { utilVal.textContent = parseFloat(utilSlider.value).toFixed(2); });

  document.getElementById('calc-submit').addEventListener('click', async () => {
    const payload = {
      params_b:    document.getElementById('calc-params').value,
      quant:       document.getElementById('calc-quant').value,
      context_len: document.getElementById('calc-context').value,
      num_layers:  document.getElementById('calc-layers').value,
      kv_heads:    document.getElementById('calc-kv-heads').value,
      head_dim:    document.getElementById('calc-head-dim').value,
      gpu_vram:    document.getElementById('calc-vram').value,
      num_gpus:    document.getElementById('calc-ngpus').value,
      gpu_util:    document.getElementById('calc-util').value,
      kv_dtype:    document.getElementById('calc-kv-dtype').value,
    };
    showLoading('calc-results');
    const data = await postApi('/api/calculate', payload);
    renderCalcResults(data, payload);
  });

  // Quick-fill model buttons
  document.querySelectorAll('.pill-btn').forEach(btn => {
    btn.addEventListener('click', async () => {
      const models = await (await fetch('/api/models')).json();
      const m = models[btn.dataset.model];
      if (!m) return;
      document.getElementById('calc-params').value = m.params_b;
      document.getElementById('calc-layers').value = m.layers;
      document.getElementById('calc-kv-heads').value = m.kv_heads;
      document.getElementById('calc-head-dim').value = m.head_dim;
    });
  });
}

function renderCalcResults(data, payload) {
  if (!data || data.error) { showError('calc-results', data?.error || 'Failed'); return; }

  const totalVram = parseFloat(payload.gpu_vram) * parseInt(payload.num_gpus);
  const modelPct = Math.min(100, (data.model_vram_gb / totalVram) * 100);
  const kvPct    = Math.min(100, ((totalVram * parseFloat(payload.gpu_util) - data.model_vram_gb) / totalVram) * 100);
  const fitColor = data.fits_on_gpu ? 'val-green' : 'val-red';

  const html = `
    <div class="calc-results-grid">
      <div class="calc-row">
        <span class="calc-row-label">Model weights VRAM</span>
        <span class="calc-row-val val-purple">${data.model_vram_gb} GB</span>
      </div>
      <div class="calc-row">
        <span class="calc-row-label">KV cache per sequence</span>
        <span class="calc-row-val val-blue">${data.kv_per_seq_gb} GB</span>
      </div>
      <div class="calc-row">
        <span class="calc-row-label">Total GPU VRAM</span>
        <span class="calc-row-val">${data.total_vram_gb} GB</span>
      </div>
      <div class="calc-row">
        <span class="calc-row-label">Available for KV cache</span>
        <span class="calc-row-val ${data.available_for_kv_gb > 0 ? 'val-green' : 'val-red'}">${data.available_for_kv_gb} GB</span>
      </div>
      <div class="calc-row" style="border-color:${data.fits_on_gpu ? 'var(--green)' : 'var(--red)'};">
        <span class="calc-row-label">Model fits on GPU?</span>
        <span class="calc-row-val ${fitColor}">${data.fits_on_gpu ? '✓ YES' : '✗ NO — need more VRAM or quantization'}</span>
      </div>
      <div class="calc-row" style="border-color:var(--accent);">
        <span class="calc-row-label">Max concurrent sequences</span>
        <span class="calc-row-val val-blue" style="font-size:24px">${data.max_concurrent_seqs}</span>
      </div>
    </div>

    <div style="margin-top:16px">
      <div class="config-label" style="margin-bottom:8px">VRAM BREAKDOWN</div>
      <div class="mem-bar-wrap">
        <div style="display:flex;height:20px;border-radius:6px;overflow:hidden;gap:2px;margin-bottom:6px">
          <div style="width:${modelPct.toFixed(1)}%;background:var(--purple);display:flex;align-items:center;justify-content:center;font-size:9px;font-weight:700;color:#000;min-width:30px">${modelPct.toFixed(0)}%</div>
          <div style="width:${(kvPct).toFixed(1)}%;background:var(--accent);display:flex;align-items:center;justify-content:center;font-size:9px;font-weight:700;color:#000;min-width:30px">${kvPct.toFixed(0)}%</div>
          <div style="flex:1;background:var(--bg3)"></div>
        </div>
        <div class="mem-bar-labels">
          <span style="color:var(--purple)">■ Model weights (${data.model_vram_gb}GB)</span>
          <span style="color:var(--accent)">■ KV cache budget (${data.available_for_kv_gb}GB)</span>
          <span>■ Reserved/Other</span>
        </div>
      </div>
    </div>

    <div class="explanation-box" style="margin-top:14px">
      <strong>KV cache formula:</strong><br>
      2 × layers × kv_heads × head_dim × context_len × bytes/elem<br>
      = 2 × ${payload.num_layers} × ${payload.kv_heads} × ${payload.head_dim} × ${payload.context_len} × ${payload.kv_dtype === 'fp8' ? 1 : 2}<br>
      = <strong style="color:var(--accent)">${data.kv_per_seq_gb} GB per sequence</strong><br><br>
      <strong>Max sequences:</strong> ${data.available_for_kv_gb}GB ÷ ${data.kv_per_seq_gb}GB = <strong style="color:var(--green)">${data.max_concurrent_seqs}</strong>
    </div>`;

  document.getElementById('calc-results').innerHTML = html;
}

// ── PARAM REFERENCE ──────────────────────────────────────────────────────

function initParamRef() {
  const container = document.getElementById('param-ref-app');
  let activeFocus = 'throughput';
  let activeCategory = 'All';
  let search = '';
  let expanded = null;
  let impactFilter = 'all';
  let showAll = false;

  function render() {
    const fm = window.FOCUS_META[activeFocus];
    const filtered = window.VLLM_PARAMS.filter(p => {
      const matchFocus = showAll || p.focus.includes(activeFocus);
      const matchCat = activeCategory === 'All' || p.category === activeCategory;
      const matchImp = impactFilter === 'all' || p.impact === impactFilter;
      const q = search.toLowerCase();
      const matchSearch = !q || p.name.includes(q) || p.summary.toLowerCase().includes(q) || p.category.toLowerCase().includes(q);
      return matchFocus && matchCat && matchImp && matchSearch;
    }).sort((a,b) => { const o={critical:0,high:1,medium:2,low:3}; return o[a.impact]-o[b.impact]; });

    let html = `
      <div class="focus-tabs">
        ${Object.entries(window.FOCUS_META).map(([k,f]) => `
          <button class="focus-tab ${k === activeFocus ? 'active' : ''}" data-focus="${k}" style="${k===activeFocus ? `border-color:${f.color};color:${f.color};background:${f.bg}` : ''}">
            ${f.icon} ${f.label}
          </button>`).join('')}
      </div>

      <div class="focus-banner" style="border-left-color:${fm.color};background:${fm.bg};color:${fm.color}">
        <span>${fm.icon} <strong>${fm.label}</strong> — ${fm.desc}</span>
        <span class="focus-banner-count">${filtered.length} params</span>
      </div>

      <div class="param-filters">
        <input class="param-search" placeholder="Search params..." value="${escHtml(search)}" id="pref-search">
        <div class="cat-pills">
          ${window.CATEGORIES.map(c => `<button class="cat-pill ${c===activeCategory?'active':''}" data-cat="${c}">${c}</button>`).join('')}
        </div>
        <div style="display:flex;gap:4px;margin-left:auto;align-items:center">
          ${['all','critical','high','medium','low'].map(i => {
            const s = i==='all' ? {color:'#94a3b8'} : window.IMPACT_META[i];
            return `<button class="cat-pill ${i===impactFilter?'active':''}" data-impact="${i}" style="${i===impactFilter?`border-color:${s.color};color:${s.color}`:''}">${i.toUpperCase()}</button>`;
          }).join('')}
          <label style="font-size:10px;color:var(--muted);margin-left:8px;cursor:pointer;display:flex;align-items:center;gap:4px">
            <input type="checkbox" id="pref-showall" ${showAll?'checked':''}> All
          </label>
        </div>
      </div>

      <div id="param-list">
        ${filtered.map(p => renderParamCard(p, activeFocus, expanded === p.name)).join('')}
        ${filtered.length === 0 ? '<div class="empty-state">No parameters match</div>' : ''}
      </div>

      <div class="recipe-strip">
        <div class="recipe-strip-label" style="color:${fm.color}">${fm.icon} ${fm.label.toUpperCase()} — RECOMMENDED CONFIG</div>
        <div class="recipe-code" style="border-left-color:${fm.color}">${escHtml(fm.recipe)}</div>
      </div>`;

    container.innerHTML = html;

    // Bind events
    container.querySelectorAll('.focus-tab').forEach(b => b.addEventListener('click', () => { activeFocus = b.dataset.focus; expanded=null; render(); }));
    container.querySelectorAll('.cat-pill[data-cat]').forEach(b => b.addEventListener('click', () => { activeCategory = b.dataset.cat; render(); }));
    container.querySelectorAll('.cat-pill[data-impact]').forEach(b => b.addEventListener('click', () => { impactFilter = b.dataset.impact; render(); }));
    container.querySelector('#pref-search').addEventListener('input', e => { search = e.target.value; render(); });
    container.querySelector('#pref-showall').addEventListener('change', e => { showAll = e.target.checked; render(); });
    container.querySelectorAll('.param-card-header').forEach(h => {
      h.addEventListener('click', () => {
        const name = h.dataset.name;
        expanded = expanded === name ? null : name;
        render();
      });
    });
  }

  render();
}

function renderParamCard(p, focus, isOpen) {
  const fm = window.FOCUS_META[focus];
  const im = window.IMPACT_META[p.impact];
  const whenText = p.when[focus] || '';

  return `
    <div class="param-card ${isOpen ? 'open' : ''}">
      <div class="param-card-header" data-name="${escHtml(p.name)}">
        <div class="param-impact-bar" style="background:${im.color}"></div>
        <div style="flex:1;min-width:0">
          <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap">
            <span class="param-name" style="color:${fm.color}">${escHtml(p.name)}</span>
            ${p.alias ? `<span class="param-alias">${escHtml(p.alias)}</span>` : ''}
            <span style="padding:2px 7px;border-radius:4px;font-size:9px;font-weight:800;background:${im.bg};color:${im.color}">${im.label}</span>
            <span style="padding:2px 7px;border-radius:4px;font-size:9px;background:var(--bg3);color:var(--muted)">${escHtml(p.category)}</span>
            ${p.focus.filter(f => f !== focus).map(f => `<span style="padding:2px 6px;border-radius:4px;font-size:9px;background:${FOCUS_META[f].bg};color:${FOCUS_META[f].color}">${FOCUS_META[f].icon}</span>`).join('')}
          </div>
          <div class="param-summary">${escHtml(p.summary)}</div>
          ${whenText && !isOpen ? `<div class="param-focus-when" style="background:${fm.bg};color:${fm.color}">${fm.icon} ${escHtml(whenText)}</div>` : ''}
        </div>
        <div class="param-chevron">${isOpen ? '▲' : '▼'}</div>
      </div>
      ${isOpen ? renderParamDetail(p, focus, fm) : ''}
    </div>`;
}

function renderParamDetail(p, focus, fm) {
  return `
    <div class="param-detail">
      <div>
        <div class="detail-section">
          <div class="detail-title" style="color:#38bdf8">HOW IT WORKS</div>
          <div class="detail-text">${escHtml(p.detail)}</div>
        </div>
        <div class="detail-section">
          <div class="detail-title" style="color:#f59e0b">TRADEOFFS</div>
          ${p.tradeoffs.map(t => `<div class="tradeoff-row"><span class="tradeoff-label">${escHtml(t.l)}</span><span class="tradeoff-val">${escHtml(t.v)}</span></div>`).join('')}
        </div>
      </div>
      <div>
        <div class="detail-section">
          <div class="detail-title" style="color:${fm.color}">${fm.icon} WHEN TO USE — ${fm.label.toUpperCase()}</div>
          <div style="padding:8px 10px;border-radius:6px;background:${fm.bg};color:${fm.color};font-size:12px;line-height:1.7">${escHtml(p.when[focus] || '—')}</div>
          <div style="margin-top:10px">
            ${Object.entries(p.when).filter(([k]) => k !== focus).map(([k,v]) => {
              const f = window.FOCUS_META[k];
              return `<div class="recipe-row"><span style="color:${f.color};font-size:10px;font-weight:700">${f.icon} ${f.label}: </span><span style="font-size:11px;color:var(--muted)">${escHtml(v)}</span></div>`;
            }).join('')}
          </div>
        </div>
        <div class="detail-section">
          <div class="detail-title" style="color:#a78bfa">DEFAULT & OPTIONS</div>
          <div style="font-size:12px"><span style="color:var(--muted)">default: </span><code style="color:#c4b5fd">${escHtml(p.default)}</code></div>
          ${p.options ? `<div style="margin-top:4px;font-size:11px;color:var(--muted2)">options: ${escHtml(p.options)}</div>` : ''}
        </div>
        <div class="detail-section">
          <div class="detail-title" style="color:#34d399">EXAMPLE</div>
          <div class="example-code">${escHtml(p.example)}</div>
        </div>
        ${p.recipe && Object.keys(p.recipe).length ? `
        <div class="detail-section">
          <div class="detail-title" style="color:#f472b6">RECIPES BY GOAL</div>
          ${Object.entries(p.recipe).map(([k,v]) => {
            const f = window.FOCUS_META[k];
            return `<div class="recipe-row"><span style="color:${f.color};font-size:10px;font-weight:700">${f.icon} ${f.label.toUpperCase()}: </span><code style="font-size:11px;color:var(--green)">${escHtml(v)}</code></div>`;
          }).join('')}
        </div>` : ''}
      </div>
    </div>`;
}

// ── CHAT ─────────────────────────────────────────────────────────────────

const CHAT_KEY     = 'vllm_chat_history';
const CHAT_MAX_MSG = 60;  // max messages stored (30 turns)

function _loadChatMessages() {
  try { return JSON.parse(localStorage.getItem(CHAT_KEY) || '[]'); } catch { return []; }
}
function _saveChatMessages(msgs) {
  const trimmed = msgs.slice(-CHAT_MAX_MSG);
  try { localStorage.setItem(CHAT_KEY, JSON.stringify(trimmed)); } catch {}
}

window.newChat = function() {
  localStorage.removeItem(CHAT_KEY);
  const chatEl = document.getElementById('chatMessages');
  chatEl.innerHTML = `<div class="chat-welcome" id="chatWelcome">
    <div class="welcome-icon">🤖</div>
    <h3>vLLM Infrastructure Advisor</h3>
    <p>I can help you choose the right GPU, generate optimal configs, explain the math, and debug performance issues. Ask me anything.</p>
    <p class="api-note" id="apiNote"></p>
  </div>`;
  _updateTurnCount(0);
  // Re-init so messages array is cleared
  _chatMessages.length = 0;
};

// Module-level array so newChat() can clear it
const _chatMessages = _loadChatMessages();

function _updateTurnCount(n) {
  const el = document.getElementById('chatTurnCount');
  if (!el) return;
  el.textContent = n > 0 ? `${Math.ceil(n / 2)} turn${Math.ceil(n / 2) !== 1 ? 's' : ''}` : '';
}

function initChat() {
  const chatEl  = document.getElementById('chatMessages');
  const inputEl = document.getElementById('chatInput');
  const sendBtn = document.getElementById('chatSend');

  // Restore prior session
  if (_chatMessages.length > 0) {
    const welcome = document.getElementById('chatWelcome');
    if (welcome) welcome.remove();
    _chatMessages.forEach(m => appendMsg(m.role, m.content, chatEl));
    _updateTurnCount(_chatMessages.length);
    chatEl.scrollTop = chatEl.scrollHeight;
  }

  document.querySelectorAll('.qp-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      inputEl.value = btn.dataset.prompt;
      inputEl.focus();
    });
  });

  sendBtn.addEventListener('click', sendMessage);
  inputEl.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
  });

  async function sendMessage() {
    const text = inputEl.value.trim();
    if (!text) return;
    inputEl.value = '';
    sendBtn.disabled = true;

    // Hide welcome screen on first message
    const welcome = document.getElementById('chatWelcome');
    if (welcome) welcome.remove();

    _chatMessages.push({ role: 'user', content: text });
    appendMsg('user', text, chatEl);

    const assistantDiv = appendMsg('assistant', '', chatEl);
    const bubble = assistantDiv.querySelector('.msg-bubble');
    bubble.classList.add('streaming-cursor');

    let fullText = '';

    try {
      const resp = await fetch('/api/advisor', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ messages: _chatMessages }),
      });

      const reader = resp.body.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        const lines = decoder.decode(value).split('\n');
        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          const payload = line.slice(6);
          if (payload === '[DONE]') break;
          try {
            const { text: t } = JSON.parse(payload);
            fullText += t;
            bubble.innerHTML = renderMarkdown(fullText);
            bubble.classList.add('streaming-cursor');
            chatEl.scrollTop = chatEl.scrollHeight;
          } catch {}
        }
      }
    } catch (err) {
      fullText = `❌ Error: ${err.message}`;
    }

    bubble.classList.remove('streaming-cursor');
    bubble.innerHTML = renderMarkdown(fullText || '(no response)');

    _chatMessages.push({ role: 'assistant', content: fullText });
    _saveChatMessages(_chatMessages);
    _updateTurnCount(_chatMessages.length);

    sendBtn.disabled = false;
    chatEl.scrollTop = chatEl.scrollHeight;
  }
}

function appendMsg(role, text, container) {
  const div = document.createElement('div');
  div.className = `chat-msg ${role}`;
  div.innerHTML = `<div class="msg-bubble">${role === 'user' ? escHtml(text) : renderMarkdown(text)}</div>`;
  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
  return div;
}

// Simple markdown renderer
function renderMarkdown(text) {
  return text
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    // Code blocks
    .replace(/```(\w*)\n([\s\S]*?)```/g, (_, lang, code) =>
      `<pre><code>${code.trim()}</code></pre>`)
    // Inline code
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    // Bold
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    // H3
    .replace(/^### (.+)$/gm, '<h3>$1</h3>')
    // H2
    .replace(/^## (.+)$/gm, '<h2>$1</h2>')
    // H1
    .replace(/^# (.+)$/gm, '<h1>$1</h1>')
    // Lists
    .replace(/^[-*] (.+)$/gm, '<li>$1</li>')
    .replace(/(<li>[\s\S]*?<\/li>)/g, '<ul>$1</ul>')
    .replace(/<\/ul>\s*<ul>/g, '')
    // Ordered lists
    .replace(/^\d+\. (.+)$/gm, '<li>$1</li>')
    // Tables
    .replace(/\|(.+)\|\n\|[-| :]+\|\n((?:\|.+\|\n?)*)/g, (_, header, rows) => {
      const ths = header.split('|').filter(s => s.trim()).map(s => `<th>${s.trim()}</th>`).join('');
      const trs = rows.trim().split('\n').map(row => {
        const tds = row.split('|').filter(s => s.trim()).map(s => `<td>${s.trim()}</td>`).join('');
        return `<tr>${tds}</tr>`;
      }).join('');
      return `<table><thead><tr>${ths}</tr></thead><tbody>${trs}</tbody></table>`;
    })
    // Paragraphs (double newline)
    .replace(/\n\n/g, '</p><p>')
    .replace(/^(.+)$/, '<p>$1</p>')
    // Single newlines
    .replace(/\n/g, '<br>');
}

// ── UTILITIES ─────────────────────────────────────────────────────────────

async function postApi(url, payload) {
  try {
    const r = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    return await r.json();
  } catch (e) {
    return { error: e.message };
  }
}

function showLoading(id) {
  document.getElementById(id).innerHTML = `<div style="text-align:center;padding:30px 0;color:var(--muted)"><div class="loading-spinner"></div> Calculating...</div>`;
}

function showError(id, msg) {
  document.getElementById(id).innerHTML = `<div style="padding:14px;background:rgba(248,113,113,0.08);border:1px solid rgba(248,113,113,0.2);border-radius:8px;color:var(--red);font-size:12px;">⚠️ ${escHtml(msg)}</div>`;
}

function escHtml(str) {
  return String(str ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}
