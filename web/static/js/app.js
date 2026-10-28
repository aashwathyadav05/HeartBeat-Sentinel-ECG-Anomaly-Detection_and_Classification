// HeartBeat Sentinel frontend
// Unified PTB-XL and legacy ECG5000 workflows share one page but never share inputs.
const API_BASE = '';
const LEAD_ORDER = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'];
const PLOT_BG = '#162033';
const PAPER_BG = '#162033';
const GRID = '#2b3a52';
const TEXT = '#eef2f7';
const MUTED = '#a9b7ca';
const CATEGORY_COLORS = {
    NORM: '#34d399',
    MI: '#f87171',
    STTC: '#fbbf24',
    CD: '#a78bfa',
    HYP: '#22d3ee',
};

let unifiedAvailable = false;
let legacyAvailable = false;
let currentUnifiedResult = null;
let currentUnifiedSignal = null;

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------
function escapeHTML(value) {
    return String(value)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

function setText(id, value) {
    const node = document.getElementById(id);
    if (node) node.textContent = value;
}

function showError(id, message) {
    const node = document.getElementById(id);
    if (!node) return;
    node.textContent = message || 'Unknown error';
    node.style.display = 'block';
}

function hideError(id) {
    const node = document.getElementById(id);
    if (!node) return;
    node.textContent = '';
    node.style.display = 'none';
}

function formatNumber(value, digits = 6) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) return '—';
    return Number(value).toFixed(digits);
}

function formatMetric(value, digits = 3) {
    if (value === null || value === undefined) return 'undefined';
    const number = Number(value);
    return Number.isFinite(number) ? number.toFixed(digits) : 'undefined';
}

async function fetchJSON(url, options = {}) {
    const response = await fetch(API_BASE + url, options);
    let payload = null;
    try {
        payload = await response.json();
    } catch (error) {
        throw new Error(`Server returned a non-JSON response (${response.status}).`);
    }
    if (!response.ok) {
        throw new Error(payload && payload.error ? payload.error : `Request failed (${response.status}).`);
    }
    return payload;
}

function darkLayout(title, xTitle, yTitle) {
    return {
        title: { text: title, font: { color: TEXT, size: 14 } },
        paper_bgcolor: PAPER_BG,
        plot_bgcolor: PAPER_BG,
        font: { color: TEXT, size: 11 },
        margin: { l: 58, r: 20, t: 48, b: 46 },
        xaxis: { title: xTitle, gridcolor: GRID, zerolinecolor: GRID },
        yaxis: { title: yTitle, gridcolor: GRID, zerolinecolor: GRID },
        legend: { orientation: 'h', y: 1.12, font: { color: MUTED } },
        hovermode: 'x unified',
    };
}

function plot(target, traces, layout) {
    const node = document.getElementById(target);
    if (!node) return;
    if (!window.Plotly) {
        node.innerHTML = '<p class="muted">Plot library unavailable. Waveforms were not rendered.</p>';
        return;
    }
    Plotly.newPlot(target, traces, layout, { responsive: true, displaylogo: false });
}

// ---------------------------------------------------------------------------
// Navigation
// ---------------------------------------------------------------------------
document.querySelectorAll('.nav-btn').forEach((button) => {
    button.addEventListener('click', () => {
        document.querySelectorAll('.nav-btn').forEach((candidate) => {
            candidate.classList.remove('active');
            candidate.setAttribute('aria-selected', 'false');
        });
        document.querySelectorAll('.tab-content').forEach((section) => section.classList.remove('active'));
        button.classList.add('active');
        button.setAttribute('aria-selected', 'true');
        const target = document.getElementById('tab-' + button.dataset.tab);
        if (target) target.classList.add('active');
        if (button.dataset.tab === 'evaluation') loadEvaluation();
        if (button.dataset.tab === 'explorer') loadLegacyExamples();
    });
});

// ---------------------------------------------------------------------------
// Unified PTB-XL status and examples
// ---------------------------------------------------------------------------
async function loadUnifiedHealth() {
    const banner = document.getElementById('unified-status');
    const chip = document.getElementById('header-model-status');
    const button = document.getElementById('unified-predict-btn');
    try {
        const health = await fetchJSON('/api/unified/health');
        if (health.available) {
            unifiedAvailable = true;
            if (chip) {
                chip.className = 'status-chip status-chip-ok';
                chip.textContent = 'PTB-XL unified model ready';
            }
            if (banner) {
                banner.className = 'status-banner status-banner-ok';
                banner.innerHTML = `<strong>Unified model loaded.</strong> Version ${escapeHTML(health.model_version || 'unknown')} · ` +
                    `anomaly threshold ${formatNumber(health.anomaly_threshold, 5)} · ` +
                    `five validation-tuned category thresholds.`;
            }
            setText('artifact-version', `Artifact: ${health.model_version || 'loaded'}`);
            if (button) button.disabled = false;
            const threshold = document.getElementById('unified-anomaly-threshold');
            if (threshold && threshold.textContent === '—') threshold.textContent = formatNumber(health.anomaly_threshold, 6);
        } else {
            unifiedAvailable = false;
            if (chip) {
                chip.className = 'status-chip status-chip-warn';
                chip.textContent = 'PTB-XL unified model not trained';
            }
            if (banner) {
                banner.className = 'status-banner status-banner-warn';
                banner.innerHTML = '<strong>Unified PTB-XL model is not available.</strong> ' +
                    'The condition guide works, but integrated analysis requires a trained checkpoint, ' +
                    'saved thresholds and artifact metadata. Train it with ' +
                    '<code>python -m src.train_unified_model all --data-dir data/ptbxl</code>. ' +
                    'The legacy ECG5000 workflow remains available separately.';
                if (health.error) {
                    const detail = document.createElement('div');
                    detail.className = 'field-hint';
                    detail.textContent = health.error;
                    banner.appendChild(detail);
                }
            }
            setText('artifact-version', 'Artifact: not available');
            if (button) button.disabled = true;
        }
    } catch (error) {
        unifiedAvailable = false;
        if (chip) {
            chip.className = 'status-chip status-chip-error';
            chip.textContent = 'Status check failed';
        }
        if (banner) {
            banner.className = 'status-banner status-banner-error';
            banner.textContent = 'Could not check the unified model: ' + error.message;
        }
        if (button) button.disabled = true;
    }
}

async function loadUnifiedExamples() {
    const select = document.getElementById('unified-example-select');
    const note = document.getElementById('unified-example-note');
    if (!select) return;
    try {
        const payload = await fetchJSON('/api/unified/examples');
        select.innerHTML = '<option value="">-- Select a test record --</option>';
        if (payload.examples && payload.examples.length) {
            payload.examples.forEach((example) => {
                const option = document.createElement('option');
                option.value = example.ecg_id;
                const labels = (example.true_categories || []).join(', ') || 'no mapped label';
                option.textContent = `ecg_id ${example.ecg_id} — dataset labels: ${labels}`;
                select.appendChild(option);
            });
            if (note) note.textContent = payload.message || '';
        } else if (note) {
            note.textContent = payload.message || 'No PTB-XL test examples available (dataset not present).';
        }
    } catch (error) {
        if (note) note.textContent = 'Could not load PTB-XL examples: ' + error.message;
    }
}

function parsePastedSignal() {
    const node = document.getElementById('unified-signal-input');
    if (!node || !node.value.trim()) return null;
    let parsed;
    try {
        parsed = JSON.parse(node.value);
    } catch (error) {
        throw new Error('Pasted input is not valid JSON. Provide a 12×1000 JSON array.');
    }
    if (parsed && !Array.isArray(parsed) && Array.isArray(parsed.signal)) parsed = parsed.signal;
    if (!Array.isArray(parsed) || parsed.length !== 12) {
        throw new Error('Pasted JSON must contain exactly 12 lead rows.');
    }
    parsed.forEach((row, index) => {
        if (!Array.isArray(row) || row.length !== 1000) {
            throw new Error(`Lead row ${index + 1} must contain exactly 1000 numeric samples.`);
        }
    });
    return parsed;
}

async function runUnifiedAnalysis() {
    hideError('unified-error');
    const button = document.getElementById('unified-predict-btn');
    const fileInput = document.getElementById('unified-file');
    const exampleSelect = document.getElementById('unified-example-select');
    const previousLabel = button ? button.textContent : 'Run Integrated Analysis';
    if (!unifiedAvailable) {
        showError('unified-error', 'The unified model is not available. Train it and reload this page.');
        return;
    }
    if (button) {
        button.disabled = true;
        button.textContent = 'Analyzing…';
    }
    try {
        let payload;
        if (fileInput && fileInput.files && fileInput.files[0]) {
            const form = new FormData();
            form.append('file', fileInput.files[0]);
            payload = await fetchJSON('/api/unified/predict-file', { method: 'POST', body: form });
        } else if (exampleSelect && exampleSelect.value) {
            payload = await fetchJSON('/api/unified/example/' + encodeURIComponent(exampleSelect.value));
        } else {
            const signal = parsePastedSignal();
            if (!signal) throw new Error('Upload a file, select a PTB-XL test record, or paste a 12×1000 JSON array.');
            payload = await fetchJSON('/api/unified/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ signal, sampling_rate: 100 }),
            });
        }
        renderUnifiedResult(payload);
    } catch (error) {
        showError('unified-error', error.message);
    } finally {
        if (button) {
            button.disabled = !unifiedAvailable;
            button.textContent = previousLabel;
        }
    }
}

function renderConditionRows(result) {
    const container = document.getElementById('unified-condition-results');
    if (!container) return;
    const items = (result.conditions && result.conditions.items) || [];
    container.innerHTML = '';
    items.forEach((item) => {
        const score = Number(item.score);
        const threshold = Number(item.threshold);
        const positive = Boolean(item.predicted);
        const row = document.createElement('div');
        row.className = 'condition-result-row' + (positive ? ' is-positive' : '');
        const code = document.createElement('div');
        code.className = 'condition-result-code';
        code.textContent = item.category;
        code.style.color = CATEGORY_COLORS[item.category] || TEXT;
        const barWrap = document.createElement('div');
        barWrap.className = 'condition-result-bar-wrap';
        const track = document.createElement('div');
        track.className = 'condition-bar-track';
        const fill = document.createElement('div');
        fill.className = 'condition-bar-fill';
        fill.style.width = `${Math.max(0, Math.min(1, score)) * 100}%`;
        track.appendChild(fill);
        const meta = document.createElement('div');
        meta.className = 'condition-meta';
        const scoreLabel = document.createElement('span');
        scoreLabel.textContent = `score ${formatNumber(score, 4)}`;
        const thresholdLabel = document.createElement('span');
        thresholdLabel.textContent = `threshold ${formatNumber(threshold, 4)}`;
        meta.appendChild(scoreLabel);
        meta.appendChild(thresholdLabel);
        barWrap.appendChild(track);
        barWrap.appendChild(meta);
        const status = document.createElement('div');
        status.className = 'condition-result-status' + (positive ? ' is-positive' : '');
        status.textContent = positive ? 'ABOVE' : 'below';
        row.appendChild(code);
        row.appendChild(barWrap);
        row.appendChild(status);
        container.appendChild(row);
    });

    const summary = document.getElementById('unified-predicted-summary');
    const conflict = document.getElementById('unified-classifier-conflict');
    const flagged = (result.predicted_categories || []).map(String);
    if (conflict) {
        const hasNorm = flagged.includes('NORM');
        const hasAbnormal = flagged.some((category) => category !== 'NORM');
        conflict.hidden = !(hasNorm && hasAbnormal);
        conflict.textContent = hasNorm && hasAbnormal
            ? 'Conflicting model flags: NORM and an abnormal category are both above threshold. This can be a classifier false positive; neither output is a confirmed diagnosis.'
            : '';
    }
    if (summary) {
        const predicted = flagged;
        if (predicted.length) {
            summary.innerHTML = '<strong>Flagged categories:</strong> ' + predicted.map(escapeHTML).join(', ') +
                '. These are model outputs, not confirmed diagnoses.';
        } else {
            summary.innerHTML = '<strong>No condition category reached its frozen threshold.</strong> ' +
                'This does not rule out every possible heart condition.';
        }
    }
}

function plotUnifiedPlots() {
    if (!currentUnifiedResult) return;
    const reconstruction = currentUnifiedResult.reconstruction || {};
    const original = reconstruction.normalized_input;
    const rebuilt = reconstruction.reconstructed;
    const error = reconstruction.pointwise_squared_error;
    if (!original || !rebuilt || !error) return;
    const leadSelect = document.getElementById('unified-lead-select');
    const leadName = leadSelect ? leadSelect.value : 'II';
    const leadIndex = Math.max(0, LEAD_ORDER.indexOf(leadName));
    const x = Array.from({ length: original[leadIndex].length }, (_, index) => index);

    plot('plot-unified-reconstruction', [
        { x, y: original[leadIndex], type: 'scatter', mode: 'lines', name: 'Normalized input', line: { color: '#60a5fa', width: 1.3 } },
        { x, y: rebuilt[leadIndex], type: 'scatter', mode: 'lines', name: 'Reconstruction', line: { color: '#34d399', width: 1.3 } },
    ], darkLayout(`Lead ${leadName}: normalized input vs reconstruction`, 'Sample index (100 Hz)', 'Normalized amplitude'));

    plot('plot-unified-error', [
        {
            x,
            y: error[leadIndex],
            type: 'scatter',
            mode: 'lines',
            name: 'Squared error',
            line: { color: '#f87171', width: 1 },
            fill: 'tozeroy',
            fillcolor: 'rgba(248,113,113,0.22)',
        },
    ], darkLayout(
        `Lead ${leadName} reconstruction error · record mean ${formatNumber(currentUnifiedResult.anomaly_score, 5)} ` +
        `· threshold ${formatNumber(currentUnifiedResult.anomaly_threshold, 5)}`,
        'Sample index (100 Hz)',
        'Squared error (normalized units)'
    ));

    const items = (currentUnifiedResult.conditions && currentUnifiedResult.conditions.items) || [];
    plot('plot-unified-scores', [
        {
            x: items.map((item) => item.category),
            y: items.map((item) => Number(item.score)),
            type: 'bar',
            name: 'Sigmoid score',
            marker: { color: items.map((item) => CATEGORY_COLORS[item.category] || '#60a5fa') },
        },
        {
            x: items.map((item) => item.category),
            y: items.map((item) => Number(item.threshold)),
            type: 'scatter',
            mode: 'markers',
            name: 'Frozen threshold',
            marker: { color: '#f8fafc', symbol: 'line-ew-open', size: 18, line: { width: 3 } },
        },
    ], {
        ...darkLayout('Independent condition scores and thresholds', 'Category', 'Score (not a calibrated probability)'),
        yaxis: { title: 'Score / threshold', gridcolor: GRID, range: [0, 1.05] },
        barmode: 'group',
    });
}

function renderUnifiedResult(result) {
    currentUnifiedResult = result;
    currentUnifiedSignal = result.reconstruction ? result.reconstruction.original : null;
    setText('unified-anomaly-score', formatNumber(result.anomaly_score, 6));
    setText('unified-anomaly-threshold', formatNumber(result.anomaly_threshold, 6));
    const status = document.getElementById('unified-anomaly-status');
    if (status) {
        status.textContent = result.anomaly_status === 'anomalous' ? 'ABOVE THRESHOLD' : 'BELOW THRESHOLD';
        status.className = 'result-value ' + (result.anomaly_status === 'anomalous' ? 'is-anomalous' : 'is-normal');
    }
    renderConditionRows(result);
    const panel = document.getElementById('unified-results-panel');
    if (panel) panel.hidden = false;
    const explainNote = document.getElementById('unified-explain-note');
    if (explainNote) explainNote.textContent = '';
    plotUnifiedPlots();
}

async function runUnifiedExplanation() {
    const category = document.getElementById('unified-explain-category');
    const note = document.getElementById('unified-explain-note');
    if (!currentUnifiedSignal) {
        if (note) note.textContent = 'Run an analysis first.';
        return;
    }
    if (note) note.textContent = 'Computing temporal occlusion sensitivity…';
    try {
        const payload = await fetchJSON('/api/unified/explain', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                signal: currentUnifiedSignal,
                category: category ? category.value : 'NORM',
                sampling_rate: 100,
                window_samples: 100,
                stride_samples: 50,
            }),
        });
        const x = Array.from({ length: payload.saliency.length }, (_, index) => index);
        plot('plot-unified-explain', [
            {
                x,
                y: payload.saliency,
                type: 'scatter',
                mode: 'lines',
                name: `${payload.category} logit change`,
                line: { color: CATEGORY_COLORS[payload.category] || '#60a5fa' },
                fill: 'tozeroy',
                fillcolor: 'rgba(96,165,250,0.18)',
            },
        ], darkLayout(
            `${payload.category} temporal occlusion sensitivity (window ${payload.window_samples} samples)`,
            'Sample index (100 Hz)',
            'Logit change'
        ));
        if (note) {
            note.textContent = 'Model-sensitivity regions, not a clinical explanation. ' +
                'Positive values indicate that occluding the region reduced the category logit.';
        }
    } catch (error) {
        if (note) note.textContent = 'Explanation failed: ' + error.message;
    }
}

function clearUnified() {
    currentUnifiedResult = null;
    currentUnifiedSignal = null;
    const file = document.getElementById('unified-file');
    if (file) file.value = '';
    const select = document.getElementById('unified-example-select');
    if (select) select.value = '';
    const textarea = document.getElementById('unified-signal-input');
    if (textarea) textarea.value = '';
    hideError('unified-error');
    setText('unified-anomaly-score', '—');
    setText('unified-anomaly-status', '—');
    const status = document.getElementById('unified-anomaly-status');
    if (status) status.className = 'result-value';
    const container = document.getElementById('unified-condition-results');
    if (container) container.innerHTML = '<p class="muted">Run an analysis to see category scores.</p>';
    setText('unified-predicted-summary', '');
    const conflict = document.getElementById('unified-classifier-conflict');
    if (conflict) {
        conflict.hidden = true;
        conflict.textContent = '';
    }
    const panel = document.getElementById('unified-results-panel');
    if (panel) panel.hidden = true;
    const note = document.getElementById('unified-explain-note');
    if (note) note.textContent = '';
}

// ---------------------------------------------------------------------------
// Legacy ECG5000
// ---------------------------------------------------------------------------
async function loadLegacyHealth() {
    const node = document.getElementById('legacy-status');
    try {
        const health = await fetchJSON('/api/health');
        legacyAvailable = Boolean(health.model_loaded);
        if (!node) return;
        if (legacyAvailable) {
            node.innerHTML = `<strong>Legacy model loaded.</strong> 140-sample input · threshold ${formatNumber(health.threshold, 6)}`;
        } else {
            node.innerHTML = '<strong>Legacy ECG5000 model is not available.</strong> ' +
                'Train it with <code>python src/train.py</code>. ' +
                escapeHTML(health.error || '');
        }
    } catch (error) {
        if (node) node.textContent = 'Could not check the legacy model: ' + error.message;
    }
}

async function loadLegacyExamples() {
    const select = document.getElementById('example-select');
    if (!select) return;
    try {
        const payload = await fetchJSON('/api/examples');
        select.innerHTML = '<option value="">-- Select example --</option>';
        (payload.examples || []).forEach((example) => {
            const option = document.createElement('option');
            option.value = example.index;
            option.textContent = `Example ${example.index} (label ${example.true_label})`;
            select.appendChild(option);
        });
    } catch (error) {
        // Dataset may not be prepared yet; keep the dropdown empty.
    }
}

async function loadLegacyExample() {
    const select = document.getElementById('example-select');
    if (!select || select.value === '') return;
    hideError('analyzer-error');
    try {
        if (!legacyAvailable) throw new Error('Legacy ECG5000 model is not available.');
        const result = await fetchJSON('/api/example/' + encodeURIComponent(select.value));
        const input = document.getElementById('signal-input');
        if (input) input.value = result.original.join(', ');
        renderLegacyResult(result);
        plotLegacy(result);
    } catch (error) {
        showError('analyzer-error', error.message);
    }
}

async function runLegacyPrediction() {
    hideError('analyzer-error');
    const input = document.getElementById('signal-input');
    const button = document.getElementById('predict-btn');
    if (!input) return;
    const values = input.value.replace(/\n/g, ' ').split(/[\s,]+/).filter((value) => value !== '').map(Number);
    if (values.length !== 140) {
        showError('analyzer-error', `Signal must have exactly 140 values, got ${values.length}.`);
        return;
    }
    if (values.some((value) => !Number.isFinite(value))) {
        showError('analyzer-error', 'All 140 values must be finite numbers.');
        return;
    }
    if (!legacyAvailable) {
        showError('analyzer-error', 'Legacy ECG5000 model is not available. Train it with python src/train.py.');
        return;
    }
    const label = button ? button.textContent : '';
    if (button) {
        button.disabled = true;
        button.textContent = 'Analyzing…';
    }
    try {
        const result = await fetchJSON('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ signal: values }),
        });
        renderLegacyResult(result);
        plotLegacy(result);
    } catch (error) {
        showError('analyzer-error', error.message);
    } finally {
        if (button) {
            button.disabled = false;
            button.textContent = label;
        }
    }
}

function renderLegacyResult(result) {
    setText('res-mse', formatNumber(result.segment_mse, 6));
    setText('res-threshold', formatNumber(result.threshold, 6));
    setText('res-score', formatNumber(result.anomaly_score, 6));
    const prediction = document.getElementById('res-pred');
    if (prediction) {
        prediction.textContent = result.predicted_label === 'anomalous' ? 'ANOMALOUS' : 'NORMAL';
        prediction.className = 'result-value ' + (result.predicted_label === 'anomalous' ? 'is-anomalous' : 'is-normal');
    }
    const trueNode = document.getElementById('legacy-true');
    if (trueNode) {
        trueNode.textContent = result.true_label !== undefined
            ? `Dataset label: ${result.true_label} (${Number(result.true_label) === 1 ? 'normal class' : 'non-normal class'})`
            : '';
    }
}

function plotLegacy(result) {
    const x = Array.from({ length: result.original.length }, (_, index) => index);
    plot('plot-waveform', [
        { x, y: result.original, type: 'scatter', mode: 'lines', name: 'Original', line: { color: '#60a5fa' } },
        { x, y: result.reconstructed, type: 'scatter', mode: 'lines', name: 'Reconstructed', line: { color: '#34d399' } },
    ], darkLayout('ECG5000 original vs reconstruction', 'Sample index', 'Amplitude (dataset units)'));
    plot('plot-error', [
        {
            x,
            y: result.pointwise_squared_error,
            type: 'scatter',
            mode: 'lines',
            name: 'Squared error',
            line: { color: '#f87171' },
            fill: 'tozeroy',
            fillcolor: 'rgba(248,113,113,0.22)',
        },
    ], darkLayout('ECG5000 point-wise squared reconstruction error', 'Sample index', 'Squared error'));
}

// ---------------------------------------------------------------------------
// Evaluation
// ---------------------------------------------------------------------------
function metricCell(value) {
    return `<td class="num">${escapeHTML(formatMetric(value))}</td>`;
}

function renderClassificationTable(classification) {
    const order = classification.label_order || [];
    let html = '<table class="eval-table"><thead><tr>' +
        '<th>Category</th><th>Threshold</th><th>Support</th><th>Pred. pos.</th>' +
        '<th>Precision</th><th>Recall</th><th>F1</th><th>AUROC</th><th>AP</th>' +
        '</tr></thead><tbody>';
    order.forEach((category) => {
        const item = classification.per_category[category] || {};
        html += '<tr>' +
            `<td>${escapeHTML(category)}</td>` +
            metricCell(item.threshold) + metricCell(item.positive_support) + metricCell(item.predicted_positive_count) +
            metricCell(item.precision) + metricCell(item.recall) + metricCell(item.f1) +
            metricCell(item.auroc) + metricCell(item.average_precision) +
            '</tr>';
    });
    html += '</tbody></table>';
    html += `<p class="field-hint">Macro-F1: ${escapeHTML(formatMetric(classification.macro_f1))} · ` +
        `Micro-F1: ${escapeHTML(formatMetric(classification.micro_f1))} · ` +
        `“undefined” means the metric could not be computed for this partition.</p>`;
    return html;
}

function renderAnomalyBlock(anomaly) {
    if (!anomaly || anomaly.n_evaluated === 0) {
        return '<p class="muted">No anomaly evaluation available for this partition.</p>';
    }
    const confusion = anomaly.confusion || {};
    return '<div class="metrics-list">' +
        `<div><span>Threshold</span><span>${formatNumber(anomaly.threshold, 6)}</span></div>` +
        `<div><span>Evaluated records</span><span>${anomaly.n_evaluated}</span></div>` +
        `<div><span>Precision</span><span>${formatMetric(anomaly.precision)}</span></div>` +
        `<div><span>Recall</span><span>${formatMetric(anomaly.recall)}</span></div>` +
        `<div><span>Specificity</span><span>${formatMetric(anomaly.specificity)}</span></div>` +
        `<div><span>F1</span><span>${formatMetric(anomaly.f1)}</span></div>` +
        `<div><span>AUROC</span><span>${formatMetric(anomaly.auroc)}</span></div>` +
        `<div><span>TN / FP / FN / TP</span><span>${confusion.tn ?? '—'} / ${confusion.fp ?? '—'} / ${confusion.fn ?? '—'} / ${confusion.tp ?? '—'}</span></div>` +
        '</div>';
}

async function loadEvaluation() {
    loadUnifiedEvaluation();
    loadLegacyEvaluation();
}

async function loadUnifiedEvaluation() {
    const container = document.getElementById('unified-eval-content');
    if (!container) return;
    try {
        const payload = await fetchJSON('/api/unified/metrics');
        if (!payload.available) {
            container.innerHTML = '<p class="status-banner status-banner-warn">No unified evaluation artifacts are available. ' +
                'Train and evaluate the model first.</p>';
            return;
        }
        let html = '';
        const evaluations = payload.evaluations || {};
        const splits = Object.keys(evaluations);
        if (!splits.length) {
            html += '<p class="status-banner status-banner-info">Model artifact found, but no evaluation JSON has been generated. ' +
                'Run <code>python -m src.unified_evaluate --split val</code> and then <code>--split test</code> after freezing thresholds.</p>';
        }
        splits.forEach((split) => {
            const evaluation = evaluations[split];
            html += '<div class="eval-block">';
            html += `<h4>${escapeHTML(split.toUpperCase())} partition · thresholds frozen: ${evaluation.thresholds_frozen ? 'yes' : 'no'}</h4>`;
            html += '<h4>Anomaly detection (reconstruction error)</h4>';
            html += renderAnomalyBlock(evaluation.anomaly);
            html += '<h4 style="margin-top:1rem">Condition classification (independent sigmoid scores)</h4>';
            html += renderClassificationTable(evaluation.classification || {});
            html += '</div>';
        });
        html += '<p class="field-hint">Anomaly metrics and condition metrics use different score scales and must not be compared directly.</p>';
        container.innerHTML = html;
    } catch (error) {
        container.innerHTML = `<p class="status-banner status-banner-error">Could not load unified evaluation: ${escapeHTML(error.message)}</p>`;
    }
}

async function loadLegacyEvaluation() {
    const metricsNode = document.getElementById('eval-metrics');
    if (!metricsNode) return;
    try {
        const payload = await fetchJSON('/api/metrics');
        const metrics = payload.metrics;
        if (!metrics) throw new Error('No legacy metrics file');
        metricsNode.innerHTML = `
            <div><span>Threshold</span><span>${formatNumber(metrics.threshold, 6)}</span></div>
            <div><span>Precision</span><span>${formatMetric(metrics.precision)}</span></div>
            <div><span>Recall / sensitivity</span><span>${formatMetric(metrics.recall_sensitivity)}</span></div>
            <div><span>Specificity</span><span>${formatMetric(metrics.specificity)}</span></div>
            <div><span>F1</span><span>${formatMetric(metrics.f1_score)}</span></div>
            <div><span>AUROC</span><span>${formatMetric(metrics.auroc)}</span></div>
            <div><span>Accuracy</span><span>${formatMetric(metrics.accuracy)}</span></div>
            <div><span>TP / FP / TN / FN</span><span>${metrics.tp} / ${metrics.fp} / ${metrics.tn} / ${metrics.fn}</span></div>
        `;
        if (metrics.confusion_matrix && window.Plotly) {
            plot('plot-cm', [{
                z: metrics.confusion_matrix,
                x: ['Normal', 'Anomalous'],
                y: ['Normal', 'Anomalous'],
                type: 'heatmap',
                colorscale: 'Blues',
            }], darkLayout('ECG5000 confusion matrix', 'Predicted', 'True'));
        }
    } catch (error) {
        metricsNode.innerHTML = '<p class="muted">Legacy metrics unavailable.</p>';
    }
}

// ---------------------------------------------------------------------------
// Wiring
// ---------------------------------------------------------------------------
document.addEventListener('DOMContentLoaded', () => {
    const predictButton = document.getElementById('unified-predict-btn');
    if (predictButton) predictButton.addEventListener('click', runUnifiedAnalysis);
    const clearButton = document.getElementById('unified-clear-btn');
    if (clearButton) clearButton.addEventListener('click', clearUnified);
    const leadSelect = document.getElementById('unified-lead-select');
    if (leadSelect) leadSelect.addEventListener('change', plotUnifiedPlots);
    const explainButton = document.getElementById('unified-explain-btn');
    if (explainButton) explainButton.addEventListener('click', runUnifiedExplanation);

    const loadExampleButton = document.getElementById('load-example-btn');
    if (loadExampleButton) loadExampleButton.addEventListener('click', loadLegacyExample);
    const legacyPredictButton = document.getElementById('predict-btn');
    if (legacyPredictButton) legacyPredictButton.addEventListener('click', runLegacyPrediction);

    loadUnifiedHealth();
    loadUnifiedExamples();
    loadLegacyHealth();
    loadLegacyExamples();
});
