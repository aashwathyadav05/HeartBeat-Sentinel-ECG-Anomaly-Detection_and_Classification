// HeartBeat Sentinel Frontend
const API_BASE = '';

let currentData = {};

// Tab navigation
document.querySelectorAll('.nav-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
        btn.classList.add('active');
        document.getElementById('tab-' + btn.dataset.tab).classList.add('active');
        if (btn.dataset.tab === 'evaluation') loadEvaluation();
        if (btn.dataset.tab === 'explorer') loadExamples();
    });
});

// Load health and overview
async function loadOverview() {
    try {
        const health = await fetch(API_BASE + '/api/health').then(r => r.json());
        const healthDiv = document.getElementById('health-status');
        healthDiv.innerHTML = `
            <strong>Status:</strong> ${health.status}<br>
            <strong>Model Loaded:</strong> ${health.model_loaded ? '✅ Yes' : '❌ No'}<br>
            <strong>Sequence Length:</strong> ${health.seq_len}<br>
            <strong>Threshold:</strong> ${health.threshold ? health.threshold.toFixed(6) : 'N/A'}
        `;
        const metrics = await fetch(API_BASE + '/api/metrics').then(r => r.json());
        if (metrics.metrics) {
            const m = metrics.metrics;
            document.getElementById('overview-metrics').innerHTML = `
                <div><strong>AUROC</strong><br>${m.auroc?.toFixed(3) || '-'}</div>
                <div><strong>F1 Score</strong><br>${m.f1_score?.toFixed(3) || '-'}</div>
                <div><strong>Accuracy</strong><br>${m.accuracy?.toFixed(3) || '-'}</div>
                <div><strong>Precision</strong><br>${m.precision?.toFixed(3) || '-'}</div>
            `;
        }
    } catch (e) {
        document.getElementById('health-status').innerHTML = '<span style="color:var(--accent-red)">Error loading status</span>';
    }
}

// Load examples for dropdown
async function loadExampleDropdown() {
    try {
        const res = await fetch(API_BASE + '/api/examples').then(r => r.json());
        const select = document.getElementById('example-select');
        if (res.examples) {
            res.examples.forEach(ex => {
                const opt = document.createElement('option');
                opt.value = ex.index;
                opt.textContent = `Example ${ex.index} (Label: ${ex.true_label})`;
                select.appendChild(opt);
            });
        }
    } catch (e) {}
}

document.getElementById('load-example-btn').addEventListener('click', async () => {
    const idx = document.getElementById('example-select').value;
    if (idx === '') return;
    try {
        const res = await fetch(API_BASE + '/api/example/' + idx).then(r => r.json());
        if (res.error) {
            document.getElementById('analyzer-error').textContent = res.error;
            document.getElementById('analyzer-error').style.display = 'block';
            return;
        }
        document.getElementById('analyzer-error').style.display = 'none';
        document.getElementById('signal-input').value = res.original.join(', ');
        displayResults(res);
        plotWaveforms(res);
    } catch (e) {
        document.getElementById('analyzer-error').textContent = 'Error loading example';
        document.getElementById('analyzer-error').style.display = 'block';
    }
});

document.getElementById('predict-btn').addEventListener('click', async () => {
    const input = document.getElementById('signal-input').value;
    let signal = [];
    try {
        // Try comma or whitespace separated
        signal = input.replace(/\n/g, ' ').split(/[\s,]+/).filter(x => x !== '').map(x => parseFloat(x));
    } catch (e) {
        document.getElementById('analyzer-error').textContent = 'Invalid input format';
        document.getElementById('analyzer-error').style.display = 'block';
        return;
    }
    if (signal.length !== 140) {
        document.getElementById('analyzer-error').textContent = `Signal must have exactly 140 values, got ${signal.length}`;
        document.getElementById('analyzer-error').style.display = 'block';
        return;
    }
    document.getElementById('analyzer-error').style.display = 'none';
    document.getElementById('predict-btn').disabled = true;
    document.getElementById('predict-btn').textContent = 'Analyzing...';
    try {
        const res = await fetch(API_BASE + '/api/predict', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({signal: signal})
        }).then(r => r.json());
        if (res.error) {
            document.getElementById('analyzer-error').textContent = res.error;
            document.getElementById('analyzer-error').style.display = 'block';
        } else {
            displayResults(res);
            plotWaveforms(res);
        }
    } catch (e) {
        document.getElementById('analyzer-error').textContent = 'Request failed';
        document.getElementById('analyzer-error').style.display = 'block';
    }
    document.getElementById('predict-btn').disabled = false;
    document.getElementById('predict-btn').textContent = 'Run Anomaly Detection';
});

function displayResults(res) {
    document.getElementById('res-mse').textContent = res.segment_mse?.toFixed(6) || '-';
    document.getElementById('res-threshold').textContent = res.threshold?.toFixed(6) || '-';
    document.getElementById('res-pred').textContent = res.predicted_label === 'anomalous' ? '⚠️ ANOMALOUS' : '✅ NORMAL';
    document.getElementById('res-score').textContent = res.anomaly_score?.toFixed(6) || '-';
    if (res.true_label !== undefined) {
        document.getElementById('res-true').innerHTML = `<strong>True Label:</strong> ${res.true_label} (${res.true_label == 1.0 ? 'Normal' : 'Non-normal'})`;
    } else {
        document.getElementById('res-true').innerHTML = '';
    }
}

function plotWaveforms(res) {
    const x = Array.from({length: res.original.length}, (_, i) => i);
    const trace1 = {x: x, y: res.original, type: 'scatter', mode: 'lines', name: 'Original', line: {color: '#3b82f6'}};
    const trace2 = {x: x, y: res.reconstructed, type: 'scatter', mode: 'lines', name: 'Reconstructed', line: {color: '#10b981'}};
    Plotly.newPlot('plot-waveform', [trace1, trace2], {
        title: 'Original vs Reconstructed ECG',
        paper_bgcolor: '#1e293b',
        plot_bgcolor: '#0f172a',
        font: {color: '#e2e8f0'},
        xaxis: {title: 'Sample Index', gridcolor: '#334155'},
        yaxis: {title: 'Amplitude', gridcolor: '#334155'}
    });
    const traceErr = {x: x, y: res.pointwise_squared_error, type: 'scatter', mode: 'lines', name: 'Point-wise Error', line: {color: '#ef4444'}, fill: 'tozeroy'};
    Plotly.newPlot('plot-error', [traceErr], {
        title: 'Point-wise Squared Reconstruction Error',
        paper_bgcolor: '#1e293b',
        plot_bgcolor: '#0f172a',
        font: {color: '#e2e8f0'},
        xaxis: {title: 'Sample Index', gridcolor: '#334155'},
        yaxis: {title: 'Error', gridcolor: '#334155'}
    });
}

async function loadExamples() {
    try {
        const res = await fetch(API_BASE + '/api/examples').then(r => r.json());
        const div = document.getElementById('examples-list');
        div.innerHTML = '';
        if (res.examples) {
            res.examples.slice(0, 12).forEach(ex => {
                const item = document.createElement('div');
                item.className = 'example-item';
                item.innerHTML = `<strong>Index ${ex.index}</strong><br>Label: ${ex.true_label}<br><small>${ex.true_label == 1.0 ? 'Normal' : 'Non-normal'}</small>`;
                item.addEventListener('click', async () => {
                    document.querySelectorAll('.nav-btn').forEach(b => {
                        if (b.dataset.tab === 'analyzer') b.click();
                    });
                    document.getElementById('example-select').value = ex.index;
                    document.getElementById('load-example-btn').click();
                });
                div.appendChild(item);
            });
        }
    } catch (e) {}
}

async function loadEvaluation() {
    try {
        const res = await fetch(API_BASE + '/api/metrics').then(r => r.json());
        if (res.metrics) {
            const m = res.metrics;
            document.getElementById('eval-metrics').innerHTML = `
                <div><span>Threshold</span><span>${m.threshold?.toFixed(6)}</span></div>
                <div><span>Precision</span><span>${m.precision?.toFixed(4)}</span></div>
                <div><span>Recall/Sensitivity</span><span>${m.recall_sensitivity?.toFixed(4)}</span></div>
                <div><span>Specificity</span><span>${m.specificity?.toFixed(4)}</span></div>
                <div><span>F1-Score</span><span>${m.f1_score?.toFixed(4)}</span></div>
                <div><span>AUROC</span><span>${m.auroc?.toFixed(4)}</span></div>
                <div><span>Accuracy</span><span>${m.accuracy?.toFixed(4)}</span></div>
                <div><span>TP/FP/TN/FN</span><span>${m.tp}/${m.fp}/${m.tn}/${m.fn}</span></div>
            `;
            if (m.confusion_matrix) {
                const cm = m.confusion_matrix;
                const trace = {z: cm, x: ['Normal', 'Anom'], y: ['Normal', 'Anom'], type: 'heatmap', colorscale: 'Blues'};
                Plotly.newPlot('plot-cm', [trace], {
                    title: 'Confusion Matrix',
                    paper_bgcolor: '#1e293b',
                    plot_bgcolor: '#0f172a',
                    font: {color: '#e2e8f0'}
                });
            }
        }
    } catch (e) {}
}

document.addEventListener('DOMContentLoaded', () => {
    loadOverview();
    loadExampleDropdown();
});
