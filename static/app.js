// API Base URL
const API_BASE = '';

// Initialize app on load
document.addEventListener('DOMContentLoaded', () => {
    initializeTabs();
    refreshStatus();
    loadModels();
    loadLoggingConfig();
    setInterval(refreshStatus, 10000); // Update status every 10 seconds
});

// Tab Management
function initializeTabs() {
    const tabButtons = document.querySelectorAll('.tab-btn');
    tabButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const tabName = btn.dataset.tab;
            switchTab(tabName);
        });
    });
}

function switchTab(tabName) {
    // Update tab buttons
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');

    // Update tab content
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.remove('active');
    });
    document.getElementById(`${tabName}-tab`).classList.add('active');

    // Load tab-specific data
    if (tabName === 'models') {
        loadModels();
    } else if (tabName === 'analytics') {
        loadModelListForAnalytics();
    }
}

// Status Management
async function refreshStatus() {
    try {
        const response = await fetch(`${API_BASE}/api/status`);
        const data = await response.json();

        // Update status indicator
        const statusDot = document.getElementById('status-indicator');
        const statusText = document.getElementById('status-text');

        if (data.loaded_model) {
            statusDot.classList.add('online');
            statusText.textContent = `Model Loaded: ${data.loaded_model}`;
        } else {
            statusDot.classList.remove('online');
            statusText.textContent = 'No Model Loaded';
        }

        // Update sidebar info
        document.getElementById('current-model').textContent = data.loaded_model || 'None';
        document.getElementById('gpu-available').textContent = data.gpu_available ? 'Yes' : 'No';

        if (data.gpu_available) {
            const gpuMem = `${data.gpu_memory_allocated_mb?.toFixed(0) || 0} MB / ${data.gpu_memory_reserved_mb?.toFixed(0) || 0} MB`;
            document.getElementById('gpu-memory').textContent = gpuMem;
        } else {
            document.getElementById('gpu-memory').textContent = 'N/A';
        }

        document.getElementById('perf-logging').textContent = data.performance_logging ? 'Enabled' : 'Disabled';

    } catch (error) {
        console.error('Failed to refresh status:', error);
        document.getElementById('status-text').textContent = 'Error';
    }
}

// Model Management
async function loadModels() {
    try {
        const response = await fetch(`${API_BASE}/api/models`);
        const models = await response.json();

        const modelsList = document.getElementById('models-list');
        modelsList.innerHTML = '';

        if (models.length === 0) {
            modelsList.innerHTML = '<p class="loading">No models registered. Click "Register New Model" to add one.</p>';
            return;
        }

        models.forEach(model => {
            const card = createModelCard(model);
            modelsList.appendChild(card);
        });

    } catch (error) {
        console.error('Failed to load models:', error);
        document.getElementById('models-list').innerHTML = '<p class="loading text-danger">Failed to load models</p>';
    }
}

function createModelCard(model) {
    const card = document.createElement('div');
    card.className = 'model-card';

    const formatDate = (dateStr) => {
        return dateStr ? new Date(dateStr).toLocaleString() : 'Never';
    };

    card.innerHTML = `
        <h4>${model.model_name}</h4>
        <p><strong>HF Path:</strong> ${model.hf_path}</p>
        ${model.parameter_count ? `<p><strong>Parameters:</strong> ${(model.parameter_count / 1e9).toFixed(2)}B</p>` : ''}
        ${model.architecture ? `<p><strong>Architecture:</strong> ${model.architecture}</p>` : ''}
        <p><strong>Total Loads:</strong> ${model.total_loads || 0}</p>
        <p><strong>Total Inferences:</strong> ${model.total_inferences || 0}</p>
        <p><strong>Last Loaded:</strong> ${formatDate(model.last_loaded)}</p>
        <div class="model-card-actions">
            <button class="btn btn-primary" onclick="loadModelToMemory('${model.model_name}')">Load</button>
            <button class="btn btn-danger" onclick="deleteModel('${model.model_name}')">Delete</button>
        </div>
    `;

    return card;
}

async function loadModelToMemory(modelName) {
    if (!confirm(`Load model "${modelName}"? This may take a few minutes.`)) {
        return;
    }

    try {
        const response = await fetch(`${API_BASE}/api/orchestrate/load`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model_name: modelName })
        });

        const data = await response.json();

        if (response.ok) {
            alert(`Model loaded successfully!\nParameters: ${data.parameter_count?.toLocaleString() || 'Unknown'}`);
            refreshStatus();
        } else {
            alert(`Failed to load model: ${data.detail}`);
        }
    } catch (error) {
        alert(`Error loading model: ${error.message}`);
    }
}

async function unloadModel() {
    try {
        const response = await fetch(`${API_BASE}/api/orchestrate/unload`, {
            method: 'POST'
        });

        const data = await response.json();

        if (response.ok) {
            alert(data.message);
            refreshStatus();
        } else {
            alert(`Failed to unload model: ${data.detail}`);
        }
    } catch (error) {
        alert(`Error unloading model: ${error.message}`);
    }
}

async function deleteModel(modelName) {
    if (!confirm(`Delete model "${modelName}"? This cannot be undone.`)) {
        return;
    }

    try {
        const response = await fetch(`${API_BASE}/api/models/${modelName}`, {
            method: 'DELETE'
        });

        const data = await response.json();

        if (response.ok) {
            alert(data.message);
            loadModels();
        } else {
            alert(`Failed to delete model: ${data.detail}`);
        }
    } catch (error) {
        alert(`Error deleting model: ${error.message}`);
    }
}

// Model Registration
function showRegisterModal() {
    document.getElementById('register-modal').classList.add('active');
}

function closeRegisterModal() {
    document.getElementById('register-modal').classList.remove('active');
    document.getElementById('register-form').reset();
}

async function registerModel(event) {
    event.preventDefault();

    const formData = {
        model_name: document.getElementById('model-name').value,
        hf_path: document.getElementById('hf-path').value,
        trust_remote_code: document.getElementById('trust-remote').checked
    };

    try {
        const response = await fetch(`${API_BASE}/api/models`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(formData)
        });

        const data = await response.json();

        if (response.ok) {
            alert('Model registered successfully!');
            closeRegisterModal();
            loadModels();
        } else {
            alert(`Failed to register model: ${data.detail}`);
        }
    } catch (error) {
        alert(`Error registering model: ${error.message}`);
    }
}

function showLoadModel() {
    switchTab('models');
}

// Inference
async function generateText(event) {
    event.preventDefault();

    const formData = {
        prompt: document.getElementById('prompt').value,
        max_tokens: parseInt(document.getElementById('max-tokens').value),
        temperature: parseFloat(document.getElementById('temperature').value),
        top_p: parseFloat(document.getElementById('top-p').value),
        do_sample: document.getElementById('do-sample').checked
    };

    const submitBtn = event.target.querySelector('button[type="submit"]');
    submitBtn.textContent = 'Generating...';
    submitBtn.disabled = true;

    try {
        const response = await fetch(`${API_BASE}/api/generate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(formData)
        });

        const data = await response.json();

        if (response.ok) {
            // Show results
            document.getElementById('generation-result').style.display = 'block';
            document.getElementById('generated-text').textContent = data.generated_text;

            const stats = `
                <strong>Input Tokens:</strong> ${data.input_tokens} |
                <strong>Output Tokens:</strong> ${data.output_tokens} |
                <strong>Total Tokens:</strong> ${data.total_tokens} |
                <strong>Time:</strong> ${data.inference_time_ms.toFixed(0)}ms |
                <strong>Speed:</strong> ${data.tokens_per_second.toFixed(2)} tok/s
            `;
            document.getElementById('generation-stats').innerHTML = stats;
        } else {
            alert(`Generation failed: ${data.detail}`);
        }
    } catch (error) {
        alert(`Error during generation: ${error.message}`);
    } finally {
        submitBtn.textContent = 'Generate';
        submitBtn.disabled = false;
    }
}

// Settings
async function loadLoggingConfig() {
    try {
        const response = await fetch(`${API_BASE}/api/config/logging`);
        const data = await response.json();

        const toggle = document.getElementById('logging-toggle');
        toggle.checked = data.performance_logging;
        updateLoggingStatus(data.performance_logging);
    } catch (error) {
        console.error('Failed to load logging config:', error);
    }
}

async function toggleLogging() {
    const toggle = document.getElementById('logging-toggle');

    try {
        const response = await fetch(`${API_BASE}/api/config/logging`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ enable: toggle.checked })
        });

        const data = await response.json();

        if (response.ok) {
            updateLoggingStatus(toggle.checked);
            refreshStatus();
        } else {
            toggle.checked = !toggle.checked;
            alert(`Failed to update logging: ${data.detail}`);
        }
    } catch (error) {
        toggle.checked = !toggle.checked;
        alert(`Error updating logging: ${error.message}`);
    }
}

function updateLoggingStatus(enabled) {
    const statusText = enabled ? 'Enabled (collecting detailed metrics)' : 'Disabled (efficiency mode)';
    document.getElementById('logging-status').textContent = statusText;
}

async function validateModel() {
    const resultDiv = document.getElementById('validation-result');
    resultDiv.innerHTML = '<p class="loading">Running validation test...</p>';

    try {
        const response = await fetch(`${API_BASE}/api/orchestrate/validate`);
        const data = await response.json();

        if (data.status === 'healthy') {
            resultDiv.innerHTML = `
                <div class="info-box" style="background: #f0fdf4; border-color: #bbf7d0; color: #166534;">
                    <strong>✓ Validation Passed</strong><br>
                    Model: ${data.model_name}<br>
                    Device: ${data.device}<br>
                    Data Type: ${data.dtype}
                </div>
            `;
        } else {
            resultDiv.innerHTML = `
                <div class="info-box" style="background: #fef2f2; border-color: #fecaca; color: #991b1b;">
                    <strong>✗ Validation Failed</strong><br>
                    ${data.detail || data.error}
                </div>
            `;
        }
    } catch (error) {
        resultDiv.innerHTML = `
            <div class="info-box" style="background: #fef2f2; border-color: #fecaca; color: #991b1b;">
                <strong>Error:</strong> ${error.message}
            </div>
        `;
    }
}

// Analytics
async function loadModelListForAnalytics() {
    try {
        const response = await fetch(`${API_BASE}/api/models`);
        const models = await response.json();

        const select = document.getElementById('analytics-model-select');
        select.innerHTML = '<option value="">-- Select a model --</option>';

        models.forEach(model => {
            const option = document.createElement('option');
            option.value = model.model_name;
            option.textContent = model.model_name;
            select.appendChild(option);
        });
    } catch (error) {
        console.error('Failed to load models for analytics:', error);
    }
}

async function loadAnalytics() {
    const modelName = document.getElementById('analytics-model-select').value;

    if (!modelName) {
        document.getElementById('analytics-data').style.display = 'none';
        return;
    }

    try {
        // Load performance stats
        const statsResponse = await fetch(`${API_BASE}/api/analytics/performance/${modelName}`);
        const stats = await statsResponse.json();

        if (statsResponse.ok) {
            document.getElementById('stat-total').textContent = stats.total_inferences;
            document.getElementById('stat-input').textContent = stats.avg_input_tokens.toFixed(1);
            document.getElementById('stat-output').textContent = stats.avg_output_tokens.toFixed(1);
            document.getElementById('stat-time').textContent = stats.avg_inference_ms.toFixed(0) + ' ms';
            document.getElementById('stat-tps').textContent = stats.avg_tokens_per_second.toFixed(2);
            document.getElementById('stat-gpu').textContent = stats.avg_gpu_mem_mb
                ? stats.avg_gpu_mem_mb.toFixed(0) + ' MB'
                : 'N/A';

            document.getElementById('analytics-data').style.display = 'block';
        } else {
            alert(`No performance data found for ${modelName}`);
            return;
        }

        // Load recent logs
        const logsResponse = await fetch(`${API_BASE}/api/analytics/logs/${modelName}?limit=20`);
        const logsData = await logsResponse.json();

        if (logsResponse.ok && logsData.logs.length > 0) {
            renderLogsTable(logsData.logs);
        }

    } catch (error) {
        console.error('Failed to load analytics:', error);
        alert(`Error loading analytics: ${error.message}`);
    }
}

function renderLogsTable(logs) {
    const tableContainer = document.getElementById('logs-table');

    let html = `
        <table>
            <thead>
                <tr>
                    <th>Timestamp</th>
                    <th>Input Tokens</th>
                    <th>Output Tokens</th>
                    <th>Time (ms)</th>
                    <th>Tok/s</th>
                    <th>GPU Mem (MB)</th>
                    <th>Temp</th>
                </tr>
            </thead>
            <tbody>
    `;

    logs.forEach(log => {
        const date = new Date(log.timestamp).toLocaleString();
        html += `
            <tr>
                <td>${date}</td>
                <td>${log.input_tokens}</td>
                <td>${log.output_tokens}</td>
                <td>${log.total_inference_ms.toFixed(0)}</td>
                <td>${log.tokens_per_second?.toFixed(2) || 'N/A'}</td>
                <td>${log.gpu_mem_peak_alloc_mb?.toFixed(0) || 'N/A'}</td>
                <td>${log.temperature || 'N/A'}</td>
            </tr>
        `;
    });

    html += '</tbody></table>';
    tableContainer.innerHTML = html;
}
