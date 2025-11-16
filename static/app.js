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
    } else if (tabName === 'settings') {
        refreshCacheStats();
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

    const formatScore = (score) => {
        return score !== null && score !== undefined ? score.toFixed(1) : '-';
    };

    // Build benchmark scores section if any are available
    const hasBenchmarks = model.mmlu_score || model.gpqa_score || model.hellaswag_score ||
                         model.humaneval_score || model.mbpp_score || model.math_score ||
                         model.truthfulqa_score || model.perplexity;

    let benchmarkSection = '';
    if (hasBenchmarks) {
        benchmarkSection = `
            <div class="model-section">
                <strong>📊 Benchmarks:</strong>
                ${model.mmlu_score ? `<span class="metric-badge">MMLU: ${formatScore(model.mmlu_score)}</span>` : ''}
                ${model.gpqa_score ? `<span class="metric-badge">GPQA: ${formatScore(model.gpqa_score)}</span>` : ''}
                ${model.hellaswag_score ? `<span class="metric-badge">HellaSwag: ${formatScore(model.hellaswag_score)}</span>` : ''}
                ${model.humaneval_score ? `<span class="metric-badge">HumanEval: ${formatScore(model.humaneval_score)}</span>` : ''}
                ${model.mbpp_score ? `<span class="metric-badge">MBPP: ${formatScore(model.mbpp_score)}</span>` : ''}
                ${model.math_score ? `<span class="metric-badge">MATH: ${formatScore(model.math_score)}</span>` : ''}
                ${model.truthfulqa_score ? `<span class="metric-badge">TruthfulQA: ${formatScore(model.truthfulqa_score)}</span>` : ''}
                ${model.perplexity ? `<span class="metric-badge">Perplexity: ${formatScore(model.perplexity)}</span>` : ''}
            </div>
        `;
    }

    card.innerHTML = `
        <h4>${model.model_name}</h4>
        <p><strong>HF Path:</strong> ${model.hf_path}</p>
        ${model.parameter_count ? `<p><strong>Parameters:</strong> ${(model.parameter_count / 1e9).toFixed(2)}B</p>` : ''}
        ${model.architecture ? `<p><strong>Architecture:</strong> ${model.architecture}</p>` : ''}
        ${model.default_dtype ? `<p><strong>Tensor Type:</strong> ${model.default_dtype}</p>` : ''}
        ${model.context_length ? `<p><strong>Context Window:</strong> ${model.context_length.toLocaleString()} tokens</p>` : ''}
        ${model.quantization ? `<p><strong>Quantization:</strong> ${model.quantization}</p>` : ''}
        ${benchmarkSection}
        ${model.max_throughput_tokens_sec || model.avg_latency_ms ? `
            <div class="model-section">
                <strong>⚡ Performance:</strong>
                ${model.max_throughput_tokens_sec ? `<span class="metric-badge">Throughput: ${formatScore(model.max_throughput_tokens_sec)} tok/s</span>` : ''}
                ${model.avg_latency_ms ? `<span class="metric-badge">TTFT: ${formatScore(model.avg_latency_ms)}ms</span>` : ''}
            </div>
        ` : ''}
        <p><strong>Total Loads:</strong> ${model.total_loads || 0} | <strong>Inferences:</strong> ${model.total_inferences || 0}</p>
        <p><strong>Last Loaded:</strong> ${formatDate(model.last_loaded)}</p>
        ${model.current_commit ? `
            <div class="model-section version-info">
                <strong>📦 Version:</strong>
                <span class="metric-badge">${model.current_commit.substring(0, 8)}</span>
                ${model.last_updated ? `<span style="font-size: 0.85em; color: #888;">Updated: ${formatDate(model.last_updated)}</span>` : ''}
                ${model.update_available ? '<span class="update-badge">Update Available!</span>' : ''}
            </div>
        ` : ''}
        <div class="model-card-actions">
            <button class="btn btn-primary" onclick="loadModelToMemory('${model.model_name}')">Load</button>
            <button class="btn btn-secondary" onclick="checkForUpdates('${model.model_name}')">🔄 Check Updates</button>
            ${model.update_available ? `<button class="btn btn-success" onclick="updateModel('${model.model_name}')">⬆️ Update</button>` : ''}
            <button class="btn btn-secondary" onclick="showEditMetadataModal('${model.model_name}')">Edit Metrics</button>
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

// Model Update Functions
async function checkForUpdates(modelName) {
    try {
        const response = await fetch(`${API_BASE}/api/models/${modelName}/updates`);
        const data = await response.json();

        if (response.ok) {
            if (data.update_available) {
                const message = `Update available for "${modelName}"!\n\n` +
                              `Current version: ${data.local_commit ? data.local_commit.substring(0, 8) : 'Unknown'}\n` +
                              `Latest version: ${data.remote_commit.substring(0, 8)}\n` +
                              `Last modified: ${new Date(data.last_modified).toLocaleString()}\n\n` +
                              `Click "Update" button to download the latest version.`;
                alert(message);
            } else {
                alert(`"${modelName}" is up to date!\n\nVersion: ${data.local_commit ? data.local_commit.substring(0, 8) : 'Unknown'}`);
            }
            // Refresh models list to show update badge if needed
            loadModels();
        } else {
            alert(`Failed to check for updates: ${data.detail}`);
        }
    } catch (error) {
        alert(`Error checking for updates: ${error.message}`);
    }
}

async function updateModel(modelName) {
    if (!confirm(`Update "${modelName}" to the latest version?\n\nThis will download new/changed files and clean up old blobs. The model must not be loaded.`)) {
        return;
    }

    try {
        // Show loading indicator
        const statusText = document.getElementById('status-text');
        const originalText = statusText.textContent;
        statusText.textContent = `Updating ${modelName}...`;

        const response = await fetch(`${API_BASE}/api/models/${modelName}/update`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ garbage_collect: true })
        });

        const data = await response.json();

        if (response.ok) {
            let message = `✅ "${modelName}" updated successfully!\n\n`;
            message += `Old version: ${data.old_commit ? data.old_commit.substring(0, 8) : 'Unknown'}\n`;
            message += `New version: ${data.new_commit.substring(0, 8)}\n`;

            if (data.garbage_collection) {
                message += `\n🧹 Cleaned up ${data.garbage_collection.deleted_blobs} orphaned blobs`;
                message += `\n💾 Freed ${data.garbage_collection.freed_mb.toFixed(1)} MB`;
            }

            alert(message);
            loadModels();
        } else {
            alert(`Failed to update model: ${data.detail}`);
        }

        // Restore status text
        statusText.textContent = originalText;

    } catch (error) {
        alert(`Error updating model: ${error.message}`);
        document.getElementById('status-text').textContent = 'No model loaded';
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

    const cacheLocation = document.getElementById('cache-location').value;
    const formData = {
        model_name: document.getElementById('model-name').value,
        hf_path: document.getElementById('hf-path').value,
        trust_remote_code: document.getElementById('trust-remote').checked,
        cache_location: cacheLocation,
        estimated_size_mb: parseInt(document.getElementById('estimated-size').value) || 5000
    };

    // Add cache_path if custom location is selected
    if (cacheLocation === 'custom') {
        const cachePath = document.getElementById('cache-path').value;
        if (!cachePath) {
            alert('Please specify a custom cache path');
            return;
        }
        formData.cache_path = cachePath;
    }

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

    const enableStreaming = document.getElementById('enable-streaming').checked;

    if (enableStreaming) {
        return generateTextStreaming(event);
    } else {
        return generateTextNonStreaming(event);
    }
}

// Non-streaming generation (original)
async function generateTextNonStreaming(event) {
    const formData = {
        prompt: document.getElementById('prompt').value,
        max_tokens: parseInt(document.getElementById('max-tokens').value),
        temperature: parseFloat(document.getElementById('temperature').value),
        top_p: parseFloat(document.getElementById('top-p').value),
        do_sample: document.getElementById('do-sample').checked
    };

    const submitBtn = document.getElementById('generate-btn');
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

// Streaming generation (new)
async function generateTextStreaming(event) {
    const formData = {
        prompt: document.getElementById('prompt').value,
        max_tokens: parseInt(document.getElementById('max-tokens').value),
        temperature: parseFloat(document.getElementById('temperature').value),
        top_p: parseFloat(document.getElementById('top-p').value),
        do_sample: document.getElementById('do-sample').checked
    };

    const submitBtn = document.getElementById('generate-btn');
    submitBtn.textContent = 'Generating...';
    submitBtn.disabled = true;

    // Show results container and clear previous content
    document.getElementById('generation-result').style.display = 'block';
    const textDiv = document.getElementById('generated-text');
    textDiv.textContent = '';
    textDiv.classList.add('streaming');

    let fullText = '';
    let stats = null;

    try {
        const response = await fetch(`${API_BASE}/api/generate/stream`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(formData)
        });

        if (!response.ok) {
            const error = await response.json();
            alert(`Generation failed: ${error.detail}`);
            return;
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            // Decode the chunk and add to buffer
            buffer += decoder.decode(value, { stream: true });

            // Process complete lines
            const lines = buffer.split('\n');

            // Keep the last incomplete line in the buffer
            buffer = lines.pop() || '';

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    try {
                        const data = JSON.parse(line.slice(6));

                        if (data.type === 'start') {
                            // Initial metadata received
                            console.log('Stream started, input tokens:', data.input_tokens);
                        } else if (data.type === 'token') {
                            // New token received - update immediately
                            fullText += data.text;
                            textDiv.textContent = fullText;
                            // Force immediate render
                            textDiv.scrollTop = textDiv.scrollHeight;
                            // Force browser to render immediately
                            await new Promise(resolve => setTimeout(resolve, 0));
                        } else if (data.type === 'done') {
                            // Generation complete
                            stats = data;
                        } else if (data.type === 'error') {
                            alert(`Generation error: ${data.error}`);
                            return;
                        }
                    } catch (e) {
                        console.error('Failed to parse SSE data:', line, e);
                    }
                }
            }
        }

        // Display final stats
        if (stats) {
            const statsHtml = `
                <strong>Input Tokens:</strong> ${stats.input_tokens} |
                <strong>Output Tokens:</strong> ${stats.output_tokens} |
                <strong>Total Tokens:</strong> ${stats.total_tokens} |
                <strong>Time:</strong> ${stats.inference_time_ms.toFixed(0)}ms |
                <strong>TTFT:</strong> ${stats.ttft_ms.toFixed(0)}ms |
                <strong>Speed:</strong> ${stats.tokens_per_second.toFixed(2)} tok/s
            `;
            document.getElementById('generation-stats').innerHTML = statsHtml;
        }

    } catch (error) {
        alert(`Error during streaming: ${error.message}`);
    } finally {
        submitBtn.textContent = 'Generate';
        submitBtn.disabled = false;
        textDiv.classList.remove('streaming');
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

// Model Metadata Editing
let currentEditingModel = null;

async function showEditMetadataModal(modelName) {
    currentEditingModel = modelName;

    // Fetch current model data
    try {
        const response = await fetch(`${API_BASE}/api/models`);
        const models = await response.json();
        const model = models.find(m => m.model_name === modelName);

        if (!model) {
            alert('Model not found');
            return;
        }

        // Populate form with current values
        document.getElementById('edit-model-name-display').textContent = modelName;
        document.getElementById('edit-mmlu').value = model.mmlu_score || '';
        document.getElementById('edit-gpqa').value = model.gpqa_score || '';
        document.getElementById('edit-hellaswag').value = model.hellaswag_score || '';
        document.getElementById('edit-humaneval').value = model.humaneval_score || '';
        document.getElementById('edit-mbpp').value = model.mbpp_score || '';
        document.getElementById('edit-math').value = model.math_score || '';
        document.getElementById('edit-truthfulqa').value = model.truthfulqa_score || '';
        document.getElementById('edit-perplexity').value = model.perplexity || '';
        document.getElementById('edit-throughput').value = model.max_throughput_tokens_sec || '';
        document.getElementById('edit-latency').value = model.avg_latency_ms || '';
        document.getElementById('edit-quantization').value = model.quantization || '';

        document.getElementById('edit-metadata-modal').classList.add('active');
    } catch (error) {
        alert(`Error loading model data: ${error.message}`);
    }
}

function closeEditMetadataModal() {
    document.getElementById('edit-metadata-modal').classList.remove('active');
    document.getElementById('edit-metadata-form').reset();
    currentEditingModel = null;
}

async function updateModelMetadata(event) {
    event.preventDefault();

    if (!currentEditingModel) {
        alert('No model selected');
        return;
    }

    const formData = {
        mmlu_score: parseFloat(document.getElementById('edit-mmlu').value) || null,
        gpqa_score: parseFloat(document.getElementById('edit-gpqa').value) || null,
        hellaswag_score: parseFloat(document.getElementById('edit-hellaswag').value) || null,
        humaneval_score: parseFloat(document.getElementById('edit-humaneval').value) || null,
        mbpp_score: parseFloat(document.getElementById('edit-mbpp').value) || null,
        math_score: parseFloat(document.getElementById('edit-math').value) || null,
        truthfulqa_score: parseFloat(document.getElementById('edit-truthfulqa').value) || null,
        perplexity: parseFloat(document.getElementById('edit-perplexity').value) || null,
        max_throughput_tokens_sec: parseFloat(document.getElementById('edit-throughput').value) || null,
        avg_latency_ms: parseFloat(document.getElementById('edit-latency').value) || null,
        quantization: document.getElementById('edit-quantization').value || null
    };

    try {
        const response = await fetch(`${API_BASE}/api/models/${currentEditingModel}/metadata`, {
            method: 'PATCH',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(formData)
        });

        const data = await response.json();

        if (response.ok) {
            alert('Model metadata updated successfully!');
            closeEditMetadataModal();
            loadModels();
        } else {
            alert(`Failed to update metadata: ${data.detail}`);
        }
    } catch (error) {
        alert(`Error updating metadata: ${error.message}`);
    }
}


// ===========================================================================
// Cache Management Functions
// ===========================================================================

function toggleCustomCachePath() {
    const cacheLocation = document.getElementById('cache-location').value;
    const customPathGroup = document.getElementById('custom-cache-path-group');

    if (cacheLocation === 'custom') {
        customPathGroup.style.display = 'block';
    } else {
        customPathGroup.style.display = 'none';
    }

    // Check cache space and show warning if needed
    checkCacheSpaceWarning();
}

async function checkCacheSpaceWarning() {
    const cacheLocation = document.getElementById('cache-location').value;
    const estimatedSize = parseInt(document.getElementById('estimated-size').value) || 5000;

    try {
        const response = await fetch(
            `${API_BASE}/api/cache/check?cache_location=${cacheLocation}&required_space_mb=${estimatedSize}`,
            { method: 'POST' }
        );
        const data = await response.json();

        const warningBox = document.getElementById('cache-warning');
        const warningText = document.getElementById('cache-warning-text');

        if (!data.sufficient) {
            warningBox.style.display = 'block';
            warningBox.style.borderColor = '#dc3545';
            warningText.textContent = `Insufficient space in ${cacheLocation} cache. ` +
                `Available: ${data.disk_free_gb?.toFixed(1)}GB, ` +
                `Required: ~${(estimatedSize / 1024).toFixed(1)}GB. ` +
                `Consider using a different cache location.`;
        } else if (data.warning) {
            warningBox.style.display = 'block';
            warningBox.style.borderColor = '#ffc107';
            warningText.textContent = `Cache usage will be at ${data.usage_percent?.toFixed(0)}% ` +
                `after downloading this model. Consider using secondary cache.`;
        } else {
            warningBox.style.display = 'none';
        }
    } catch (error) {
        console.error('Error checking cache space:', error);
    }
}

async function refreshCacheStats() {
    try {
        const response = await fetch(`${API_BASE}/api/cache/stats`);
        const data = await response.json();

        updateCacheDisplay('primary', data.primary);
        updateCacheDisplay('secondary', data.secondary);
    } catch (error) {
        console.error('Failed to refresh cache stats:', error);
    }
}

function updateCacheDisplay(cacheType, cacheData) {
    if (!cacheData || !cacheData.path) {
        return;
    }

    const prefix = cacheType === 'primary' ? 'primary' : 'secondary';

    // Update path
    const pathEl = document.getElementById(`${prefix}-cache-path`);
    if (pathEl) {
        pathEl.textContent = cacheData.path;
        pathEl.title = cacheData.path; // Show full path on hover
    }

    // Update used
    const usedEl = document.getElementById(`${prefix}-cache-used`);
    if (usedEl) {
        usedEl.textContent = cacheData.used || '-';
    }

    // Update limit
    const limitEl = document.getElementById(`${prefix}-cache-limit`);
    if (limitEl) {
        limitEl.textContent = cacheData.limit || '-';
    }

    // Update progress bar
    const barEl = document.getElementById(`${prefix}-cache-bar`);
    const percentEl = document.getElementById(`${prefix}-cache-percent`);
    if (barEl && percentEl) {
        const usage = cacheData.usage_percent || 0;
        barEl.style.width = `${Math.min(usage, 100)}%`;
        percentEl.textContent = `${usage.toFixed(1)}%`;

        // Color based on usage
        if (usage >= 90) {
            barEl.style.backgroundColor = '#dc3545'; // Red
        } else if (usage >= 75) {
            barEl.style.backgroundColor = '#ffc107'; // Yellow
        } else {
            barEl.style.backgroundColor = '#28a745'; // Green
        }
    }

    // Update disk free
    const freeEl = document.getElementById(`${prefix}-disk-free`);
    if (freeEl) {
        freeEl.textContent = cacheData.disk_free || '-';
    }
}
