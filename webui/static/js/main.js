let configMetadata = {}; // config_id -> { data }
let selectedConfigs = new Set();
let dynamicParams = {}; // key -> { type: 'string'|'number', values: Set }

document.addEventListener('DOMContentLoaded', () => {
    fetchConfigs();
    
    document.getElementById('select-all').addEventListener('change', (e) => {
        const checkboxes = document.querySelectorAll('.row-checkbox');
        checkboxes.forEach(cb => {
            cb.checked = e.target.checked;
            if (cb.checked) {
                selectedConfigs.add(cb.value);
            } else {
                selectedConfigs.delete(cb.value);
            }
        });
        updatePlotButton();
    });
    
    document.getElementById('plot-selected-btn').addEventListener('click', plotSelected);
    document.getElementById('close-charts-btn').addEventListener('click', () => {
        document.getElementById('charts-section').classList.add('hidden');
    });
    
    document.getElementById('toggle-filters-btn').addEventListener('click', () => {
        const panel = document.getElementById('filters-panel');
        if (panel.classList.contains('hidden')) {
            panel.classList.remove('hidden');
            document.getElementById('toggle-filters-btn').textContent = 'Hide Filters';
        } else {
            panel.classList.add('hidden');
            document.getElementById('toggle-filters-btn').textContent = 'Show Filters';
        }
    });
    
    document.getElementById('reset-filters-btn').addEventListener('click', () => {
        document.getElementById('search-filter').value = '';
        document.querySelectorAll('.filter-input').forEach(input => input.value = '');
        applyFilters();
    });
    
    document.getElementById('search-filter').addEventListener('input', applyFilters);
    
    // Extend Modal Logic
    document.getElementById('close-extend-btn').addEventListener('click', () => {
        document.getElementById('extend-modal').classList.add('hidden');
    });
    
    document.getElementById('copy-cmd-btn').addEventListener('click', () => {
        const cmd = document.getElementById('ext-cmd-preview').value;
        navigator.clipboard.writeText(cmd).then(() => {
            const btn = document.getElementById('copy-cmd-btn');
            const orig = btn.textContent;
            btn.textContent = 'Copied!';
            setTimeout(() => btn.textContent = orig, 1500);
        });
    });
    
    // Generate command preview automatically
    const extFormInputs = document.querySelectorAll('#extend-form input');
    extFormInputs.forEach(input => {
        input.addEventListener('input', updateExtendCommandPreview);
    });

    document.getElementById('extend-form').addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const form = e.target;
        const submitBtn = document.getElementById('submit-extend-btn');
        const statusMsg = document.getElementById('extend-status');
        
        submitBtn.disabled = true;
        submitBtn.textContent = 'Launching...';
        statusMsg.textContent = '';
        
        const payload = {
            matrix: form.matrix.value,
            methods: [form.methods.value],
            zVals: form.zVals.value,
            seed: form.seed.value,
            lMax: form.lMax.value,
            trainEpisodes: form.trainEpisodes.value,
            evalSnrs: form.evalSnrs.value,
            maxFrames: form.maxFrames.value,
            targetFrameErrors: form.targetFrameErrors.value,
            workers: form.workers.value
        };
        
        try {
            const response = await fetch('/api/run_experiment', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const data = await response.json();
            
            if (response.ok) {
                statusMsg.textContent = `Started successfully! (PID: ${data.pid})`;
                statusMsg.style.color = "#4ade80";
                setTimeout(() => {
                    document.getElementById('extend-modal').classList.add('hidden');
                    submitBtn.disabled = false;
                    submitBtn.textContent = 'Run in Background';
                    statusMsg.textContent = '';
                }, 2000);
            } else {
                statusMsg.textContent = `Error: ${data.error}`;
                statusMsg.style.color = "#ef4444";
                submitBtn.disabled = false;
                submitBtn.textContent = 'Run in Background';
            }
        } catch (error) {
            statusMsg.textContent = `Error: ${error.message}`;
            statusMsg.style.color = "#ef4444";
            submitBtn.disabled = false;
            submitBtn.textContent = 'Run in Background';
        }
    });
    
    // Launch Modal Logic
    document.getElementById('launch-experiment-btn').addEventListener('click', () => {
        document.getElementById('launch-modal').classList.remove('hidden');
        fetchMatrices();
        fetchMethods();
    });
    
    document.getElementById('close-modal-btn').addEventListener('click', () => {
        document.getElementById('launch-modal').classList.add('hidden');
    });
    
    document.getElementById('copy-launch-cmd-btn').addEventListener('click', () => {
        const cmd = document.getElementById('launch-cmd-preview').value;
        navigator.clipboard.writeText(cmd).then(() => {
            const btn = document.getElementById('copy-launch-cmd-btn');
            const orig = btn.textContent;
            btn.textContent = 'Copied!';
            setTimeout(() => btn.textContent = orig, 1500);
        });
    });
    
    document.getElementById('launch-form').addEventListener('input', updateLaunchCommandPreview);
    document.getElementById('launch-form').addEventListener('change', updateLaunchCommandPreview);
    
    document.getElementById('launch-form').addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const form = e.target;
        const submitBtn = document.getElementById('submit-launch-btn');
        const statusMsg = document.getElementById('launch-status');
        
        // Collect checked methods
        const methods = Array.from(form.querySelectorAll('input[name="methods"]:checked')).map(cb => cb.value);
        if (methods.length === 0) {
            statusMsg.textContent = "Error: Select at least one method";
            statusMsg.style.color = "#ef4444";
            return;
        }
        
        submitBtn.disabled = true;
        submitBtn.textContent = 'Launching...';
        statusMsg.textContent = '';
        
        const payload = {
            matrix: form.matrix.value,
            methods: methods,
            zVals: form.zVals.value,
            lMax: form.lMax.value,
            trainSnrs: form.trainSnrs.value,
            evalSnrs: form.evalSnrs.value,
            trainEpisodes: form.trainEpisodes.value,
            maxFrames: form.maxFrames.value,
            workers: form.workers.value
        };
        
        try {
            const response = await fetch('/api/run_experiment', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const data = await response.json();
            
            if (response.ok) {
                statusMsg.textContent = `Started successfully! (PID: ${data.pid})`;
                statusMsg.style.color = "#4ade80";
                setTimeout(() => {
                    document.getElementById('launch-modal').classList.add('hidden');
                    submitBtn.disabled = false;
                    submitBtn.textContent = 'Launch';
                    statusMsg.textContent = '';
                }, 2000);
            } else {
                statusMsg.textContent = `Error: ${data.error}`;
                statusMsg.style.color = "#ef4444";
                submitBtn.disabled = false;
                submitBtn.textContent = 'Launch';
            }
        } catch (error) {
            statusMsg.textContent = `Error: ${error.message}`;
            statusMsg.style.color = "#ef4444";
            submitBtn.disabled = false;
            submitBtn.textContent = 'Launch';
        }
    });
});

async function fetchMatrices() {
    try {
        const response = await fetch('/api/matrices');
        const data = await response.json();
        const select = document.getElementById('matrix-select');
        
        // Save current selection if any
        const currentVal = select.value;
        select.innerHTML = '';
        
        data.matrices.forEach(m => {
            const opt = document.createElement('option');
            opt.value = m;
            opt.textContent = m;
            select.appendChild(opt);
        });
        
        if (currentVal && data.matrices.includes(currentVal)) {
            select.value = currentVal;
        }
        
        updateLaunchCommandPreview();
    } catch (e) {
        console.error("Failed to load matrices", e);
    }
}

async function fetchMethods() {
    try {
        const response = await fetch('/api/methods');
        const data = await response.json();
        const container = document.getElementById('methods-checkboxes');
        
        // Only populate if empty to avoid losing selection on reopen
        if (container.children.length > 0) return;
        
        container.innerHTML = '';
        data.methods.forEach(m => {
            const label = document.createElement('label');
            label.innerHTML = `<input type="checkbox" name="methods" value="${m}"> ${m}`;
            container.appendChild(label);
        });
        
        updateLaunchCommandPreview();
    } catch (e) {
        console.error("Failed to load methods", e);
    }
}

function updateLaunchCommandPreview() {
    const form = document.getElementById('launch-form');
    const methods = Array.from(form.querySelectorAll('input[name="methods"]:checked')).map(cb => cb.value);
    
    let cmd = `python3 run_experiments.py`;
    
    if (form.matrix.value) cmd += ` --matrix ${form.matrix.value}`;
    if (methods.length > 0) cmd += ` --methods ${methods.join(' ')}`;
    if (form.zVals.value) cmd += ` --z-vals ${form.zVals.value}`;
    if (form.trainSnrs.value) cmd += ` --train-snrs ${form.trainSnrs.value}`;
    if (form.evalSnrs.value) cmd += ` --eval-snrs ${form.evalSnrs.value}`;
    if (form.trainEpisodes.value) cmd += ` --train-episodes ${form.trainEpisodes.value}`;
    if (form.maxFrames.value) cmd += ` --max-frames ${form.maxFrames.value}`;
    if (form.workers.value) cmd += ` --workers ${form.workers.value}`;
    if (form.lMax && form.lMax.value) cmd += ` --l-max ${form.lMax.value}`;
    
    document.getElementById('launch-cmd-preview').value = cmd;
}

async function fetchConfigs() {
    try {
        const response = await fetch('/api/configs');
        const configs = await response.json();
        
        const tbody = document.getElementById('experiments-body');
        tbody.innerHTML = '';
        
        configs.forEach(config => {
            const matrixFull = config.data.matrix || 'Unknown';
            const matrix = matrixFull.split('/').pop();
            const method = config.data.method || 'Unknown';
            const z = config.data.z || '?';
            const eps = config.data.train_episodes || '?';
            const shortId = config.id.substring(0, 8);
            
            configMetadata[config.id] = { method, z, eps, matrix, title: `${method} (z=${z}, ${matrix})`, raw: config.data };
            
            // Main Row
            const mainRow = document.createElement('tr');
            mainRow.className = 'main-row';
            mainRow.innerHTML = `
                <td class="col-check">
                    <input type="checkbox" class="row-checkbox" value="${config.id}">
                </td>
                <td>${matrix}</td>
                <td>${method}</td>
                <td>${z}</td>
                <td>${eps}</td>
                <td>${config.frames_done.toLocaleString()}</td>
                <td><span class="badge">${shortId}</span></td>
                <td class="col-action">
                    <button class="btn-expand" data-id="${config.id}" style="margin-bottom: 4px;">View Data</button><br>
                    <button class="btn-extend" data-id="${config.id}" style="background: transparent; color: #4ade80; border: 1px solid var(--border-color); padding: 4px 8px; font-size: 0.75rem; border-radius: 4px; cursor: pointer; margin-bottom: 4px;">Extend Eval</button><br>
                    <button class="btn-copy-config" data-id="${config.id}" style="background: transparent; color: #60a5fa; border: 1px solid var(--border-color); padding: 4px 8px; font-size: 0.75rem; border-radius: 4px; cursor: pointer;">Copy Config</button>
                </td>
            `;
            
            // Details Row
            const detailsRow = document.createElement('tr');
            detailsRow.className = 'details-row';
            detailsRow.id = `details-row-${config.id}`;
            detailsRow.innerHTML = `
                <td colspan="8">
                    <div class="details-content" id="details-${config.id}">
                        <!-- Loading... -->
                    </div>
                </td>
            `;
            
            // Events
            const checkbox = mainRow.querySelector('.row-checkbox');
            checkbox.addEventListener('change', (e) => {
                if (e.target.checked) selectedConfigs.add(config.id);
                else selectedConfigs.delete(config.id);
                updatePlotButton();
            });
            
            const expandBtn = mainRow.querySelector('.btn-expand');
            expandBtn.addEventListener('click', async () => {
                const isOpen = detailsRow.classList.contains('open');
                if (isOpen) {
                    detailsRow.classList.remove('open');
                    expandBtn.textContent = 'View Data';
                } else {
                    detailsRow.classList.add('open');
                    detailsRow.style.display = ''; // Ensure visible if filter is active
                    expandBtn.textContent = 'Hide Data';
                    await loadRowDetails(config.id);
                }
            });
            
            const extendBtn = mainRow.querySelector('.btn-extend');
            extendBtn.addEventListener('click', () => {
                openExtendModal(config.id);
            });
            
            const copyConfigBtn = mainRow.querySelector('.btn-copy-config');
            copyConfigBtn.addEventListener('click', () => {
                openLaunchModalWithConfig(config.id);
            });
            
            tbody.appendChild(mainRow);
            tbody.appendChild(detailsRow);
            
            // Extract dynamic params
            Object.keys(config.data).forEach(key => {
                const val = config.data[key];
                if (val === null || val === undefined) return;
                
                if (!dynamicParams[key]) {
                    dynamicParams[key] = { type: typeof val, values: new Set() };
                }
                // If we see a string, force type to string
                if (typeof val === 'string') dynamicParams[key].type = 'string';
                
                if (typeof val === 'string' || typeof val === 'boolean') {
                    dynamicParams[key].values.add(String(val));
                }
            });
        });
        
        renderDynamicFilters();
        
    } catch (error) {
        console.error('Error fetching configs:', error);
    }
}

function renderDynamicFilters() {
    const container = document.getElementById('dynamic-filters-container');
    container.innerHTML = '';
    
    const skipKeys = ['seed', 'workers', 'chunk_size', 'train_snr_vals']; // Skip execution noise
    
    Object.keys(dynamicParams).sort().forEach(key => {
        if (skipKeys.includes(key)) return;
        
        const param = dynamicParams[key];
        const group = document.createElement('div');
        group.className = 'filter-group';
        
        let labelText = key.replace(/_/g, ' ');
        group.innerHTML = `<label>${labelText}</label>`;
        
        if (param.type === 'string' || param.type === 'boolean') {
            let select = `<select class="filter-input param-filter" data-key="${key}" data-type="string">
                <option value="">All</option>`;
            Array.from(param.values).sort().forEach(val => {
                select += `<option value="${val}">${val}</option>`;
            });
            select += `</select>`;
            group.innerHTML += select;
        } else if (param.type === 'number') {
            group.innerHTML += `
                <div class="filter-range">
                    <input type="number" step="any" class="filter-input param-filter param-min" data-key="${key}" placeholder="Min">
                    <input type="number" step="any" class="filter-input param-filter param-max" data-key="${key}" placeholder="Max">
                </div>
            `;
        }
        
        container.appendChild(group);
    });
    
    // Bind events
    document.querySelectorAll('.param-filter').forEach(input => {
        input.addEventListener('input', applyFilters);
    });
}

function applyFilters() {
    const searchTerm = document.getElementById('search-filter').value.toLowerCase();
    
    // Gather dynamic filters
    const filterState = {};
    document.querySelectorAll('.param-filter').forEach(input => {
        const key = input.dataset.key;
        if (!input.value) return;
        
        if (!filterState[key]) filterState[key] = {};
        
        if (input.tagName === 'SELECT') {
            filterState[key].val = input.value;
        } else {
            if (input.classList.contains('param-min')) filterState[key].min = Number(input.value);
            if (input.classList.contains('param-max')) filterState[key].max = Number(input.value);
        }
    });

    const rows = document.querySelectorAll('.main-row');
    rows.forEach(row => {
        const text = row.textContent.toLowerCase();
        const id = row.querySelector('.btn-expand').dataset.id;
        const detailsRow = document.getElementById(`details-row-${id}`);
        const rawData = configMetadata[id].raw;
        
        let match = true;
        
        // Text search
        if (searchTerm && !text.includes(searchTerm)) {
            match = false;
        }
        
        // Dynamic filters
        if (match) {
            for (const key in filterState) {
                const rules = filterState[key];
                const rawVal = rawData[key];
                
                if (rules.val !== undefined && String(rawVal) !== rules.val) {
                    match = false;
                    break;
                }
                if (rules.min !== undefined && Number(rawVal) < rules.min) {
                    match = false;
                    break;
                }
                if (rules.max !== undefined && Number(rawVal) > rules.max) {
                    match = false;
                    break;
                }
            }
        }
        
        if (match) {
            row.style.display = '';
            if (detailsRow.classList.contains('open')) detailsRow.style.display = '';
        } else {
            row.style.display = 'none';
            detailsRow.style.display = 'none';
        }
    });
}

async function openLaunchModalWithConfig(configId) {
    const raw = configMetadata[configId].raw;
    
    // Ensure dropdowns and checkboxes are loaded
    await Promise.all([fetchMatrices(), fetchMethods()]);
    
    const form = document.getElementById('launch-form');
    
    // Prefill fields
    if (raw.matrix && form.matrix.querySelector(`option[value="${raw.matrix}"]`)) {
        form.matrix.value = raw.matrix;
    }
    
    // Clear existing method checkboxes and check the correct one
    const methodCheckboxes = form.querySelectorAll('input[name="methods"]');
    methodCheckboxes.forEach(cb => {
        cb.checked = (cb.value === raw.method);
    });
    
    if (raw.z !== undefined) form.zVals.value = raw.z;
    if (raw.l_max !== undefined) form.lMax.value = raw.l_max;
    if (raw.train_episodes !== undefined) form.trainEpisodes.value = raw.train_episodes;
    if (raw.workers !== undefined) form.workers.value = raw.workers;
    
    if (raw.train_snr_vals) {
        form.trainSnrs.value = Array.isArray(raw.train_snr_vals) ? raw.train_snr_vals.join(' ') : String(raw.train_snr_vals);
    }
    
    updateLaunchCommandPreview();
    
    document.getElementById('launch-modal').classList.remove('hidden');
}

function openExtendModal(configId) {
    const raw = configMetadata[configId].raw;
    
    // Read-only text display
    document.getElementById('ext-matrix').textContent = raw.matrix;
    document.getElementById('ext-method').textContent = raw.method;
    document.getElementById('ext-z').textContent = raw.z;
    document.getElementById('ext-seed').textContent = raw.seed;
    document.getElementById('ext-lmax').textContent = raw.l_max;
    document.getElementById('ext-train-eps').textContent = raw.train_episodes;
    
    // Hidden inputs for form submit
    document.getElementById('ext-matrix-val').value = raw.matrix;
    document.getElementById('ext-method-val').value = raw.method;
    document.getElementById('ext-z-val').value = raw.z;
    document.getElementById('ext-seed-val').value = raw.seed;
    document.getElementById('ext-lmax-val').value = raw.l_max;
    document.getElementById('ext-train-eps-val').value = raw.train_episodes;
    
    // Editable defaults
    // Default to existing SNRs
    if (raw.train_snr_vals) {
        document.getElementById('ext-eval-snrs').value = Array.isArray(raw.train_snr_vals) ? raw.train_snr_vals.join(' ') : String(raw.train_snr_vals);
    } else {
        document.getElementById('ext-eval-snrs').value = '1.0 1.5 2.0 2.5 3.0';
    }
    document.getElementById('ext-max-frames').value = 5000; // Sensible default for extending
    document.getElementById('ext-tfe').value = 100000;
    document.getElementById('ext-workers').value = raw.workers || 40;
    
    updateExtendCommandPreview();
    
    document.getElementById('extend-modal').classList.remove('hidden');
}

function updateExtendCommandPreview() {
    const form = document.getElementById('extend-form');
    
    let cmd = `python3 run_experiments.py --matrix ${form.matrix.value} --methods ${form.methods.value}`;
    if (form.zVals.value) cmd += ` --z-vals ${form.zVals.value}`;
    if (form.evalSnrs.value) cmd += ` --eval-snrs ${form.evalSnrs.value}`;
    if (form.maxFrames.value) cmd += ` --max-frames ${form.maxFrames.value}`;
    if (form.targetFrameErrors.value) cmd += ` --target-frame-errors ${form.targetFrameErrors.value}`;
    if (form.workers.value) cmd += ` --workers ${form.workers.value}`;
    if (form.seed.value) cmd += ` --seed ${form.seed.value}`;
    if (form.lMax.value) cmd += ` --l-max ${form.lMax.value}`;
    if (form.trainEpisodes.value) cmd += ` --train-episodes ${form.trainEpisodes.value}`;
    
    document.getElementById('ext-cmd-preview').value = cmd;
}

function updatePlotButton() {
    const btn = document.getElementById('plot-selected-btn');
    btn.textContent = `Plot Selected (${selectedConfigs.size})`;
    btn.disabled = selectedConfigs.size === 0;
}

async function loadRowDetails(configId) {
    const container = document.getElementById(`details-${configId}`);
    if (container.dataset.loaded) return; // Already loaded
    
    container.innerHTML = 'Loading...';
    
    try {
        const response = await fetch(`/api/configs/${configId}`);
        const data = await response.json();
        
        let tableRows = '';
        data.evals.forEach(ev => {
            tableRows += `
                <tr>
                    <td>${ev.snr_db}</td>
                    <td>${ev.max_frames}</td>
                    <td>${ev.frames_done} ${ev.completed ? '✓' : ''}</td>
                    <td>${ev.ber.toExponential(4)}</td>
                    <td>${ev.fer.toExponential(4)}</td>
                    <td>${ev.avg_messages.toFixed(2)}</td>
                </tr>
            `;
        });
        
        container.innerHTML = `
            <table class="mini-table">
                <thead>
                    <tr>
                        <th>SNR (dB)</th>
                        <th>Max Frames</th>
                        <th>Frames Done</th>
                        <th>BER</th>
                        <th>FER</th>
                        <th>Avg Msgs</th>
                    </tr>
                </thead>
                <tbody>
                    ${tableRows}
                </tbody>
            </table>
            <div class="raw-config">${JSON.stringify(data.config, null, 2)}</div>
        `;
        container.dataset.loaded = "true";
        
    } catch (error) {
        container.innerHTML = `<div style="color: red;">Failed to load data.</div>`;
    }
}

async function plotSelected() {
    if (selectedConfigs.size === 0) return;
    
    // Show loading state
    document.getElementById('charts-section').classList.remove('hidden');
    window.scrollTo({ top: 0, behavior: 'smooth' });
    document.getElementById('matplotlib-output').style.opacity = '0.5';
    
    try {
        const response = await fetch('/api/plot', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ config_ids: Array.from(selectedConfigs) })
        });
        const data = await response.json();
        
        const img = document.getElementById('matplotlib-output');
        const dlBtn = document.getElementById('download-plot-btn');
        
        img.src = data.image;
        img.style.opacity = '1';
        
        // Setup download button
        dlBtn.href = data.image;
        dlBtn.download = "duckdb_plot.png";
        
    } catch (error) {
        console.error('Error fetching plot data:', error);
        document.getElementById('matplotlib-output').style.opacity = '1';
    }
}
