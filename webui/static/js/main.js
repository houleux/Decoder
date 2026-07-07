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
});

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
                <td><span class="badge">${shortId}</span></td>
                <td class="col-action">
                    <button class="btn-expand" data-id="${config.id}">View Data</button>
                </td>
            `;
            
            // Details Row
            const detailsRow = document.createElement('tr');
            detailsRow.className = 'details-row';
            detailsRow.id = `details-row-${config.id}`;
            detailsRow.innerHTML = `
                <td colspan="7">
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
