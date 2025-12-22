// DOM Elements
const elements = {
    simStatus: document.getElementById('sim-status'),
    currentRul: document.getElementById('current-rul'),
    simplexState: document.getElementById('simplex-state'),
    baselineRul: document.getElementById('baseline-rul'),
    complexRul: document.getElementById('complex-rul'),
    currentCycle: document.getElementById('current-cycle'),
    alertList: document.getElementById('alert-list'),
    alertCount: document.getElementById('alert-count'),

    // Accuracy
    rulError: document.getElementById('rul-error'),
    trueRul: document.getElementById('true-rul'),
    accuracyStatus: document.getElementById('accuracy-status'),

    // Controls
    btnPause: document.getElementById('btn-pause'),
    btnResume: document.getElementById('btn-resume'),
    btnReset: document.getElementById('btn-reset'),

    // Tabs & Views
    menuItems: document.querySelectorAll('.menu-item'),
    views: {
        dashboard: document.getElementById('view-dashboard'),
        fleet: document.getElementById('view-fleet'),
        settings: document.getElementById('view-settings')
    },

    // Chart Controls
    sensorButtons: document.querySelectorAll('.chart-controls button')
};

// State
let activeSensorSet = '1-4';

// Chart Configuration
const ctx = document.getElementById('sensorChart').getContext('2d');
const maxDataPoints = 50;

// Dataset configurations for different sensor sets
const sensorConfigs = {
    '1-4': [
        { label: 'Sensor 2 (Pressure)', idx: 1, color: '#38bdf8' },
        { label: 'Sensor 3 (Temp)', idx: 2, color: '#fb923c', dash: [5, 5] },
        { label: 'Sensor 4 (RPM)', idx: 3, color: '#4ade80' }
    ],
    '5-8': [
        { label: 'Sensor 6', idx: 5, color: '#38bdf8' },
        { label: 'Sensor 7', idx: 6, color: '#fb923c', dash: [5, 5] },
        { label: 'Sensor 8', idx: 7, color: '#4ade80' }
    ],
    '9-14': [
        { label: 'Sensor 9', idx: 8, color: '#38bdf8' },
        { label: 'Sensor 12', idx: 11, color: '#fb923c', dash: [5, 5] },
        { label: 'Sensor 14', idx: 13, color: '#4ade80' }
    ]
};

const sensorChart = new Chart(ctx, {
    type: 'line',
    data: {
        labels: [],
        datasets: [] // Will be populated dynamically
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        interaction: {
            mode: 'index',
            intersect: false,
        },
        plugins: {
            legend: { labels: { color: '#94a3b8' } }
        },
        scales: {
            x: {
                grid: { color: 'rgba(148, 163, 184, 0.1)' },
                ticks: { color: '#94a3b8' }
            },
            y: {
                grid: { color: 'rgba(148, 163, 184, 0.1)' },
                ticks: { color: '#94a3b8' }
            }
        }
    }
});

// Initialize Chart
updateChartConfig('1-4');

function updateChartConfig(setKey) {
    activeSensorSet = setKey;
    const config = sensorConfigs[setKey];

    sensorChart.data.datasets = config.map(c => ({
        label: c.label,
        data: [],
        borderColor: c.color,
        backgroundColor: c.color + '1A', // 10% opacity
        borderWidth: 2,
        tension: 0.4,
        borderDash: c.dash || [],
        fill: false,
        sensorIdx: c.idx
    }));
    sensorChart.data.labels = [];
    sensorChart.update();
}


// WebSocket Connection
const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
const wsUrl = `${protocol}//${window.location.host}/ws`;
let socket;

function connect() {
    socket = new WebSocket(wsUrl);

    socket.onopen = () => {
        console.log('Connected to server');
        elements.simStatus.textContent = 'Connected';
        elements.simStatus.style.color = '#4ade80';
    };

    socket.onclose = () => {
        console.log('Disconnected');
        elements.simStatus.textContent = 'Disconnected';
        elements.simStatus.style.color = '#f87171';
        setTimeout(connect, 3000);
    };

    socket.onmessage = (event) => {
        const data = JSON.parse(event.data);

        if (data.status === "Paused") {
            elements.simStatus.textContent = 'Paused';
            elements.simStatus.style.color = '#fb923c';
            elements.btnPause.disabled = true;
            elements.btnResume.disabled = false;
        } else if (data.status === "Running") {
            elements.simStatus.textContent = 'Running';
            elements.simStatus.style.color = '#4ade80';
            elements.btnPause.disabled = false;
            elements.btnResume.disabled = true;
            updateDashboard(data);
        } else {
            updateDashboard(data);
        }
    };
}

function updateDashboard(data) {
    if (!data.cycle) return;

    // 1. Update Text Fields
    elements.currentCycle.textContent = data.cycle;
    elements.currentRul.textContent = Math.round(data.rul.final);
    elements.baselineRul.textContent = Math.round(data.rul.baseline);
    elements.complexRul.textContent = Math.round(data.rul.complex);
    elements.simplexState.textContent = data.simplex_state;

    if (data.simplex_state === 'BASELINE') {
        elements.simplexState.style.color = '#fb923c';
    } else {
        elements.simplexState.style.color = '#4ade80';
    }

    // Update Accuracy
    elements.trueRul.textContent = Math.round(data.rul.true);
    const error = data.rul.error;
    elements.rulError.textContent = (error > 0 ? '+' : '') + error;

    elements.accuracyStatus.textContent = data.rul.status;
    if (data.rul.status === 'Good') {
        elements.rulError.className = 'error-value good';
        elements.accuracyStatus.style.background = 'var(--accent-green)';
        elements.accuracyStatus.style.color = 'var(--bg-dark)';
    } else {
        elements.rulError.className = 'error-value deviating';
        elements.accuracyStatus.style.background = 'var(--accent-red)';
        elements.accuracyStatus.style.color = 'var(--bg-dark)';
    }

    // 2. Update Charts
    updateChart(data);

    // 3. Update Alerts
    updateAlerts(data.alerts);
}

function updateChart(data) {
    sensorChart.data.labels.push(data.cycle);

    // Iterate over active datasets and push corresponding sensor data
    sensorChart.data.datasets.forEach(dataset => {
        const value = data.sensors[dataset.sensorIdx];
        dataset.data.push(value);
        if (dataset.data.length > maxDataPoints) dataset.data.shift();
    });

    if (sensorChart.data.labels.length > maxDataPoints) {
        sensorChart.data.labels.shift();
    }

    sensorChart.update();
}

function updateAlerts(alerts) {
    elements.alertCount.textContent = alerts.length;
    elements.alertList.innerHTML = '';

    if (alerts.length === 0) {
        elements.alertList.innerHTML = '<div class="empty-state">No active alerts</div>';
        return;
    }

    alerts.forEach(alert => {
        const div = document.createElement('div');
        div.className = `alert-item ${alert.level.toLowerCase()}`;
        div.textContent = alert.message;
        elements.alertList.appendChild(div);
    });
}

// --- Event Listeners ---

// 1. Simulation Controls
if (elements.btnPause) elements.btnPause.onclick = () => socket.send('pause');
if (elements.btnResume) elements.btnResume.onclick = () => socket.send('resume');
if (elements.btnReset) elements.btnReset.onclick = () => {
    socket.send('reset');
    updateChartConfig(activeSensorSet);
};

// 2. Tab Navigation
elements.menuItems.forEach(item => {
    item.addEventListener('click', () => {
        // Remove active class from all
        elements.menuItems.forEach(i => i.classList.remove('active'));
        // Add to clicked
        item.classList.add('active');

        // Switch Views
        const targetId = ['view-dashboard', 'view-fleet', 'view-settings'][Array.from(elements.menuItems).indexOf(item)];
        if (targetId) {
            Object.values(elements.views).forEach(el => {
                if (el) el.classList.add('hidden');
            });
            const targetEl = document.getElementById(targetId);
            if (targetEl) targetEl.classList.remove('hidden');
        }
    });
});

// 3. Chart Filters
elements.sensorButtons.forEach(btn => {
    btn.addEventListener('click', () => {
        elements.sensorButtons.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        const set = btn.getAttribute('data-sensors');
        updateChartConfig(set);
    });
});

// Start
connect();
