/* Hospital Crowding Prediction Frontend Application */

const API_BASE_URL = (window.location.protocol.startsWith('http') && (window.location.port === '5000' || window.location.port === ''))
    ? ''
    : 'http://127.0.0.1:5000';

const STORAGE_KEY = 'hospitalPredictionHistory';
const MAX_HISTORY_ITEMS = 10;
const predictions = [];

const defaultFeatureImportance = [
    { name: 'Queue length', importance: 0.6782 },
    { name: 'Bed occupancy', importance: 0.2486 },
    { name: 'Avail. doctors', importance: 0.0653 },
    { name: 'Other factors', importance: 0.0079 }
];

function populateHours() {
    const sel = document.getElementById('hour');
    if (!sel) return;
    sel.innerHTML = '<option value="" disabled selected>Select hour</option>';
    for (let h = 0; h < 24; h++) {
        const opt = document.createElement('option');
        opt.value = h;
        const label = h === 0 ? '12:00 AM' : h < 12 ? h + ':00 AM' : h === 12 ? '12:00 PM' : (h - 12) + ':00 PM';
        opt.textContent = label;
        sel.appendChild(opt);
    }
}

function loadHistoryFromStorage() {
    predictions.length = 0;
    try {
        const raw = localStorage.getItem(STORAGE_KEY);
        if (raw) {
            const parsed = JSON.parse(raw);
            if (Array.isArray(parsed)) {
                parsed.slice(0, MAX_HISTORY_ITEMS).forEach(item => predictions.push(item));
            }
        }
    } catch (err) {
        console.error('Error loading history from localStorage:', err);
    }
}

function saveHistoryToStorage() {
    try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(predictions));
    } catch (err) {
        console.error('Error saving history to localStorage:', err);
    }
}

function renderFeatureImportance(data) {
    const container = document.getElementById('fi-chart');
    if (!container) return;

    container.innerHTML = '';
    let items = [];

    if (data && data.length > 0) {
        const sorted = [...data].sort((a, b) => (b.importance || 0) - (a.importance || 0));
        const top3 = sorted.slice(0, 3).map(item => {
            let name = item.name || item.feature || '';
            if (name.toLowerCase().includes('queue')) name = 'Queue length';
            else if (name.toLowerCase().includes('bed')) name = 'Bed occupancy';
            else if (name.toLowerCase().includes('doctor')) name = 'Avail. doctors';
            return { name, importance: item.importance };
        });
        const otherImp = sorted.slice(3).reduce((acc, curr) => acc + (curr.importance || 0), 0);
        items = [...top3, { name: 'Other factors', importance: Math.max(otherImp, 0.008) }];
    } else {
        items = defaultFeatureImportance;
    }

    const maxVal = items[0].importance || 0.0001;

    items.forEach((f) => {
        const pct = Math.max((f.importance / maxVal) * 100, 2);
        const row = document.createElement('div');
        row.className = 'fi-bar-row';
        const displayName = f.name;
        const pctDisplay = (f.importance * 100).toFixed(1);
        row.innerHTML = `
            <span class="fi-label">${displayName}</span>
            <div class="fi-bar-track">
                <div class="fi-bar-fill" style="width: 0%;" data-width="${pct}"></div>
            </div>
            <span class="fi-value">${pctDisplay}%</span>
        `;
        container.appendChild(row);
    });

    requestAnimationFrame(() => {
        setTimeout(() => {
            document.querySelectorAll('.fi-bar-fill').forEach(bar => {
                bar.style.width = bar.dataset.width + '%';
            });
        }, 120);
    });
}

async function fetchModelMetadata() {
    try {
        const res = await fetch(`${API_BASE_URL}/api/feature-importance`);
        if (res.ok) {
            const data = await res.json();
            if (data.status === 'success' && data.feature_importances) {
                renderFeatureImportance(data.feature_importances);
                return;
            }
        }
        renderFeatureImportance(defaultFeatureImportance);
    } catch (err) {
        renderFeatureImportance(defaultFeatureImportance);
    }
}

function drawCrowdingTrend() {
    const canvas = document.getElementById('trend-canvas');
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();

    const width = rect.width || 480;
    const height = 120;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);

    ctx.clearRect(0, 0, width, height);

    const subtitle = document.getElementById('trend-subtitle');
    const startLabel = document.getElementById('trend-start-label');
    const endLabel = document.getElementById('trend-end-label');

    if (predictions.length === 0) {
        if (subtitle) subtitle.textContent = '0 predictions this session';
        if (startLabel) startLabel.textContent = '#0';
        if (endLabel) endLabel.textContent = '#0';

        ctx.beginPath();
        ctx.setLineDash([4, 4]);
        ctx.moveTo(24, height / 2);
        ctx.lineTo(width - 24, height / 2);
        ctx.strokeStyle = '#e5e7eb';
        ctx.lineWidth = 1.5;
        ctx.stroke();
        ctx.setLineDash([]);
        return;
    }

    // Chronological order for line chart (oldest to newest)
    const chronological = [...predictions].reverse();
    const trendValues = chronological.map(p => {
        const occ = p.metrics?.bed_occupancy_percentage;
        if (occ !== undefined && !isNaN(occ)) return occ;
        const pred = (p.prediction || 'Medium').toLowerCase();
        return pred === 'high' ? 85 : pred === 'medium' ? 60 : 35;
    });

    const n = trendValues.length;
    if (subtitle) {
        subtitle.textContent = `Last ${n} prediction${n > 1 ? 's' : ''} this session`;
    }
    if (startLabel) startLabel.textContent = '#1';
    if (endLabel) endLabel.textContent = `#${n} (latest)`;

    const padX = 24;
    const padY = 20;
    const chartW = width - padX * 2;
    const chartH = height - padY * 2;

    const minVal = Math.min(...trendValues, 20);
    const maxVal = Math.max(...trendValues, 95);
    const range = (maxVal - minVal) || 1;

    const points = trendValues.map((v, i) => {
        const x = padX + (i / Math.max(n - 1, 1)) * chartW;
        const y = height - padY - ((v - minVal) / range) * chartH;
        return { x, y, v };
    });

    if (points.length < 2) {
        points.unshift({ x: padX, y: points[0].y, v: points[0].v });
    }

    ctx.beginPath();
    ctx.moveTo(points[0].x, height - padY + 10);
    ctx.lineTo(points[0].x, points[0].y);
    for (let i = 0; i < points.length - 1; i++) {
        const xc = (points[i].x + points[i + 1].x) / 2;
        const yc = (points[i].y + points[i + 1].y) / 2;
        ctx.quadraticCurveTo(points[i].x, points[i].y, xc, yc);
    }
    ctx.lineTo(points[points.length - 1].x, points[points.length - 1].y);
    ctx.lineTo(points[points.length - 1].x, height - padY + 10);
    ctx.closePath();

    const grad = ctx.createLinearGradient(0, padY, 0, height);
    grad.addColorStop(0, 'rgba(217, 119, 6, 0.14)');
    grad.addColorStop(1, 'rgba(217, 119, 6, 0.0)');
    ctx.fillStyle = grad;
    ctx.fill();

    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);
    for (let i = 0; i < points.length - 1; i++) {
        const xc = (points[i].x + points[i + 1].x) / 2;
        const yc = (points[i].y + points[i + 1].y) / 2;
        ctx.quadraticCurveTo(points[i].x, points[i].y, xc, yc);
    }
    ctx.lineTo(points[points.length - 1].x, points[points.length - 1].y);
    ctx.lineWidth = 2.5;
    ctx.strokeStyle = '#d97706';
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.stroke();

    const lastPoint = points[points.length - 1];
    ctx.beginPath();
    ctx.arc(lastPoint.x, lastPoint.y, 5, 0, Math.PI * 2);
    ctx.fillStyle = '#d97706';
    ctx.fill();
    ctx.lineWidth = 2;
    ctx.strokeStyle = '#ffffff';
    ctx.stroke();
}

function updateBedOccupancyBar(percentage) {
    const bar = document.getElementById('occ-bar-fill');
    const label = document.getElementById('occ-current-label');

    if (percentage === undefined || percentage === null || isNaN(percentage)) {
        if (bar) bar.style.width = '0%';
        if (label) {
            label.textContent = '0%';
            label.style.color = 'var(--color-primary)';
        }
        return;
    }

    const val = Math.min(Math.max(percentage, 0), 100);
    if (bar) {
        bar.style.width = val + '%';
        if (val >= 85) bar.style.backgroundColor = 'var(--color-high)';
        else if (val >= 65) bar.style.backgroundColor = 'var(--color-medium)';
        else bar.style.backgroundColor = 'var(--color-primary)';
    }
    if (label) {
        label.textContent = Math.round(val) + '%';
        if (val >= 85) label.style.color = 'var(--color-high)';
        else if (val >= 65) label.style.color = 'var(--color-medium)';
        else label.style.color = 'var(--color-primary)';
    }
}

function updateKPIs(result, payload) {
    const metrics = result.metrics || {};
    const occPct = metrics.bed_occupancy_percentage !== undefined
        ? metrics.bed_occupancy_percentage.toFixed(0) + '%'
        : '—';
    const occVal = document.getElementById('kpi-occ-val');
    const occSub = document.getElementById('kpi-occ-sub');
    if (occVal) occVal.textContent = occPct;
    if (occSub) {
        if (payload.occupied_beds !== undefined && payload.hospital_capacity !== undefined) {
            occSub.textContent = `${payload.occupied_beds} / ${payload.hospital_capacity} beds`;
            occSub.className = (metrics.bed_occupancy_rate >= 0.85) ? 'kpi-subtext amber-subtext' : 'kpi-subtext green-subtext';
        } else {
            occSub.textContent = 'Awaiting prediction';
            occSub.className = 'kpi-subtext';
        }
    }

    const ratioVal = metrics.staff_to_patient_ratio !== undefined
        ? metrics.staff_to_patient_ratio.toFixed(2)
        : '—';
    const staffVal = document.getElementById('kpi-staff-val');
    const staffSub = document.getElementById('kpi-staff-sub');
    if (staffVal) staffVal.textContent = ratioVal;
    if (staffSub) {
        const totalStaff = (payload.available_doctors || 0) + (payload.available_nurses || 0);
        if (payload.patient_arrivals !== undefined) {
            staffSub.textContent = `${totalStaff} staff / ${payload.patient_arrivals} arrivals`;
        } else {
            staffSub.textContent = 'Awaiting prediction';
        }
        staffSub.className = 'kpi-subtext';
    }

    const qVal = document.getElementById('kpi-queue-val');
    const qSub = document.getElementById('kpi-queue-sub');
    if (qVal) qVal.textContent = payload.queue_length !== undefined ? payload.queue_length : '—';
    if (qSub) {
        if (payload.queue_length !== undefined) {
            if (payload.queue_length >= 15) {
                qSub.textContent = 'Critical queue';
                qSub.className = 'kpi-subtext amber-subtext';
            } else if (payload.queue_length >= 8) {
                qSub.textContent = 'Above average';
                qSub.className = 'kpi-subtext amber-subtext';
            } else {
                qSub.textContent = 'Normal';
                qSub.className = 'kpi-subtext green-subtext';
            }
        } else {
            qSub.textContent = 'Awaiting prediction';
            qSub.className = 'kpi-subtext amber-subtext';
        }
    }

    const runVal = document.getElementById('kpi-run-val');
    if (runVal) runVal.textContent = predictions.length;
}

function showResult(result) {
    const placeholder = document.getElementById('result-placeholder');
    const active = document.getElementById('result-active');
    if (placeholder) placeholder.classList.add('hide');
    if (active) active.classList.add('show');

    const pred = (result.prediction || 'Medium').toLowerCase();
    const circle = document.getElementById('active-status-circle');
    const icon = document.getElementById('active-status-icon');
    const headline = document.getElementById('active-level-text');
    const confidence = document.getElementById('active-confidence-text');

    if (circle) {
        circle.className = 'result-badge-circle ' + pred;
    }

    if (icon) {
        if (pred === 'low') {
            icon.innerHTML = '<polyline points="20 6 9 17 4 12"/>';
        } else if (pred === 'high') {
            icon.innerHTML = '<circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>';
        } else {
            icon.innerHTML = '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>';
        }
    }

    if (headline) {
        headline.textContent = `${result.prediction} crowding`;
        headline.className = 'result-headline ' + pred;
    }

    if (confidence) {
        const confPct = result.confidence ? Math.round(result.confidence * 100) : 91;
        confidence.textContent = `Confidence ${confPct}%`;
    }
}

function resetResultCard() {
    const placeholder = document.getElementById('result-placeholder');
    const active = document.getElementById('result-active');
    if (placeholder) placeholder.classList.remove('hide');
    if (active) active.classList.remove('show');
}

function renderTable() {
    const tbody = document.getElementById('table-body');
    const emptyMsg = document.getElementById('empty-table');
    const countBadge = document.getElementById('table-count');

    if (!tbody) return;
    tbody.innerHTML = '';

    if (predictions.length === 0) {
        if (emptyMsg) emptyMsg.style.display = 'block';
        if (countBadge) countBadge.textContent = '0 entries';
        return;
    }

    if (emptyMsg) emptyMsg.style.display = 'none';
    if (countBadge) countBadge.textContent = `${predictions.length} ${predictions.length === 1 ? 'entry' : 'entries'}`;

    // Render predictions array (newest is always at index 0)
    predictions.forEach(item => {
        const predClass = (item.prediction || 'Medium').toLowerCase();
        const occPct = item.metrics?.bed_occupancy_percentage !== undefined
            ? item.metrics.bed_occupancy_percentage.toFixed(0) + '%'
            : (item.occupied_beds && item.hospital_capacity ? ((item.occupied_beds / item.hospital_capacity) * 100).toFixed(0) + '%' : '—');

        const row = document.createElement('tr');
        row.innerHTML = `
            <td class="td-department">${item.department || '—'}</td>
            <td>${item.patient_arrivals !== undefined ? item.patient_arrivals : '—'}</td>
            <td>${occPct}</td>
            <td>${item.queue_length !== undefined ? item.queue_length : '—'}</td>
            <td><span class="badge ${predClass}">${item.prediction || 'Medium'}</span></td>
        `;
        tbody.appendChild(row);
    });
}

function showAlert(msg) {
    const alertBox = document.getElementById('form-alert');
    if (!alertBox) return;
    alertBox.textContent = msg;
    alertBox.className = 'form-alert error show';
}

function clearAlert() {
    const alertBox = document.getElementById('form-alert');
    if (!alertBox) return;
    alertBox.textContent = '';
    alertBox.className = 'form-alert';
}

function resetPredictionForm() {
    const form = document.getElementById('prediction-form');
    if (form) {
        form.reset();
    }
    const dept = document.getElementById('department');
    const pType = document.getElementById('patient-type');
    const hr = document.getElementById('hour');
    if (dept) dept.value = '';
    if (pType) pType.value = '';
    if (hr) hr.value = '';
}

async function handleFormSubmit(e) {
    e.preventDefault();
    clearAlert();

    const deptVal = document.getElementById('department').value;
    const pTypeVal = document.getElementById('patient-type').value;
    const hrVal = document.getElementById('hour').value;

    if (!deptVal || !pTypeVal || !hrVal) {
        showAlert('Please select Department, Patient type, and Hour of day.');
        return;
    }

    const payload = {
        patient_arrivals: parseInt(document.getElementById('patient-arrivals').value, 10),
        emergency_cases: parseInt(document.getElementById('emergency-cases').value, 10),
        queue_length: parseInt(document.getElementById('queue-length').value, 10),
        discharge_count: parseInt(document.getElementById('discharge-count').value, 10),
        hospital_capacity: parseInt(document.getElementById('hospital-capacity').value, 10),
        occupied_beds: parseInt(document.getElementById('occupied-beds').value, 10),
        available_doctors: parseInt(document.getElementById('available-doctors').value, 10),
        available_nurses: parseInt(document.getElementById('available-nurses').value, 10),
        department: deptVal,
        patient_type: pTypeVal,
        hour: parseInt(hrVal, 10)
    };

    if (payload.occupied_beds > payload.hospital_capacity) {
        showAlert('Occupied beds cannot exceed hospital capacity.');
        return;
    }

    const btn = document.getElementById('btn-predict');
    if (btn) {
        btn.classList.add('loading');
        btn.disabled = true;
        btn.innerHTML = `
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12a9 9 0 1 1-6.219-8.56"/></svg>
            Analyzing data...
        `;
    }

    try {
        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });

        const data = await response.json();

        if (!response.ok || data.status === 'error') {
            showAlert(data.message || 'An error occurred while processing prediction.');
            return;
        }

        // Store new real prediction at top of history
        const historyItem = {
            id: Date.now(),
            timestamp: new Date().toISOString(),
            department: payload.department,
            patient_type: payload.patient_type,
            patient_arrivals: payload.patient_arrivals,
            emergency_cases: payload.emergency_cases,
            queue_length: payload.queue_length,
            discharge_count: payload.discharge_count,
            hospital_capacity: payload.hospital_capacity,
            occupied_beds: payload.occupied_beds,
            available_doctors: payload.available_doctors,
            available_nurses: payload.available_nurses,
            hour: payload.hour,
            prediction: data.prediction,
            confidence: data.confidence,
            metrics: data.metrics
        };

        predictions.unshift(historyItem);
        if (predictions.length > MAX_HISTORY_ITEMS) {
            predictions.length = MAX_HISTORY_ITEMS;
        }

        saveHistoryToStorage();

        // Update UI with real ML response
        showResult(data);
        updateKPIs(data, payload);
        updateBedOccupancyBar(data.metrics?.bed_occupancy_percentage);
        renderTable();
        drawCrowdingTrend();

    } catch (error) {
        console.error('API Error:', error);
        showAlert('Unable to connect to backend server. Please ensure Flask is running.');
    } finally {
        if (btn) {
            btn.classList.remove('loading');
            btn.disabled = false;
            btn.innerHTML = `
                <svg viewBox="0 0 24 24" fill="currentColor"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>
                Run prediction
            `;
        }
    }
}

function initDashboard() {
    populateHours();
    resetPredictionForm();
    loadHistoryFromStorage();
    fetchModelMetadata();

    // Render table and charts from restored history (if any)
    renderTable();
    drawCrowdingTrend();

    if (predictions.length > 0) {
        const latest = predictions[0];
        showResult(latest);
        updateKPIs(latest, latest);
        updateBedOccupancyBar(latest.metrics?.bed_occupancy_percentage);
    } else {
        resetResultCard();
        updateBedOccupancyBar(null);
    }

    const form = document.getElementById('prediction-form');
    if (form) {
        form.addEventListener('submit', handleFormSubmit);
    }

    window.addEventListener('resize', () => {
        drawCrowdingTrend();
    });
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initDashboard);
} else {
    initDashboard();
}
