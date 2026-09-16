/* ═══════════════════════════════════════════════════════════════
   Adaptive Federated IoT IDS — Dashboard Frontend Logic
   ═══════════════════════════════════════════════════════════════ */

document.addEventListener('DOMContentLoaded', () => {

  // ── Chart.js Global Defaults ──
  Chart.defaults.color = '#94a3b8';
  Chart.defaults.borderColor = 'rgba(255,255,255,0.05)';
  Chart.defaults.font.family = "'Inter', system-ui, sans-serif";

  const COLORS = {
    safe: '#10b981',
    safeDim: 'rgba(16, 185, 129, 0.15)',
    danger: '#ef4444',
    dangerDim: 'rgba(239, 68, 68, 0.15)',
    blue: '#3b82f6',
    blueDim: 'rgba(59, 130, 246, 0.15)',
    violet: '#8b5cf6',
    violetDim: 'rgba(139, 92, 246, 0.15)',
    cyan: '#22d3ee',
    cyanDim: 'rgba(34, 211, 238, 0.15)',
    amber: '#f59e0b',
    amberDim: 'rgba(245, 158, 11, 0.15)',
    rose: '#f43f5e',
    muted: '#64748b',
  };
  const STRATEGY_COLORS = {
    adaptive: COLORS.cyan,
    fedavg: COLORS.blue,
    weighted: COLORS.violet,
    fedprox: COLORS.amber,
  };
  const CLASS_COLORS = [COLORS.safe, COLORS.danger, COLORS.amber, COLORS.violet, COLORS.cyan, COLORS.rose, COLORS.blue];

  // ═══════════════════════════════════════
  // 1. NAVIGATION — Active Section Tracking
  // ═══════════════════════════════════════
  const navLinks = document.querySelectorAll('.nav-link[data-section]');
  const sections = {};
  navLinks.forEach(link => {
    const id = link.getAttribute('data-section');
    sections[id] = document.getElementById(id);
  });

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        navLinks.forEach(l => l.classList.remove('active'));
        const link = document.querySelector(`.nav-link[data-section="${entry.target.id}"]`);
        if (link) link.classList.add('active');
      }
    });
  }, { threshold: 0.3, rootMargin: '-80px 0px -40% 0px' });

  Object.values(sections).forEach(sec => { if (sec) observer.observe(sec); });

  // ═══════════════════════════════════════
  // 2. HERO — Load Model Info
  // ═══════════════════════════════════════
  fetch('/api/model-info')
    .then(r => r.json())
    .then(info => {
      document.getElementById('globalStatusDot').classList.add('online');
      document.getElementById('globalStatusLabel').textContent = 'System Online';
      document.getElementById('heroModelParams').textContent = info.param_count.toLocaleString();
      document.getElementById('heroExperiments').textContent = info.experiments.length;

      // Animate terminal lines
      const terminal = document.getElementById('heroTerminal');
      const statusLine = document.createElement('div');
      statusLine.className = 'terminal-line';
      statusLine.innerHTML = `<span class="t-label">Params:</span> ${info.param_count.toLocaleString()} trainable`;
      terminal.appendChild(statusLine);

      const scapyLine = document.createElement('div');
      scapyLine.className = 'terminal-line';
      scapyLine.innerHTML = `<span class="t-label">PCAP:</span> ${info.scapy_available ? '<span class="t-ok">● Scapy loaded</span>' : '<span style="color:#f59e0b">✗ Not available</span>'}`;
      terminal.appendChild(scapyLine);

      const readyLine = document.createElement('div');
      readyLine.className = 'terminal-line';
      readyLine.innerHTML = `<span class="t-ok">● Ready for classification</span>`;
      terminal.appendChild(readyLine);
    })
    .catch(() => {
      document.getElementById('globalStatusLabel').textContent = 'Offline';
    });

  // ═══════════════════════════════════════
  // 3. FILE UPLOAD & ANALYSIS
  // ═══════════════════════════════════════
  const uploadZone = document.getElementById('uploadZone');
  const uploadZoneInner = document.getElementById('uploadZoneInner');
  const fileInput = document.getElementById('fileInput');
  const uploadProgress = document.getElementById('uploadProgress');
  const uploadResults = document.getElementById('uploadResults');
  const uploadTableSection = document.getElementById('uploadTableSection');

  // Drag and Drop
  ['dragenter', 'dragover'].forEach(evt => {
    uploadZone.addEventListener(evt, (e) => {
      e.preventDefault();
      uploadZone.classList.add('dragover');
    });
  });
  ['dragleave', 'drop'].forEach(evt => {
    uploadZone.addEventListener(evt, (e) => {
      e.preventDefault();
      uploadZone.classList.remove('dragover');
    });
  });
  uploadZone.addEventListener('drop', (e) => {
    const files = e.dataTransfer.files;
    if (files.length > 0) handleFileUpload(files[0]);
  });
  fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) handleFileUpload(e.target.files[0]);
  });

  // Upload charts (initialized lazily)
  let uploadDistChart = null;
  let uploadConfChart = null;

  function handleFileUpload(file) {
    const validExts = ['.csv', '.pcap', '.pcapng'];
    const ext = '.' + file.name.split('.').pop().toLowerCase();
    if (!validExts.includes(ext)) {
      alert('Unsupported file type. Please upload a .csv or .pcap file.');
      return;
    }

    // Show progress
    uploadZoneInner.style.display = 'none';
    uploadProgress.style.display = 'flex';
    document.getElementById('uploadProgressText').textContent = `Analyzing ${file.name}...`;

    const formData = new FormData();
    formData.append('file', file);

    fetch('/api/upload', {
      method: 'POST',
      body: formData,
    })
    .then(r => r.json())
    .then(data => {
      uploadProgress.style.display = 'none';
      uploadZoneInner.style.display = 'flex';

      if (data.error) {
        alert('Error: ' + data.error);
        return;
      }

      displayUploadResults(data);
    })
    .catch(err => {
      uploadProgress.style.display = 'none';
      uploadZoneInner.style.display = 'flex';
      alert('Upload failed: ' + err.message);
    });
  }

  function displayUploadResults(data) {
    uploadResults.style.display = 'block';
    uploadTableSection.style.display = 'block';

    document.getElementById('resultsFilename').textContent = data.filename;
    document.getElementById('resultTotal').textContent = data.total_records.toLocaleString();
    document.getElementById('resultAlerts').textContent = data.total_alerts.toLocaleString();
    document.getElementById('resultAlertRate').textContent = data.alert_rate.toFixed(1) + '%';
    document.getElementById('resultAvgConf').textContent = data.avg_confidence.toFixed(1) + '%';
    document.getElementById('uploadRecordCount').textContent = `${data.results.length} records`;

    // Distribution donut chart
    const distCtx = document.getElementById('uploadDistChart').getContext('2d');
    if (uploadDistChart) uploadDistChart.destroy();
    const distLabels = Object.keys(data.class_distribution);
    const distValues = Object.values(data.class_distribution);
    uploadDistChart = new Chart(distCtx, {
      type: 'doughnut',
      data: {
        labels: distLabels,
        datasets: [{
          data: distValues,
          backgroundColor: distLabels.map((_, i) => CLASS_COLORS[i % CLASS_COLORS.length]),
          borderWidth: 0,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: 'right', labels: { font: { size: 10 }, padding: 8, usePointStyle: true, pointStyleWidth: 8 } },
        },
      },
    });

    // Confidence histogram
    const confCtx = document.getElementById('uploadConfChart').getContext('2d');
    if (uploadConfChart) uploadConfChart.destroy();
    const confBins = [0, 0, 0, 0, 0]; // 0-20, 20-40, 40-60, 60-80, 80-100
    data.results.forEach(r => {
      const pct = r.confidence * 100;
      const bin = Math.min(4, Math.floor(pct / 20));
      confBins[bin]++;
    });
    uploadConfChart = new Chart(confCtx, {
      type: 'bar',
      data: {
        labels: ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%'],
        datasets: [{
          label: 'Records',
          data: confBins,
          backgroundColor: [COLORS.danger, COLORS.amber, COLORS.blue, COLORS.cyan, COLORS.safe],
          borderRadius: 4,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          y: { beginAtZero: true, grid: { color: 'rgba(255,255,255,0.04)' } },
          x: { grid: { display: false } },
        },
        plugins: { legend: { display: false } },
      },
    });

    // Results table
    const tbody = document.getElementById('uploadTableBody');
    tbody.innerHTML = '';
    data.results.forEach(r => {
      const tr = document.createElement('tr');
      tr.className = r.is_normal ? 'row-normal' : 'row-alert';

      const verdict = r.is_normal
        ? `<span class="verdict-tag normal">✓ NORMAL</span>`
        : `<span class="verdict-tag alert">⚠ ${r.pred_label}</span>`;

      // Mini probability bars
      const probBars = (r.probabilities || []).map((p, i) =>
        `<div class="prob-bar" style="height:${Math.max(2, p * 22)}px; background:${CLASS_COLORS[i % CLASS_COLORS.length]}"></div>`
      ).join('');

      tr.innerHTML = `
        <td style="color:${COLORS.muted}">${r.seq}</td>
        <td style="font-family:var(--font-mono);font-size:0.75rem">${escapeHtml(r.summary)}</td>
        <td>${verdict}</td>
        <td>${(r.confidence * 100).toFixed(1)}%</td>
        <td><div class="prob-bar-group">${probBars}</div></td>
      `;
      tbody.appendChild(tr);
    });

    // Scroll to results
    uploadResults.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  // ═══════════════════════════════════════
  // 4. LIVE STREAM (SSE)
  // ═══════════════════════════════════════
  let isPlaying = true;
  let liveProcessed = 0;
  let liveAlerts = 0;
  let liveConfSum = 0;
  const liveClassCounts = {};

  // Live charts
  const liveConfCtx = document.getElementById('liveConfChart').getContext('2d');
  const liveConfChart = new Chart(liveConfCtx, {
    type: 'bar',
    data: {
      labels: ['Class 0', 'Class 1', 'Class 2'],
      datasets: [{
        label: 'Softmax Probability',
        data: [0.33, 0.33, 0.34],
        backgroundColor: CLASS_COLORS.slice(0, 3),
        borderRadius: 4,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      indexAxis: 'y',
      scales: {
        x: { min: 0, max: 1, grid: { color: 'rgba(255,255,255,0.04)' } },
        y: { grid: { display: false } },
      },
      plugins: { legend: { display: false } },
    },
  });

  const liveDistCtx = document.getElementById('liveDistChart').getContext('2d');
  const liveDistChart = new Chart(liveDistCtx, {
    type: 'doughnut',
    data: {
      labels: ['Normal', 'Threats'],
      datasets: [{
        data: [1, 0],
        backgroundColor: [COLORS.safe, COLORS.danger],
        borderWidth: 0,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: '65%',
      plugins: {
        legend: { position: 'right', labels: { font: { size: 10 }, padding: 8, usePointStyle: true, pointStyleWidth: 8 } },
      },
    },
  });

  // Connect SSE
  const evtSource = new EventSource('/api/stream');
  evtSource.onmessage = (e) => {
    try {
      const data = JSON.parse(e.data);
      handleLiveEvent(data);
    } catch (err) {
      console.error('SSE parse error:', err);
    }
  };
  evtSource.onerror = () => {
    console.warn('SSE connection interrupted, retrying...');
  };

  function handleLiveEvent(data) {
    liveProcessed++;
    const isNormal = data.is_normal;
    if (!isNormal) liveAlerts++;
    liveConfSum += data.confidence;

    const label = data.pred_label || 'Unknown';
    liveClassCounts[label] = (liveClassCounts[label] || 0) + 1;

    // Update threat badge
    const badge = document.getElementById('liveThreatBadge');
    const badgeText = document.getElementById('liveThreatText');
    if (isNormal) {
      badge.className = 'threat-badge normal';
      badgeText.textContent = 'NORMAL TRAFFIC';
    } else {
      badge.className = 'threat-badge alert';
      badgeText.textContent = `ALERT: ${label.toUpperCase()}`;
    }

    // Update stats
    document.getElementById('liveStatProcessed').textContent = liveProcessed.toLocaleString();
    document.getElementById('liveStatAlerts').textContent = liveAlerts.toLocaleString();
    const ratio = liveProcessed > 0 ? (liveAlerts / liveProcessed * 100) : 0;
    document.getElementById('liveStatAlertRatio').textContent = `${ratio.toFixed(1)}% alert rate`;
    const avgConf = liveProcessed > 0 ? (liveConfSum / liveProcessed * 100) : 0;
    document.getElementById('liveStatConf').textContent = `${avgConf.toFixed(1)}%`;
    document.getElementById('liveStreamCounter').textContent = `${liveProcessed} events`;

    // Table row
    const tbody = document.getElementById('liveTableBody');
    const tr = document.createElement('tr');
    tr.className = isNormal ? 'row-normal' : 'row-alert';
    const verdict = isNormal
      ? '<span class="verdict-tag normal">✓ NORMAL</span>'
      : `<span class="verdict-tag alert">⚠ ${escapeHtml(label)}</span>`;
    tr.innerHTML = `
      <td style="color:${COLORS.muted}">${data.seq}</td>
      <td style="color:${COLORS.muted}">${data.timestamp}</td>
      <td style="font-family:var(--font-mono);font-size:0.75rem">${escapeHtml(data.summary)}</td>
      <td>${verdict}</td>
      <td>${(data.confidence * 100).toFixed(1)}%</td>
    `;
    tbody.insertBefore(tr, tbody.firstChild);
    if (tbody.children.length > 50) tbody.removeChild(tbody.lastChild);

    // Update charts
    if (data.probabilities) {
      liveConfChart.data.labels = data.probabilities.map((_, i) => `Class ${i}`);
      liveConfChart.data.datasets[0].data = data.probabilities;
      liveConfChart.data.datasets[0].backgroundColor = data.probabilities.map((_, i) => CLASS_COLORS[i % CLASS_COLORS.length]);
      liveConfChart.update('none');
    }
    liveDistChart.data.labels = Object.keys(liveClassCounts);
    liveDistChart.data.datasets[0].data = Object.values(liveClassCounts);
    liveDistChart.data.datasets[0].backgroundColor = Object.keys(liveClassCounts).map((_, i) => CLASS_COLORS[i % CLASS_COLORS.length]);
    liveDistChart.update('none');
  }

  // Controls
  const btnPlayPause = document.getElementById('btnPlayPause');
  const speedSlider = document.getElementById('speedSlider');
  const delayVal = document.getElementById('delayVal');

  btnPlayPause.addEventListener('click', () => {
    isPlaying = !isPlaying;
    btnPlayPause.innerHTML = isPlaying
      ? '<svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg> Pause Stream'
      : '<svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"/></svg> Resume Stream';
    btnPlayPause.className = isPlaying ? 'btn btn-primary' : 'btn btn-outline';
    postControl({ is_playing: isPlaying });
  });

  speedSlider.addEventListener('input', (e) => {
    const val = parseFloat(e.target.value);
    delayVal.textContent = `${val.toFixed(1)}s`;
    postControl({ delay: val });
  });

  function postControl(payload) {
    fetch('/api/control', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }).catch(err => console.error('Control error:', err));
  }

  // ═══════════════════════════════════════
  // 5. TRAINING INSIGHTS
  // ═══════════════════════════════════════
  fetch('/api/training-results')
    .then(r => r.json())
    .then(allData => {
      renderTrainingCharts(allData);
      renderComparisonSection(allData);
    })
    .catch(err => console.error('Training data error:', err));

  function renderTrainingCharts(allData) {
    // Find the CIL demo or first available experiment
    const primaryExp = allData['cil_demo'] || Object.values(allData)[0];
    if (!primaryExp || !primaryExp.rounds) return;

    const rounds = primaryExp.rounds;
    const roundLabels = rounds.map(r => `R${r.round + 1}`);

    // Accuracy chart
    new Chart(document.getElementById('trainAccChart').getContext('2d'), {
      type: 'line',
      data: {
        labels: roundLabels,
        datasets: [{
          label: 'Global Accuracy',
          data: rounds.map(r => r.global_accuracy),
          borderColor: COLORS.blue,
          backgroundColor: COLORS.blueDim,
          fill: true,
          tension: 0.4,
          pointRadius: 5,
          pointBackgroundColor: COLORS.blue,
        }, {
          label: 'Prototype Accuracy',
          data: rounds.map(r => r.prototype_accuracy),
          borderColor: COLORS.violet,
          backgroundColor: COLORS.violetDim,
          fill: true,
          tension: 0.4,
          pointRadius: 5,
          pointBackgroundColor: COLORS.violet,
        }],
      },
      options: chartLineOpts('Accuracy'),
    });

    // F1 chart
    new Chart(document.getElementById('trainF1Chart').getContext('2d'), {
      type: 'line',
      data: {
        labels: roundLabels,
        datasets: [{
          label: 'Macro F1',
          data: rounds.map(r => r.f1_macro),
          borderColor: COLORS.cyan,
          backgroundColor: COLORS.cyanDim,
          fill: true,
          tension: 0.4,
          pointRadius: 5,
          pointBackgroundColor: COLORS.cyan,
        }, {
          label: 'Macro Precision',
          data: rounds.map(r => r.precision_macro),
          borderColor: COLORS.safe,
          borderDash: [5, 3],
          tension: 0.4,
          pointRadius: 3,
        }, {
          label: 'Macro Recall',
          data: rounds.map(r => r.recall_macro),
          borderColor: COLORS.amber,
          borderDash: [5, 3],
          tension: 0.4,
          pointRadius: 3,
        }],
      },
      options: chartLineOpts('Score'),
    });

    // Aggregation weights chart
    if (rounds[0].aggregation_weights) {
      const nClients = rounds[0].aggregation_weights.length;
      const datasets = [];
      for (let c = 0; c < nClients; c++) {
        datasets.push({
          label: `Client ${c}`,
          data: rounds.map(r => r.aggregation_weights[c]),
          backgroundColor: CLASS_COLORS[c % CLASS_COLORS.length] + '80',
          borderColor: CLASS_COLORS[c % CLASS_COLORS.length],
          borderWidth: 1,
        });
      }
      new Chart(document.getElementById('trainWeightsChart').getContext('2d'), {
        type: 'bar',
        data: { labels: roundLabels, datasets },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            x: { stacked: true, grid: { display: false } },
            y: { stacked: true, min: 0, max: 1, grid: { color: 'rgba(255,255,255,0.04)' }, title: { display: true, text: 'Weight', color: '#64748b' } },
          },
          plugins: { legend: { labels: { font: { size: 10 }, usePointStyle: true, pointStyleWidth: 8 } } },
        },
      });
    }

    // CIL stage metrics
    const stageRounds = rounds.filter(r => r.stage_metrics);
    if (stageRounds.length > 0) {
      const cilLabels = stageRounds.map((r, i) => `Stage ${i + 1} (classes: ${(r.class_stage || []).join(',')})`);
      new Chart(document.getElementById('trainCILChart').getContext('2d'), {
        type: 'bar',
        data: {
          labels: cilLabels,
          datasets: [{
            label: 'New Class Accuracy',
            data: stageRounds.map(r => r.stage_metrics.new_accuracy),
            backgroundColor: COLORS.safe + 'cc',
            borderRadius: 4,
          }, {
            label: 'Old Class Accuracy',
            data: stageRounds.map(r => r.stage_metrics.old_accuracy),
            backgroundColor: COLORS.amber + 'cc',
            borderRadius: 4,
          }, {
            label: 'Forgetting',
            data: stageRounds.map(r => r.stage_metrics.forgetting),
            backgroundColor: COLORS.danger + 'cc',
            borderRadius: 4,
          }],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            y: { beginAtZero: true, grid: { color: 'rgba(255,255,255,0.04)' } },
            x: { grid: { display: false } },
          },
          plugins: { legend: { labels: { font: { size: 10 }, usePointStyle: true, pointStyleWidth: 8 } } },
        },
      });
    }

    // Communication chart
    new Chart(document.getElementById('trainCommChart').getContext('2d'), {
      type: 'bar',
      data: {
        labels: roundLabels,
        datasets: [{
          label: 'Communication Bytes',
          data: rounds.map(r => r.communication_bytes),
          backgroundColor: COLORS.blue + '80',
          borderColor: COLORS.blue,
          borderWidth: 1,
          borderRadius: 4,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          y: { beginAtZero: true, grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { callback: v => (v / 1000).toFixed(0) + 'KB' } },
          x: { grid: { display: false } },
        },
        plugins: { legend: { display: false } },
      },
    });

    // Per-class accuracy heatmap (bar per class per round)
    const allClassIds = new Set();
    rounds.forEach(r => { if (r.class_accuracy) Object.keys(r.class_accuracy).forEach(k => allClassIds.add(k)); });
    const sortedClasses = [...allClassIds].sort((a, b) => Number(a) - Number(b));
    if (sortedClasses.length > 0) {
      const classDatasets = sortedClasses.map((cls, i) => ({
        label: `Class ${cls}`,
        data: rounds.map(r => r.class_accuracy ? (r.class_accuracy[cls] ?? null) : null),
        backgroundColor: CLASS_COLORS[i % CLASS_COLORS.length] + 'cc',
        borderRadius: 3,
      }));
      new Chart(document.getElementById('trainClassAccChart').getContext('2d'), {
        type: 'bar',
        data: { labels: roundLabels, datasets: classDatasets },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            y: { min: 0, max: 1, grid: { color: 'rgba(255,255,255,0.04)' }, title: { display: true, text: 'Accuracy', color: '#64748b' } },
            x: { grid: { display: false } },
          },
          plugins: { legend: { labels: { font: { size: 10 }, usePointStyle: true, pointStyleWidth: 8 } } },
        },
      });
    }
  }

  // ═══════════════════════════════════════
  // 6. COMPARISON SECTION
  // ═══════════════════════════════════════
  function renderComparisonSection(allData) {
    // Find matching experiment groups for comparison
    const strategies = ['adaptive', 'fedavg', 'weighted', 'fedprox'];
    const groups = {};

    // Try TON_IoT_small_* group first, then comparison_seed42_*, then anything
    const prefixes = ['TON_IoT_small_', 'comparison_seed42_', 'synthetic10_', 'synthetic_'];
    let selectedPrefix = null;

    for (const prefix of prefixes) {
      const found = strategies.filter(s => allData[prefix + s]);
      if (found.length >= 2) {
        selectedPrefix = prefix;
        found.forEach(s => { groups[s] = allData[prefix + s]; });
        break;
      }
    }

    // Fallback: try to match any experiment name containing strategy name
    if (!selectedPrefix) {
      for (const [name, data] of Object.entries(allData)) {
        for (const s of strategies) {
          if (name.toLowerCase().includes(s) && !groups[s]) {
            groups[s] = data;
          }
        }
      }
    }

    // Populate comparison table
    const tbody = document.getElementById('comparisonTableBody');
    tbody.innerHTML = '';

    if (Object.keys(groups).length === 0) {
      tbody.innerHTML = '<tr><td colspan="5" class="loading-cell">No comparison experiments found. Run experiments with different aggregator strategies.</td></tr>';
      return;
    }

    // Find best values for highlighting
    let bestAcc = -1, bestF1 = -1;
    for (const [s, data] of Object.entries(groups)) {
      if (!data.rounds || data.rounds.length === 0) continue;
      const last = data.rounds[data.rounds.length - 1];
      if (last.global_accuracy > bestAcc) bestAcc = last.global_accuracy;
      if (last.f1_macro > bestF1) bestF1 = last.f1_macro;
    }

    for (const s of strategies) {
      if (!groups[s] || !groups[s].rounds || groups[s].rounds.length === 0) continue;
      const rounds = groups[s].rounds;
      const last = rounds[rounds.length - 1];
      const tr = document.createElement('tr');

      const isAccBest = last.global_accuracy === bestAcc;
      const isF1Best = last.f1_macro === bestF1;

      tr.innerHTML = `
        <td class="strategy-name" style="color:${STRATEGY_COLORS[s] || COLORS.muted}">${s.charAt(0).toUpperCase() + s.slice(1)}</td>
        <td class="${isAccBest ? 'highlight-value' : ''}">${last.global_accuracy.toFixed(4)}</td>
        <td class="${isF1Best ? 'highlight-value' : ''}">${last.f1_macro ? last.f1_macro.toFixed(4) : '—'}</td>
        <td>${(last.communication_bytes / 1024).toFixed(1)} KB</td>
        <td>${rounds.length}</td>
      `;
      tbody.appendChild(tr);
    }

    // Comparison accuracy chart
    const compAccDatasets = [];
    const compF1Datasets = [];
    let maxRounds = 0;

    for (const s of strategies) {
      if (!groups[s] || !groups[s].rounds) continue;
      const rounds = groups[s].rounds;
      maxRounds = Math.max(maxRounds, rounds.length);
      compAccDatasets.push({
        label: s.charAt(0).toUpperCase() + s.slice(1),
        data: rounds.map(r => r.global_accuracy),
        borderColor: STRATEGY_COLORS[s] || COLORS.muted,
        backgroundColor: (STRATEGY_COLORS[s] || COLORS.muted) + '20',
        fill: false,
        tension: 0.4,
        pointRadius: 4,
        pointBackgroundColor: STRATEGY_COLORS[s] || COLORS.muted,
        borderWidth: 2,
      });
      compF1Datasets.push({
        label: s.charAt(0).toUpperCase() + s.slice(1),
        data: rounds.map(r => r.f1_macro),
        borderColor: STRATEGY_COLORS[s] || COLORS.muted,
        fill: false,
        tension: 0.4,
        pointRadius: 4,
        pointBackgroundColor: STRATEGY_COLORS[s] || COLORS.muted,
        borderWidth: 2,
      });
    }

    const compLabels = Array.from({ length: maxRounds }, (_, i) => `Round ${i + 1}`);

    new Chart(document.getElementById('compAccChart').getContext('2d'), {
      type: 'line',
      data: { labels: compLabels, datasets: compAccDatasets },
      options: chartLineOpts('Accuracy'),
    });

    new Chart(document.getElementById('compF1Chart').getContext('2d'), {
      type: 'line',
      data: { labels: compLabels, datasets: compF1Datasets },
      options: chartLineOpts('F1 Score'),
    });
  }

  // ═══════════════════════════════════════
  // HELPERS
  // ═══════════════════════════════════════
  function chartLineOpts(yLabel) {
    return {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      scales: {
        y: { beginAtZero: true, grid: { color: 'rgba(255,255,255,0.04)' }, title: { display: true, text: yLabel, color: '#64748b', font: { size: 11 } } },
        x: { grid: { display: false } },
      },
      plugins: {
        legend: { labels: { font: { size: 10 }, usePointStyle: true, pointStyleWidth: 8 } },
        tooltip: { backgroundColor: 'rgba(15, 23, 42, 0.95)', titleFont: { size: 11 }, bodyFont: { size: 10 }, padding: 10, cornerRadius: 8, borderColor: 'rgba(99, 115, 155, 0.3)', borderWidth: 1 },
      },
    };
  }

  function escapeHtml(str) {
    return String(str).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  }

});
