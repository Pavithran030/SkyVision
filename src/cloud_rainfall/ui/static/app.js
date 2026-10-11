/**
 * SkyVision Client Dashboard Application
 * Handles file uploads, sample selection, FastAPI async requests, and Chart.js rendering.
 */

let currentMode = 'sample'; // 'sample' or 'upload'
let selectedFile = null;
let selectedSamplePath = null;
let cloudChart = null;

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
  fetchSystemStatus();
  fetchSampleList();
  setupDropzone();
  updateWeatherValues();
});

// Tab Switcher
function switchTab(tabId) {
  document.querySelectorAll('.tab-content').forEach(el => el.classList.add('hidden'));
  document.querySelectorAll('.nav-btn').forEach(btn => {
    btn.classList.remove('bg-blue-600', 'text-white');
    btn.classList.add('text-slate-400');
  });

  const activeTab = document.getElementById(`tab-${tabId}`);
  const activeBtn = document.getElementById(`nav-${tabId}`);
  if (activeTab) activeTab.classList.remove('hidden');
  if (activeBtn) {
    activeBtn.classList.remove('text-slate-400');
    activeBtn.classList.add('bg-blue-600', 'text-white');
  }
}

// Fetch system health and model metadata
async function fetchSystemStatus() {
  try {
    const res = await fetch('/api/status');
    const data = await res.json();

    const cloudBadge = document.getElementById('cloud-status-badge');
    if (data.cloud_model_loaded) {
      cloudBadge.className = 'px-2.5 py-1 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 flex items-center gap-1.5 font-medium';
      cloudBadge.innerHTML = '<span class="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span> CNN: Active';
    } else {
      cloudBadge.className = 'px-2.5 py-1 rounded-full bg-red-500/10 text-red-400 border border-red-500/20 flex items-center gap-1.5 font-medium';
      cloudBadge.innerHTML = '<span class="w-2 h-2 rounded-full bg-red-500"></span> CNN: Offline';
    }

    const rainBadge = document.getElementById('rain-status-badge');
    if (data.rainfall_model_loaded) {
      rainBadge.className = 'px-2.5 py-1 rounded-full bg-blue-500/10 text-blue-400 border border-blue-500/20 flex items-center gap-1.5 font-medium';
      rainBadge.innerHTML = '<span class="w-2 h-2 rounded-full bg-blue-500"></span> Nowcasting: Active';
    } else {
      rainBadge.className = 'px-2.5 py-1 rounded-full bg-amber-500/10 text-amber-400 border border-amber-500/20 flex items-center gap-1.5 font-medium';
      rainBadge.innerHTML = '<span class="w-2 h-2 rounded-full bg-amber-500"></span> Nowcasting: Unavailable';
    }
  } catch (err) {
    console.error('Failed to load system status:', err);
  }
}

// Fetch Demo Samples
async function fetchSampleList() {
  try {
    const res = await fetch('/api/samples');
    const data = await res.json();
    const select = document.getElementById('sample-select');
    select.innerHTML = '';

    if (!data.samples || data.samples.length === 0) {
      select.innerHTML = '<option value="">No local samples found</option>';
      return;
    }

    data.samples.forEach((s, idx) => {
      const opt = document.createElement('option');
      opt.value = s.relative_path;
      opt.textContent = `${s.class_name.replace('_', ' ').toUpperCase()} — ${s.filename}`;
      opt.dataset.url = s.url;
      select.appendChild(opt);
    });

    // Auto-select first sample
    if (data.samples.length > 0) {
      select.selectedIndex = 0;
      handleSampleSelect();
    }
  } catch (err) {
    console.error('Failed to load sample list:', err);
  }
}

// Input Mode Toggle
function setInputMode(mode) {
  currentMode = mode;
  const btnSample = document.getElementById('btn-mode-sample');
  const btnUpload = document.getElementById('btn-mode-upload');
  const sampleContainer = document.getElementById('sample-mode-container');
  const uploadContainer = document.getElementById('upload-mode-container');

  if (mode === 'sample') {
    btnSample.className = 'py-2 text-xs font-semibold rounded-lg bg-blue-600 text-white transition';
    btnUpload.className = 'py-2 text-xs font-semibold rounded-lg bg-slate-700/60 text-slate-300 hover:text-white transition';
    sampleContainer.classList.remove('hidden');
    uploadContainer.classList.add('hidden');
    handleSampleSelect();
  } else {
    btnUpload.className = 'py-2 text-xs font-semibold rounded-lg bg-blue-600 text-white transition';
    btnSample.className = 'py-2 text-xs font-semibold rounded-lg bg-slate-700/60 text-slate-300 hover:text-white transition';
    uploadContainer.classList.remove('hidden');
    sampleContainer.classList.add('hidden');
    if (selectedFile) {
      showPreview(URL.createObjectURL(selectedFile), selectedFile.name);
    }
  }
}

// Handle Sample Selection
function handleSampleSelect() {
  const select = document.getElementById('sample-select');
  const selectedOpt = select.options[select.selectedIndex];
  if (selectedOpt && selectedOpt.value) {
    selectedSamplePath = selectedOpt.value;
    const url = selectedOpt.dataset.url;
    showPreview(url, selectedOpt.textContent);
  }
}

// Handle File Upload
function handleFileUpload(event) {
  const file = event.target.files[0];
  if (file) {
    selectedFile = file;
    showPreview(URL.createObjectURL(file), file.name);
  }
}

function showPreview(src, label) {
  const container = document.getElementById('preview-container');
  const img = document.getElementById('image-preview');
  const meta = document.getElementById('preview-meta');

  img.src = src;
  meta.textContent = label;
  container.classList.remove('hidden');

  // Hide old warnings on new image
  document.getElementById('validation-alert').classList.add('hidden');
}

// Setup Drag & Drop
function setupDropzone() {
  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('file-input');

  dropzone.addEventListener('click', () => fileInput.click());

  ['dragenter', 'dragover'].forEach(name => {
    dropzone.addEventListener(name, (e) => {
      e.preventDefault();
      dropzone.classList.add('border-blue-500', 'bg-blue-500/10');
    });
  });

  ['dragleave', 'drop'].forEach(name => {
    dropzone.addEventListener(name, (e) => {
      e.preventDefault();
      dropzone.classList.remove('border-blue-500', 'bg-blue-500/10');
    });
  });

  dropzone.addEventListener('drop', (e) => {
    const file = e.dataTransfer.files[0];
    if (file) {
      selectedFile = file;
      showPreview(URL.createObjectURL(file), file.name);
    }
  });
}

// Toggle Weather Accordion
function toggleWeatherAccordion() {
  const container = document.getElementById('weather-sliders-container');
  const icon = document.getElementById('weather-toggle-icon');
  container.classList.toggle('hidden');
  icon.classList.toggle('rotate-180');
}

// Update Weather Values Display
function updateWeatherValues() {
  const temp = document.getElementById('input-temp').value;
  const humidity = document.getElementById('input-humidity').value;
  const pressure = document.getElementById('input-pressure').value;
  const wind = document.getElementById('input-wind').value;

  document.getElementById('temp-val').textContent = `${parseFloat(temp).toFixed(1)} °C`;
  document.getElementById('humidity-val').textContent = `${humidity} %`;
  document.getElementById('pressure-val').textContent = `${pressure} hPa`;
  document.getElementById('wind-val').textContent = `${parseFloat(wind).toFixed(1)} km/h`;
}

// Run Full Vision & Nowcasting Prediction
async function runAnalysis() {
  const btn = document.getElementById('btn-analyze');
  const spinner = document.getElementById('btn-spinner');

  const formData = new FormData();
  if (currentMode === 'upload') {
    if (!selectedFile) {
      alert('Please upload an image file first.');
      return;
    }
    formData.append('file', selectedFile);
  } else {
    if (!selectedSamplePath) {
      alert('Please select a sample image first.');
      return;
    }
    formData.append('sample_relative_path', selectedSamplePath);
  }

  // Append weather parameters
  formData.append('temperature_2m', document.getElementById('input-temp').value);
  formData.append('relative_humidity_2m', document.getElementById('input-humidity').value);
  formData.append('surface_pressure', document.getElementById('input-pressure').value);
  formData.append('wind_speed_10m', document.getElementById('input-wind').value);

  btn.disabled = true;
  spinner.classList.remove('hidden');

  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      body: formData,
    });
    const data = await res.json();

    if (!data.success) {
      alert(`Inference failed: ${data.error}`);
      return;
    }

    renderResults(data);
  } catch (err) {
    console.error('Analysis request error:', err);
    alert(`Server error during prediction: ${err.message}`);
  } finally {
    btn.disabled = false;
    spinner.classList.add('hidden');
  }
}

// Render Results to UI
function renderResults(data) {
  const emptyState = document.getElementById('results-empty-state');
  const activeContainer = document.getElementById('results-active-container');
  emptyState.classList.add('hidden');
  activeContainer.classList.remove('hidden');

  // 1. Validation Warnings
  const valAlert = document.getElementById('validation-alert');
  const valList = document.getElementById('validation-warnings-list');
  valList.innerHTML = '';
  if (data.validation && data.validation.warnings && data.validation.warnings.length > 0) {
    data.validation.warnings.forEach(w => {
      const li = document.createElement('li');
      li.textContent = w;
      valList.appendChild(li);
    });
    valAlert.classList.remove('hidden');
  } else {
    valAlert.classList.add('hidden');
  }

  // 2. Task A: Cloud Pattern Classification
  const cloud = data.cloud_result;
  if (cloud) {
    const formattedName = cloud.predicted_class.replace(/_/g, ' ');
    document.getElementById('cloud-pred-name').textContent = formattedName;
    document.getElementById('cloud-confidence-badge').textContent = `${(cloud.confidence * 100).toFixed(1)}% Confidence`;
    document.getElementById('cloud-explanation').textContent = cloud.explanation;

    renderChart(cloud.probabilities);
  }

  // 3. Task B: Rainfall Nowcasting
  const rain = data.rainfall_result;
  if (rain && rain.is_available) {
    document.getElementById('rain-pred-category').textContent = rain.predicted_category.replace(/_/g, ' ');
    document.getElementById('rain-pred-amount').textContent = `${rain.predicted_mm.toFixed(2)} mm`;
    document.getElementById('rain-pred-horizon').textContent = rain.horizon.replace(/_/g, ' ');
    document.getElementById('rain-model-badge').textContent = rain.model_type;
    document.getElementById('rain-label-source').textContent = `Label source: ${rain.label_source}`;

    const mmBadge = document.getElementById('multimodal-pred-badge');
    if (rain.multimodal_category) {
      mmBadge.textContent = rain.multimodal_category.replace(/_/g, ' ').toUpperCase();
      document.getElementById('multimodal-comparison-box').classList.remove('hidden');
    } else {
      document.getElementById('multimodal-comparison-box').classList.add('hidden');
    }
  }
}

// Render Chart.js Horizontal Bar Chart
function renderChart(probabilities) {
  const ctx = document.getElementById('cloud-chart').getContext('2d');
  const labels = Object.keys(probabilities).map(k => k.replace(/_/g, ' ').toUpperCase());
  const values = Object.values(probabilities).map(v => (v * 100).toFixed(1));

  // Find max for highlighting
  const maxVal = Math.max(...values.map(Number));
  const bgColors = values.map(v => Number(v) === maxVal ? 'rgba(59, 130, 246, 0.9)' : 'rgba(71, 85, 105, 0.7)');
  const borderColors = values.map(v => Number(v) === maxVal ? '#60a5fa' : '#64748b');

  if (cloudChart) {
    cloudChart.destroy();
  }

  cloudChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{
        label: 'Confidence (%)',
        data: values,
        backgroundColor: bgColors,
        borderColor: borderColors,
        borderWidth: 1.5,
        borderRadius: 6,
      }]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (item) => ` ${item.raw}% Probability`
          }
        }
      },
      scales: {
        x: {
          min: 0,
          max: 100,
          grid: { color: 'rgba(51, 65, 85, 0.4)' },
          ticks: { color: '#94a3b8', font: { size: 10 }, callback: v => `${v}%` }
        },
        y: {
          grid: { display: false },
          ticks: { color: '#e2e8f0', font: { size: 11, weight: 'bold' } }
        }
      }
    }
  });
}
