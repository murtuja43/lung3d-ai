// ═══════════════════════════════════════════
// MAIN.JS — Multimodal TB AI Frontend Logic
// ═══════════════════════════════════════════

document.addEventListener('DOMContentLoaded', () => {

  // ─────────────────────────────────────────
  // Element references
  // ─────────────────────────────────────────
  const analyzeBtn       = document.getElementById('analyze-btn');
  const loadingEl        = document.getElementById('loading');
  const predictionResult = document.getElementById('prediction-result');
  const confidenceSection= document.getElementById('confidence-section');
  const confidenceBar    = document.getElementById('confidence-bar');
  const confidenceLabel  = document.getElementById('confidence-label');
  const explanationText  = document.getElementById('explanation-text');
  const downloadBtn      = document.getElementById('download-btn');
  const uploadZone       = document.getElementById('upload-zone');
  const xrayInput        = document.getElementById('xray-input');
  const uploadPreview    = document.getElementById('upload-preview');
  const previewImg       = document.getElementById('preview-img');
  const previewName      = document.getElementById('preview-name');
  const removeImgBtn     = document.getElementById('remove-img');
  const originalBox      = document.getElementById('original-box');
  const heatmapBox       = document.getElementById('heatmap-box');
  const statsSection     = document.getElementById('stats-section');

  let selectedFile = null;

  // ─────────────────────────────────────────
  // Upload zone — click to browse
  // ─────────────────────────────────────────
  uploadZone.addEventListener('click', () => {
    xrayInput.click();
  });

  xrayInput.addEventListener('change', (e) => {
    if (e.target.files[0]) {
      handleFileSelect(e.target.files[0]);
    }
  });

  // ─────────────────────────────────────────
  // Drag and drop support
  // ─────────────────────────────────────────
  uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadZone.classList.add('dragover');
  });

  uploadZone.addEventListener('dragleave', () => {
    uploadZone.classList.remove('dragover');
  });

  uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadZone.classList.remove('dragover');
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
      handleFileSelect(file);
    }
  });

  // ─────────────────────────────────────────
  // Handle file selection
  // ─────────────────────────────────────────
  function handleFileSelect(file) {
    selectedFile = file;

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
      previewImg.src     = e.target.result;
      previewName.textContent = file.name;
      uploadZone.classList.add('hidden');
      uploadPreview.classList.remove('hidden');
    };
    reader.readAsDataURL(file);
  }

  // ─────────────────────────────────────────
  // Remove selected image
  // ─────────────────────────────────────────
  removeImgBtn.addEventListener('click', () => {
    selectedFile        = null;
    xrayInput.value     = '';
    previewImg.src      = '';
    uploadPreview.classList.add('hidden');
    uploadZone.classList.remove('hidden');
  });

  // ─────────────────────────────────────────
  // Show / hide loading
  // ─────────────────────────────────────────
  function setLoading(isLoading) {
    if (isLoading) {
      loadingEl.classList.remove('hidden');
      analyzeBtn.disabled     = true;
      analyzeBtn.textContent  = '⏳ Analyzing...';
    } else {
      loadingEl.classList.add('hidden');
      analyzeBtn.disabled     = false;
      analyzeBtn.textContent  = '🔍 Analyze Patient';
    }
  }

  // ─────────────────────────────────────────
  // Collect patient form data
  // ─────────────────────────────────────────
  function getPatientFormData() {
    return {
      age:          document.getElementById('age').value,
      sex:          document.getElementById('sex').value,
      bmi:          document.getElementById('bmi').value,
      cough_weeks:  document.getElementById('cough_weeks').value,
      fever:        document.getElementById('fever').checked,
      night_sweats: document.getElementById('night_sweats').checked,
      weight_loss:  document.getElementById('weight_loss').checked,
      fatigue:      document.getElementById('fatigue').checked,
      chest_pain:   document.getElementById('chest_pain').checked,
      tb_contact:   document.getElementById('tb_contact').checked,
      prev_tb:      document.getElementById('prev_tb').checked,
    };
  }

  // ─────────────────────────────────────────
  // Render prediction badge
  // ─────────────────────────────────────────
  function renderPrediction(data) {
    const isTB = data.prediction === 'TB Detected';

    predictionResult.className = isTB
      ? 'prediction-tb'
      : 'prediction-no-tb';

    predictionResult.innerHTML = isTB
      ? `⚠️ ${data.prediction}`
      : `✅ ${data.prediction}`;

    // Confidence bar
    confidenceSection.classList.remove('hidden');
    const pct = Math.round(data.confidence * 100);

    setTimeout(() => {
      confidenceBar.style.width = `${pct}%`;
      if (pct >= 60) {
        confidenceBar.style.background =
          'linear-gradient(90deg, #ff4757, #ff6b81)';
      } else if (pct >= 40) {
        confidenceBar.style.background =
          'linear-gradient(90deg, #ffa502, #ffcd39)';
      } else {
        confidenceBar.style.background =
          'linear-gradient(90deg, #00ff88, #00d4ff)';
      }
      confidenceLabel.textContent = `${pct}%`;
    }, 100);

    // Show download button
    downloadBtn.classList.remove('hidden');
  }

  // ─────────────────────────────────────────
  // Render stats cards
  // ─────────────────────────────────────────
  function renderStats(data) {
    const stats = [
      {
        label: 'CNN Image Prob',
        value: `${Math.round(data.cnn_probability * 100)}%`
      },
      {
        label: 'Clinical Score',
        value: `${Math.round(data.clinical_score * 100)}%`
      },
    ];

    statsSection.innerHTML = `
      <div style="
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 8px;
        margin-top: 12px;
      ">
        ${stats.map(s => `
          <div style="
            background: #0a0e1a;
            border: 1px solid #1e2d45;
            border-radius: 8px;
            padding: 8px 10px;
          ">
            <div style="color:#445566;font-size:10px;
                        text-transform:uppercase">
              ${s.label}
            </div>
            <div style="color:#00d4ff;font-size:16px;
                        font-weight:700;margin-top:2px">
              ${s.value}
            </div>
          </div>
        `).join('')}
      </div>
    `;
  }

  // ─────────────────────────────────────────
  // Render explanation report
  // ─────────────────────────────────────────
  function renderExplanation(data) {
    explanationText.classList.remove('explanation-placeholder');
    const lines   = data.explanation.split('\n');
    const colored = lines.map(line => {
      if (line.includes('✅')) {
        return `<span style="color:#00ff88">${escHtml(line)}</span>`;
      } else if (line.includes('⚠️')) {
        return `<span style="color:#ffa502">${escHtml(line)}</span>`;
      } else if (line.includes('🔬')) {
        return `<span style="color:#00d4ff;font-weight:bold">
                  ${escHtml(line)}</span>`;
      } else if (line.includes('📊') || line.includes('📋')
              || line.includes('👤')) {
        return `<span style="color:#8899aa">${escHtml(line)}</span>`;
      } else {
        return `<span style="color:#aabbcc">${escHtml(line)}</span>`;
      }
    });
    explanationText.innerHTML = colored.join('\n');
  }

  // ─────────────────────────────────────────
  // Render heatmap images
  // ─────────────────────────────────────────
  function renderHeatmaps(data) {
    if (data.original_b64) {
      originalBox.innerHTML = `
        <img src="data:image/png;base64,${data.original_b64}"
             alt="Original X-Ray"
             style="width:100%;border-radius:8px"/>
      `;
    }

    if (data.heatmap_b64) {
      heatmapBox.innerHTML = `
        <img src="data:image/png;base64,${data.heatmap_b64}"
             alt="Heatmap"
             style="width:100%;border-radius:8px"/>
      `;
    }
  }

  // ─────────────────────────────────────────
  // Helper: escape HTML
  // ─────────────────────────────────────────
  function escHtml(str) {
    return str
      .replace(/&/g,  '&amp;')
      .replace(/</g,  '&lt;')
      .replace(/>/g,  '&gt;')
      .replace(/"/g,  '&quot;')
      .replace(/'/g,  '&#039;');
  }

  // ─────────────────────────────────────────
  // Main analyze function
  // ─────────────────────────────────────────
  async function runAnalysis() {

    // Validate file
    if (!selectedFile) {
      alert('⚠️ Please upload a chest X-ray image first!');
      return;
    }

    setLoading(true);

    try {
      // Build FormData with image + patient data
      const formData   = new FormData();
      const patientData = getPatientFormData();

      formData.append('xray', selectedFile);
      Object.entries(patientData).forEach(([key, val]) => {
        formData.append(key, val.toString());
      });

      // Call the analyze API
      const response = await fetch('/api/analyze', {
        method: 'POST',
        body:   formData,
      });

      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.error || 'Analysis failed');
      }

      const data = await response.json();

      // Render all results
      renderPrediction(data);
      renderStats(data);
      renderExplanation(data);
      renderHeatmaps(data);

      // Load 3D viewer
      await Viewer3D.loadVolume(42);

    } catch (err) {
      console.error('Analysis error:', err);
      predictionResult.className   = '';
      predictionResult.textContent =
        `❌ Error: ${err.message}`;
    } finally {
      setLoading(false);
    }
  }

  // ─────────────────────────────────────────
  // PDF Download
  // ─────────────────────────────────────────
  downloadBtn.addEventListener('click', () => {
    window.open('/api/report', '_blank');
  });

  // ─────────────────────────────────────────
  // Analyze button
  // ─────────────────────────────────────────
  analyzeBtn.addEventListener('click', runAnalysis);

  // ─────────────────────────────────────────
  // Load 3D viewer on page load
  // ─────────────────────────────────────────
  setTimeout(() => {
    Viewer3D.loadVolume(42);
  }, 500);

});