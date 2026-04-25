// ═══════════════════════════════════════════
// MAIN.JS — Connects all components together
// ═══════════════════════════════════════════

document.addEventListener('DOMContentLoaded', () => {

  // ─────────────────────────────────────────
  // Element references
  // ─────────────────────────────────────────
  const analyzeBtn       = document.getElementById('analyze-btn');
  const seedInput        = document.getElementById('seed-input');
  const loadingEl        = document.getElementById('loading');
  const predictionResult = document.getElementById('prediction-result');
  const confidenceSection= document.getElementById('confidence-section');
  const confidenceBar    = document.getElementById('confidence-bar');
  const confidenceLabel  = document.getElementById('confidence-label');
  const explanationText  = document.getElementById('explanation-text');

  // ─────────────────────────────────────────
  // Show / hide loading spinner
  // ─────────────────────────────────────────
  function setLoading(isLoading) {
    if (isLoading) {
      loadingEl.classList.remove('hidden');
      analyzeBtn.disabled    = true;
      analyzeBtn.textContent = '⏳ Analyzing...';
    } else {
      loadingEl.classList.add('hidden');
      analyzeBtn.disabled    = false;
      analyzeBtn.textContent = '🔍 Analyze Scan';
    }
  }

  // ─────────────────────────────────────────
  // Render prediction result card
  // ─────────────────────────────────────────
  function renderPrediction(data) {
    const isTB = data.prediction === 'TB Detected';

    // Prediction badge
    predictionResult.className = isTB
      ? 'prediction-tb'
      : 'prediction-no-tb';

    predictionResult.innerHTML = isTB
      ? `⚠️ ${data.prediction}`
      : `✅ ${data.prediction}`;

    // Confidence bar
    confidenceSection.classList.remove('hidden');
    const pct = Math.round(data.confidence * 100);

    // Animate bar after short delay
    setTimeout(() => {
      confidenceBar.style.width = `${pct}%`;

      // Color the bar based on confidence level
      if (pct >= 70) {
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
  }

  // ─────────────────────────────────────────
  // Render explanation report
  // ─────────────────────────────────────────
  function renderExplanation(data) {
    explanationText.classList.remove('explanation-placeholder');
    explanationText.textContent = data.explanation;

    // Color-code lines by emoji prefix
    const lines = data.explanation.split('\n');
    const colored = lines.map(line => {
      if (line.includes('✅')) {
        return `<span style="color:#00ff88">${escapeHtml(line)}</span>`;
      } else if (line.includes('⚠️')) {
        return `<span style="color:#ffa502">${escapeHtml(line)}</span>`;
      } else if (line.includes('🔬')) {
        return `<span style="color:#00d4ff;font-weight:bold">${escapeHtml(line)}</span>`;
      } else if (line.includes('📊') || line.includes('📋')) {
        return `<span style="color:#8899aa">${escapeHtml(line)}</span>`;
      } else {
        return `<span style="color:#aabbcc">${escapeHtml(line)}</span>`;
      }
    });

    explanationText.innerHTML = colored.join('\n');
  }

  // ─────────────────────────────────────────
  // Helper: escape HTML special characters
  // ─────────────────────────────────────────
  function escapeHtml(str) {
    return str
      .replace(/&/g,  '&amp;')
      .replace(/</g,  '&lt;')
      .replace(/>/g,  '&gt;')
      .replace(/"/g,  '&quot;')
      .replace(/'/g,  '&#039;');
  }

  // ─────────────────────────────────────────
  // Show stats bar under prediction
  // ─────────────────────────────────────────
  function renderStats(data) {
    // Check if stats section exists, create if not
    let statsEl = document.getElementById('stats-section');
    if (!statsEl) {
      statsEl = document.createElement('div');
      statsEl.id = 'stats-section';
      statsEl.style.cssText = `
        margin-top: 14px;
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 8px;
      `;
      confidenceSection.after(statsEl);
    }

    const stats = [
      { label: 'Nodules Found',    value: data.nodule_count },
      { label: 'Largest Nodule',   value: `${data.max_nodule_size} units` },
      { label: 'Abnormal Tissue',  value: `${(data.abnormality_score * 100).toFixed(2)}%` },
      { label: 'Avg Intensity',    value: data.avg_intensity },
    ];

    statsEl.innerHTML = stats.map(s => `
      <div style="
        background: #0a0e1a;
        border: 1px solid #1e2d45;
        border-radius: 8px;
        padding: 8px 10px;
      ">
        <div style="color:#445566;font-size:10px;text-transform:uppercase;
                    letter-spacing:0.5px">${s.label}</div>
        <div style="color:#00d4ff;font-size:15px;font-weight:700;
                    margin-top:2px">${s.value}</div>
      </div>
    `).join('');
  }

  // ─────────────────────────────────────────
  // Main analyze function — runs on button click
  // ─────────────────────────────────────────
  async function runAnalysis() {
    const seed = parseInt(seedInput.value) || 42;

    setLoading(true);

    try {
      // Run all 3 requests in parallel for speed
      const [predictionRes, slicesOk, volumeOk] = await Promise.all([
        fetch(`/api/predict?seed=${seed}`).then(r => r.json()),
        Viewer2D.loadSlices(seed),
        Viewer3D.loadVolume(seed),
      ]);

      // Render prediction panel
      renderPrediction(predictionRes);
      renderExplanation(predictionRes);
      renderStats(predictionRes);

    } catch (err) {
      console.error('Analysis failed:', err);
      predictionResult.className   = '';
      predictionResult.textContent =
        '❌ Error running analysis. Check the console.';
    } finally {
      setLoading(false);
    }
  }

  // ─────────────────────────────────────────
  // Button click handler
  // ─────────────────────────────────────────
  analyzeBtn.addEventListener('click', runAnalysis);

  // Also allow pressing Enter in the seed input
  seedInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') runAnalysis();
  });

  // ─────────────────────────────────────────
  // Auto-run on page load with default seed
  // ─────────────────────────────────────────
  setTimeout(() => {
    runAnalysis();
  }, 500);

});