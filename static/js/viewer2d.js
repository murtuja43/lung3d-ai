// ═══════════════════════════════════════════
// 2D CT SLICE VIEWER
// ═══════════════════════════════════════════

const Viewer2D = (() => {

  // Current state
  let currentAxis   = 'axial';
  let currentSlices = null;

  // Canvas elements
  const ctCanvas   = document.getElementById('ct-canvas');
  const maskCanvas = document.getElementById('mask-canvas');
  const ctCtx      = ctCanvas.getContext('2d');
  const maskCtx    = maskCanvas.getContext('2d');

  // ─────────────────────────────────────────
  // Draw a base64 PNG onto a canvas
  // ─────────────────────────────────────────
  function drawBase64Image(ctx, canvas, base64String) {
    return new Promise((resolve) => {
      const img = new Image();
      img.onload = () => {
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Scale image to fill canvas
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        resolve();
      };
      img.src = 'data:image/png;base64,' + base64String;
    });
  }

  // ─────────────────────────────────────────
  // Draw the CT scan + mask overlay
  // ─────────────────────────────────────────
  async function renderSlice(axis) {
    if (!currentSlices) return;

    const sliceData = currentSlices[axis];
    const maskData  = currentSlices[axis + '_mask'];

    // Draw CT scan
    await drawBase64Image(ctCtx, ctCanvas, sliceData);

    // Draw mask canvas (abnormal regions in red overlay)
    const img = new Image();
    img.onload = () => {
      // Draw original mask in grayscale first
      maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
      maskCtx.drawImage(img, 0, 0, maskCanvas.width, maskCanvas.height);

      // Add red overlay for abnormal pixels
      const imageData = maskCtx.getImageData(
        0, 0, maskCanvas.width, maskCanvas.height
      );
      const data = imageData.data;

      for (let i = 0; i < data.length; i += 4) {
        const brightness = data[i]; // R channel
        if (brightness > 30) {
          // Abnormal pixel — paint red
          data[i]     = 255; // R
          data[i + 1] = 50;  // G
          data[i + 2] = 50;  // B
          data[i + 3] = 200; // A
        } else {
          // Normal pixel — dark background
          data[i]     = 10;
          data[i + 1] = 20;
          data[i + 2] = 30;
          data[i + 3] = 255;
        }
      }

      maskCtx.putImageData(imageData, 0, 0);
    };
    img.src = 'data:image/png;base64,' + maskData;
  }

  // ─────────────────────────────────────────
  // Fetch slices from backend API
  // ─────────────────────────────────────────
  async function loadSlices(seed = 42) {
    try {
      const response = await fetch(`/api/slices?seed=${seed}`);
      const data     = await response.json();
      currentSlices  = data.slices;

      // Render the current axis
      await renderSlice(currentAxis);

      return true;
    } catch (err) {
      console.error('Failed to load CT slices:', err);
      return false;
    }
  }

  // ─────────────────────────────────────────
  // Set up tab buttons (Axial / Coronal / Sagittal)
  // ─────────────────────────────────────────
  function initTabs() {
    const tabButtons = document.querySelectorAll('.tab-btn');

    tabButtons.forEach(btn => {
      btn.addEventListener('click', async () => {
        // Update active tab style
        tabButtons.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');

        // Switch axis and re-render
        currentAxis = btn.dataset.axis;
        await renderSlice(currentAxis);
      });
    });
  }

  // ─────────────────────────────────────────
  // Draw placeholder grid on canvas at start
  // ─────────────────────────────────────────
  function drawPlaceholder(ctx, canvas, label) {
    ctx.fillStyle = '#050a12';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    ctx.strokeStyle = '#1e2d45';
    ctx.lineWidth = 1;

    // Draw grid
    for (let i = 0; i < canvas.width; i += 32) {
      ctx.beginPath();
      ctx.moveTo(i, 0);
      ctx.lineTo(i, canvas.height);
      ctx.stroke();
    }
    for (let i = 0; i < canvas.height; i += 32) {
      ctx.beginPath();
      ctx.moveTo(0, i);
      ctx.lineTo(canvas.width, i);
      ctx.stroke();
    }

    // Center text
    ctx.fillStyle = '#1e2d45';
    ctx.font = '13px Segoe UI';
    ctx.textAlign = 'center';
    ctx.fillText(label, canvas.width / 2, canvas.height / 2);
  }

  // ─────────────────────────────────────────
  // Initialize
  // ─────────────────────────────────────────
  function init() {
    drawPlaceholder(ctCtx,   ctCanvas,   'CT Scan — Run Analysis');
    drawPlaceholder(maskCtx, maskCanvas, 'Abnormal Mask — Run Analysis');
    initTabs();
  }

  // Public API
  return { init, loadSlices };

})();

// Auto-initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
  Viewer2D.init();
});