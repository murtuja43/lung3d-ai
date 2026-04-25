const Viewer2D = (() => {

  let currentAxis   = 'axial';
  let currentSlices = null;

  const ctCanvas   = document.getElementById('ct-canvas');
  const maskCanvas = document.getElementById('mask-canvas');
  const ctCtx      = ctCanvas.getContext('2d');
  const maskCtx    = maskCanvas.getContext('2d');

  // Draw a base64 PNG onto a canvas
  function drawBase64Image(ctx, canvas, base64String) {
    return new Promise((resolve) => {
      const img = new Image();
      img.onload = () => {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        resolve();
      };
      img.src = 'data:image/png;base64,' + base64String;
    });
  }

  // Draw CT scan + red overlay for abnormal mask
  async function renderSlice(axis) {
    if (!currentSlices) return;

    const sliceData = currentSlices[axis];
    const maskData  = currentSlices[axis + '_mask'];

    // Draw CT scan on left canvas
    await drawBase64Image(ctCtx, ctCanvas, sliceData);

    // Draw abnormal mask on right canvas
    // First fill black background
    maskCtx.fillStyle = '#050a12';
    maskCtx.fillRect(0, 0, maskCanvas.width, maskCanvas.height);

    // Load mask image
    const img = new Image();
    img.onload = () => {
      // Draw to offscreen canvas to read pixels
      const offscreen = document.createElement('canvas');
      offscreen.width  = maskCanvas.width;
      offscreen.height = maskCanvas.height;
      const offCtx = offscreen.getContext('2d');
      offCtx.drawImage(img, 0, 0, maskCanvas.width, maskCanvas.height);

      const imageData = offCtx.getImageData(
        0, 0, maskCanvas.width, maskCanvas.height
      );
      const data = imageData.data;

      // Create output image data
      const outData = maskCtx.createImageData(maskCanvas.width, maskCanvas.height);
      const out     = outData.data;

      for (let i = 0; i < data.length; i += 4) {
        const brightness = data[i]; // R channel (grayscale)

        if (brightness > 10) {
          // Abnormal pixel — bright red
          out[i]     = 255;
          out[i + 1] = 40;
          out[i + 2] = 40;
          out[i + 3] = 255;
        } else {
          // Background — very dark blue
          out[i]     = 5;
          out[i + 1] = 10;
          out[i + 2] = 20;
          out[i + 3] = 255;
        }
      }

      maskCtx.putImageData(outData, 0, 0);

      // Draw a subtle border around abnormal regions
      maskCtx.strokeStyle = 'rgba(255, 100, 100, 0.3)';
      maskCtx.lineWidth   = 1;
      maskCtx.strokeRect(1, 1, maskCanvas.width - 2, maskCanvas.height - 2);
    };
    img.src = 'data:image/png;base64,' + maskData;
  }

  // Fetch slices from backend
  async function loadSlices(seed = 42) {
    try {
      const response = await fetch(`/api/slices?seed=${seed}`);
      const data     = await response.json();
      currentSlices  = data.slices;
      await renderSlice(currentAxis);
      return true;
    } catch (err) {
      console.error('Failed to load CT slices:', err);
      return false;
    }
  }

  // Tab switching
  function initTabs() {
    const tabButtons = document.querySelectorAll('.tab-btn');
    tabButtons.forEach(btn => {
      btn.addEventListener('click', async () => {
        tabButtons.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        currentAxis = btn.dataset.axis;
        await renderSlice(currentAxis);
      });
    });
  }

  // Placeholder grid
  function drawPlaceholder(ctx, canvas, label) {
    ctx.fillStyle = '#050a12';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = '#1e2d45';
    ctx.lineWidth   = 1;
    for (let i = 0; i < canvas.width; i += 32) {
      ctx.beginPath(); ctx.moveTo(i, 0);
      ctx.lineTo(i, canvas.height); ctx.stroke();
    }
    for (let i = 0; i < canvas.height; i += 32) {
      ctx.beginPath(); ctx.moveTo(0, i);
      ctx.lineTo(canvas.width, i); ctx.stroke();
    }
    ctx.fillStyle   = '#1e2d45';
    ctx.font        = '13px Segoe UI';
    ctx.textAlign   = 'center';
    ctx.fillText(label, canvas.width / 2, canvas.height / 2);
  }

  function init() {
    drawPlaceholder(ctCtx,   ctCanvas,   'CT Scan — Run Analysis');
    drawPlaceholder(maskCtx, maskCanvas, 'Abnormal Mask — Run Analysis');
    initTabs();
  }

  return { init, loadSlices };

})();

document.addEventListener('DOMContentLoaded', () => {
  Viewer2D.init();
});