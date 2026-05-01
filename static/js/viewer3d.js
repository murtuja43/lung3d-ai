// ═══════════════════════════════════════════
// 3D LUNG VISUALIZATION — Three.js
// ═══════════════════════════════════════════

const Viewer3D = (() => {

  let scene, camera, renderer, pointCloud, abnormalCloud;
  let animationId = null;
  let isDragging  = false;
  let prevMouse   = { x: 0, y: 0 };
  let rotation    = { x: 0.3, y: 0.0 };
  let zoomLevel   = 3.5;

  const container = document.getElementById('canvas-3d-container');
  const canvas    = document.getElementById('canvas-3d');

  // ─────────────────────────────────────────
  // Initialize Three.js scene
  // ─────────────────────────────────────────
  function initScene() {
    // Scene
    scene = new THREE.Scene();

    // Camera
    const w = container.clientWidth;
    const h = container.clientHeight;
    camera = new THREE.PerspectiveCamera(60, w / h, 0.01, 100);
    camera.position.set(0, 0.2, zoomLevel);

    // Renderer
    renderer = new THREE.WebGLRenderer({
      canvas:    canvas,
      antialias: true,
      alpha:     true
    });
    renderer.setSize(w, h);
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setClearColor(0x000000, 0);

    // Lighting (subtle ambient)
    const ambient = new THREE.AmbientLight(0x334466, 1.5);
    scene.add(ambient);

    // Grid helper (subtle floor grid)
    const grid = new THREE.GridHelper(2, 10, 0x1e2d45, 0x1e2d45);
    grid.position.y = -1;
    scene.add(grid);

    // Axes helper (tiny)
    const axes = new THREE.AxesHelper(0.3);
    axes.position.set(-0.9, -0.9, 0);
    scene.add(axes);

    // Draw placeholder lungs
    drawPlaceholderLungs();

    // Start render loop
    animate();

    // Events
    initControls();
    window.addEventListener('resize', onResize);
  }

  // ─────────────────────────────────────────
  // Placeholder lung outline (wireframe ellipsoids)
  // ─────────────────────────────────────────
  function drawPlaceholderLungs() {
    const mat = new THREE.MeshBasicMaterial({
      color: 0x1e2d45, wireframe: true,
      opacity: 0.3, transparent: true
    });

    // LEFT LUNG — tall tapered shape
    const leftPoints = [];
    for (let i = 0; i <= 20; i++) {
      const t      = (i / 20) * Math.PI;
      const y      = Math.cos(t) * 0.55;
      const taper  = Math.sin(t) * 0.85;
      const rx     = 0.18 * taper;
      const rz     = 0.22 * taper;
      leftPoints.push({ y, rx, rz });
    }

    // Use CylinderGeometry stacked to simulate lung shape
    const leftGeo = new THREE.CylinderGeometry(
      0.12, 0.08, 1.1, 10, 1, true
    );
    leftGeo.scale(0.85, 1, 1);
    const leftMesh = new THREE.Mesh(leftGeo, mat);
    leftMesh.position.set(-0.25, 0.05, 0);
    scene.add(leftMesh);

    // Left lung top dome
    const leftTopGeo = new THREE.SphereGeometry(0.12, 8, 6,
      0, Math.PI*2, 0, Math.PI/2);
    leftTopGeo.scale(0.85, 1, 1);
    const leftTop = new THREE.Mesh(leftTopGeo, mat);
    leftTop.position.set(-0.25, 0.6, 0);
    scene.add(leftTop);

    // Left lung bottom dome
    const leftBotGeo = new THREE.SphereGeometry(0.09, 8, 6,
      0, Math.PI*2, Math.PI/2, Math.PI/2);
    leftBotGeo.scale(0.85, 1, 1);
    const leftBot = new THREE.Mesh(leftBotGeo, mat);
    leftBot.position.set(-0.25, -0.5, 0);
    scene.add(leftBot);

    // RIGHT LUNG — slightly wider
    const rightGeo = new THREE.CylinderGeometry(
      0.13, 0.09, 1.1, 10, 1, true
    );
    const rightMesh = new THREE.Mesh(rightGeo, mat);
    rightMesh.position.set(0.27, 0.05, 0);
    scene.add(rightMesh);

    const rightTopGeo = new THREE.SphereGeometry(0.13, 8, 6,
      0, Math.PI*2, 0, Math.PI/2);
    const rightTop = new THREE.Mesh(rightTopGeo, mat);
    rightTop.position.set(0.27, 0.6, 0);
    scene.add(rightTop);

    const rightBotGeo = new THREE.SphereGeometry(0.10, 8, 6,
      0, Math.PI*2, Math.PI/2, Math.PI/2);
    const rightBot = new THREE.Mesh(rightBotGeo, mat);
    rightBot.position.set(0.27, -0.5, 0);
    scene.add(rightBot);

    scene.userData.placeholders = [
      leftMesh, leftTop, leftBot,
      rightMesh, rightTop, rightBot
    ];
  }

  // ─────────────────────────────────────────
  // Remove placeholder lungs
  // ─────────────────────────────────────────
  function clearPlaceholders() {
    if (scene.userData.placeholders) {
      scene.userData.placeholders.forEach(m => scene.remove(m));
      scene.userData.placeholders = [];
    }
  }

  // ─────────────────────────────────────────
  // Build point cloud from API data
  // ─────────────────────────────────────────
  function buildPointCloud(points, color) {
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(points.length * 3);
    const colors    = new Float32Array(points.length * 3);

    const baseColor = new THREE.Color(color);

    points.forEach((p, i) => {
      positions[i * 3]     = p[0];
      positions[i * 3 + 1] = p[1];
      positions[i * 3 + 2] = p[2];

      // Slightly vary brightness by intensity value
      const intensity = p[3] || 1.0;
      colors[i * 3]     = baseColor.r * intensity;
      colors[i * 3 + 1] = baseColor.g * intensity;
      colors[i * 3 + 2] = baseColor.b * intensity;
    });

    geometry.setAttribute('position',
      new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color',
      new THREE.BufferAttribute(colors, 3));

    const material = new THREE.PointsMaterial({
      size:         0.025,
      vertexColors: true,
      transparent:  true,
      opacity:      0.85,
      sizeAttenuation: true
    });

    return new THREE.Points(geometry, material);
  }

  // ─────────────────────────────────────────
  // Load 3D volume data from backend
  // ─────────────────────────────────────────
  async function loadVolume(seed = 42, tbDetected = false, confidence = 0.5) {
    try {
      const response = await fetch(
        `/api/volume?seed=${seed}&tb=${tbDetected}&confidence=${confidence}`
      );
      const data     = await response.json();

      // Remove old clouds
      if (pointCloud)    scene.remove(pointCloud);
      if (abnormalCloud) scene.remove(abnormalCloud);
      clearPlaceholders();

      // Build healthy lung point cloud (green)
      if (data.lung_points.length > 0) {
        pointCloud = buildPointCloud(data.lung_points, 0x00cc55);
        scene.add(pointCloud);
      }

      // Build abnormal point cloud (red, larger points)
      if (data.abnormal_points.length > 0) {
        abnormalCloud = buildPointCloud(data.abnormal_points, 0xff3333);
        abnormalCloud.material.size    = 0.05;
        abnormalCloud.material.opacity = 1.0;
        scene.add(abnormalCloud);
      }

      return true;
    } catch (err) {
      console.error('Failed to load 3D volume:', err);
      return false;
    }
  }

  // ─────────────────────────────────────────
  // Animation loop
  // ─────────────────────────────────────────
  function animate() {
    animationId = requestAnimationFrame(animate);

    // Auto-rotate slowly when not dragging
    if (!isDragging) {
      rotation.y += 0.003;
    }

    // Apply rotation to point clouds
    if (pointCloud)    {
      pointCloud.rotation.x    = rotation.x;
      pointCloud.rotation.y    = rotation.y;
    }
    if (abnormalCloud) {
      abnormalCloud.rotation.x = rotation.x;
      abnormalCloud.rotation.y = rotation.y;
    }

    // Pulse abnormal regions
    if (abnormalCloud) {
      const pulse = 0.04 * Math.sin(Date.now() * 0.004);
      abnormalCloud.material.size = 0.05 + pulse;
    }

    renderer.render(scene, camera);
  }

  // ─────────────────────────────────────────
  // Mouse / touch controls (drag to rotate)
  // ─────────────────────────────────────────
  function initControls() {
    canvas.addEventListener('mousedown', (e) => {
      isDragging = true;
      prevMouse  = { x: e.clientX, y: e.clientY };
    });

    canvas.addEventListener('mousemove', (e) => {
      if (!isDragging) return;
      const dx = e.clientX - prevMouse.x;
      const dy = e.clientY - prevMouse.y;
      rotation.y += dx * 0.008;
      rotation.x += dy * 0.008;
      prevMouse = { x: e.clientX, y: e.clientY };
    });

    canvas.addEventListener('mouseup',    () => { isDragging = false; });
    canvas.addEventListener('mouseleave', () => { isDragging = false; });

    // Scroll to zoom
    canvas.addEventListener('wheel', (e) => {
      zoomLevel += e.deltaY * 0.002;
      zoomLevel  = Math.max(1.0, Math.min(5.0, zoomLevel));
      camera.position.z = zoomLevel;
    });
  }

  // ─────────────────────────────────────────
  // Handle window resize
  // ─────────────────────────────────────────
  function onResize() {
    const w = container.clientWidth;
    const h = container.clientHeight;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
  }

  // ─────────────────────────────────────────
  // Initialize
  // ─────────────────────────────────────────
  function init() {
    initScene();
  }

  // Public API
  return { init, loadVolume };

})();

// Auto-initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
  Viewer3D.init();
});