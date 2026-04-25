from flask import Flask, jsonify, render_template, request
import numpy as np
import base64
import io
from PIL import Image
from data.mock_ct import generate_ct_scan, get_slice, compute_abnormality_score
from data.predictor import analyze_scan

app = Flask(__name__)


# ─────────────────────────────────────────
# Helper: convert a 2D NumPy array → PNG base64 string
# ─────────────────────────────────────────
def array_to_base64_png(slice_2d, colormap=True):
    """Convert a 2D numpy array to a base64-encoded PNG image."""
    # Normalize to 0-255
    normalized = (slice_2d - slice_2d.min())
    if normalized.max() > 0:
        normalized = normalized / normalized.max()
    pixel_array = (normalized * 255).astype(np.uint8)

    if colormap:
        # Apply a simple grayscale → green tint for lung tissue
        rgb = np.stack([
            (pixel_array * 0.2).astype(np.uint8),   # R
            (pixel_array * 0.8).astype(np.uint8),   # G
            (pixel_array * 0.3).astype(np.uint8),   # B
        ], axis=-1)
        img = Image.fromarray(rgb, mode='RGB')
    else:
        img = Image.fromarray(pixel_array, mode='L')

    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')


# ─────────────────────────────────────────
# Route: Main page
# ─────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


# ─────────────────────────────────────────
# API: Get 2D CT slices (axial, coronal, sagittal)
# ─────────────────────────────────────────
@app.route('/api/slices')
def get_slices():
    seed = request.args.get('seed', 42, type=int)
    volume, abnormal_mask, nodules = generate_ct_scan(seed=seed)

    slices = {}
    for axis in ['axial', 'coronal', 'sagittal']:
        slice_2d       = get_slice(volume, axis=axis)
        mask_2d        = get_slice(abnormal_mask.astype(np.float32), axis=axis)
        slices[axis]   = array_to_base64_png(slice_2d)
        slices[f'{axis}_mask'] = array_to_base64_png(mask_2d, colormap=False)

    return jsonify({
        'slices': slices,
        'shape':  list(volume.shape)
    })


# ─────────────────────────────────────────
# API: Get 3D point cloud data for visualization
# ─────────────────────────────────────────
@app.route('/api/volume')
def get_volume():
    seed = request.args.get('seed', 42, type=int)
    volume, abnormal_mask, nodules = generate_ct_scan(seed=seed)

    # Sample points where lung tissue exists (threshold > 0.15)
    # Downsample for performance — take every 3rd voxel
    step = 3
    lung_points     = []
    abnormal_points = []

    for x in range(0, volume.shape[0], step):
        for y in range(0, volume.shape[1], step):
            for z in range(0, volume.shape[2], step):
                val = float(volume[x, y, z])
                if val > 0.15:
                    # Normalize coords to [-1, 1]
                    nx = (x / volume.shape[0]) * 2 - 1
                    ny = (y / volume.shape[1]) * 2 - 1
                    nz = (z / volume.shape[2]) * 2 - 1

                    if abnormal_mask[x, y, z] == 1:
                        abnormal_points.append([nx, ny, nz, val])
                    else:
                        lung_points.append([nx, ny, nz, val])

    return jsonify({
        'lung_points':     lung_points,
        'abnormal_points': abnormal_points,
        'nodules':         [{'x': n[0], 'y': n[1], 'z': n[2], 'r': n[3]}
                            for n in nodules]
    })


# ─────────────────────────────────────────
# API: Get TB prediction result
# ─────────────────────────────────────────
@app.route('/api/predict')
def predict():
    seed = request.args.get('seed', 42, type=int)
    result = analyze_scan(seed=seed)
    return jsonify(result)


# ─────────────────────────────────────────
# Run the app
# ─────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=True, port=5000)