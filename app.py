from flask import Flask, jsonify, render_template, request
import numpy as np
import base64
import io
from PIL import Image
from data.mock_ct import generate_ct_scan, get_slice, compute_abnormality_score
from data.predictor import analyze_scan

app = Flask(__name__)


def array_to_base64_png(slice_2d, colormap=True):
    normalized = slice_2d.copy().astype(np.float32)
    mn, mx = normalized.min(), normalized.max()
    if mx > mn:
        normalized = (normalized - mn) / (mx - mn)
    else:
        normalized = np.zeros_like(normalized)
    pixel_array = (normalized * 255).astype(np.uint8)

    if colormap:
        rgb = np.stack([
            (pixel_array * 0.2).astype(np.uint8),
            (pixel_array * 0.9).astype(np.uint8),
            (pixel_array * 0.3).astype(np.uint8),
        ], axis=-1)
        img = Image.fromarray(rgb, mode='RGB')
    else:
        img = Image.fromarray(pixel_array, mode='L')

    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/slices')
def get_slices():
    seed = request.args.get('seed', 42, type=int)
    volume, abnormal_mask, nodules = generate_ct_scan(seed=seed)

    slices = {}
    for axis in ['axial', 'coronal', 'sagittal']:
        slice_2d = get_slice(volume, axis=axis)
        mask_2d  = get_slice(abnormal_mask.astype(np.float32), axis=axis)

        # Force mask to be visible — scale it up
        mask_visible = (mask_2d * 255).astype(np.float32)

        slices[axis]              = array_to_base64_png(slice_2d, colormap=True)
        slices[f'{axis}_mask']    = array_to_base64_png(mask_visible, colormap=False)

    return jsonify({'slices': slices, 'shape': list(volume.shape)})


@app.route('/api/volume')
def get_volume():
    seed = request.args.get('seed', 42, type=int)
    volume, abnormal_mask, nodules = generate_ct_scan(seed=seed)

    # Use step=2 for denser point cloud
    step = 2
    lung_points     = []
    abnormal_points = []

    xs, ys, zs = np.where(volume > 0.15)
    for i in range(0, len(xs), step):
        x, y, z = int(xs[i]), int(ys[i]), int(zs[i])
        val = float(volume[x, y, z])
        nx  = (x / volume.shape[0]) * 2 - 1
        ny  = (y / volume.shape[1]) * 2 - 1
        nz  = (z / volume.shape[2]) * 2 - 1

        if abnormal_mask[x, y, z] == 1:
            abnormal_points.append([nx, ny, nz, val])
        else:
            lung_points.append([nx, ny, nz, val])

    return jsonify({
        'lung_points':     lung_points,
        'abnormal_points': abnormal_points,
        'nodules':         [{'x': n[0], 'y': n[1],
                             'z': n[2], 'r': n[3]} for n in nodules]
    })


@app.route('/api/predict')
def predict():
    seed   = request.args.get('seed', 42, type=int)
    result = analyze_scan(seed=seed)
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
    