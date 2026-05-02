from flask import Flask, jsonify, render_template, request, send_file
import numpy as np
import base64
import io
import torch
from PIL import Image
from pathlib import Path

# Old mock CT imports (kept for 3D viewer)
from data.mock_ct import generate_ct_scan, get_slice
from data.mock_ct import compute_abnormality_score

# New AI imports
from datasets.fusion_model import TBPredictor
from datasets.gradcam import preprocess_image, generate_heatmap
from datasets.tabular_model import preprocess_patient_data
from datasets.report import generate_pdf_report

app = Flask(__name__)

# ─────────────────────────────────────────
# Load model once at startup
# ─────────────────────────────────────────
print("🔄 Loading TB AI Model...")
try:
    predictor = TBPredictor(model_path='models/tb_cnn_best.pth')
    print("✅ Model ready!")
except Exception as e:
    print(f"⚠️  Model not found: {e}")
    predictor = None
    print("⚠️  Running in demo mode (rule-based only)")


# ─────────────────────────────────────────
# Helper: numpy array → base64 PNG
# ─────────────────────────────────────────
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


# ─────────────────────────────────────────
# Route: Main page
# ─────────────────────────────────────────
@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/app')
def index():
    return render_template('index.html')


# ─────────────────────────────────────────
# API: Analyze uploaded X-ray + patient data
# ─────────────────────────────────────────
@app.route('/api/analyze', methods=['POST'])
def analyze():
    try:
        if predictor is None:
            return jsonify({
            'error': 'Model not available in demo deployment. '
                     'Run locally with trained model for full features.'
            }), 503
    
        # ── Get patient form data ──
        patient_data = {
            'age':          float(request.form.get('age', 30)),
            'sex':          request.form.get('sex', 'male'),
            'bmi':          float(request.form.get('bmi', 22)),
            'cough_weeks':  float(request.form.get('cough_weeks', 0)),
            'fever':        request.form.get('fever') == 'true',
            'night_sweats': request.form.get('night_sweats') == 'true',
            'weight_loss':  request.form.get('weight_loss') == 'true',
            'fatigue':      request.form.get('fatigue') == 'true',
            'chest_pain':   request.form.get('chest_pain') == 'true',
            'tb_contact':   request.form.get('tb_contact') == 'true',
            'prev_tb':      request.form.get('prev_tb') == 'true',
        }

        # ── Get uploaded X-ray image ──
        if 'xray' not in request.files:
            return jsonify({'error': 'No X-ray image uploaded'}), 400

        xray_file = request.files['xray']
        if xray_file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # ── Generate heatmap ──
        heatmap_result = generate_heatmap(xray_file, predictor)

        # ── Run full prediction ──
        tensor, _ = preprocess_image(xray_file)
        result     = predictor.predict(tensor, patient_data)

        # ── Store data in session for PDF ──
        app.config['last_patient']    = patient_data
        app.config['last_result']     = result
        app.config['last_original']   = heatmap_result['original_b64']
        app.config['last_heatmap']    = heatmap_result['heatmap_b64']

        return jsonify({
            'prediction':       result['prediction'],
            'confidence':       result['confidence'],
            'cnn_probability':  result['cnn_probability'],
            'clinical_score':   result['clinical_score'],
            'clinical_reasons': result['clinical_reasons'],
            'explanation':      result['explanation'],
            'original_b64':     heatmap_result['original_b64'],
            'heatmap_b64':      heatmap_result['heatmap_b64'],
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ─────────────────────────────────────────
# API: Download PDF report
# ─────────────────────────────────────────
@app.route('/api/report', methods=['GET'])
def download_report():
    try:
        patient_data = app.config.get('last_patient', {})
        result       = app.config.get('last_result',  {})
        original_b64 = app.config.get('last_original', None)
        heatmap_b64  = app.config.get('last_heatmap',  None)

        if not result:
            return jsonify({'error': 'No analysis done yet'}), 400

        pdf_bytes = generate_pdf_report(
            patient_data      = patient_data,
            prediction_result = result,
            original_b64      = original_b64,
            heatmap_b64       = heatmap_b64,
        )

        return send_file(
            io.BytesIO(pdf_bytes),
            mimetype             = 'application/pdf',
            as_attachment        = True,
            download_name        = 'lung3d_ai_report.pdf'
        )

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ─────────────────────────────────────────
# API: Old mock CT slices (kept for 3D viewer)
# ─────────────────────────────────────────
@app.route('/api/slices')
def get_slices():
    seed   = request.args.get('seed', 42, type=int)
    volume, abnormal_mask, nodules = generate_ct_scan(seed=seed)

    slices = {}
    for axis in ['axial', 'coronal', 'sagittal']:
        slice_2d     = get_slice(volume, axis=axis)
        mask_2d      = get_slice(
            abnormal_mask.astype(np.float32), axis=axis
        )
        mask_visible = (mask_2d * 255).astype(np.float32)
        slices[axis]           = array_to_base64_png(
            slice_2d, colormap=True
        )
        slices[f'{axis}_mask'] = array_to_base64_png(
            mask_visible, colormap=False
        )

    return jsonify({'slices': slices, 'shape': list(volume.shape)})


# ─────────────────────────────────────────
# API: 3D volume point cloud
# ─────────────────────────────────────────
@app.route('/api/volume')
def get_volume():
    seed       = request.args.get('seed', 42, type=int)
    tb         = request.args.get('tb', 'false').lower() == 'true'
    confidence = request.args.get('confidence', 0.5, type=float)

    volume, abnormal_mask, nodules = generate_ct_scan(
        seed=seed,
        tb_detected=tb,
        confidence=confidence
    )

    step            = 2
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
                             'z': n[2], 'r': n[3]}
                            for n in nodules]
    })


# ─────────────────────────────────────────
# API: Old mock predict (kept as fallback)
# ─────────────────────────────────────────
@app.route('/api/predict')
def predict():
    from data.predictor import analyze_scan
    seed   = request.args.get('seed', 42, type=int)
    result = analyze_scan(seed=seed)
    return jsonify(result)


# ─────────────────────────────────────────
# Run
# ─────────────────────────────────────────
if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)