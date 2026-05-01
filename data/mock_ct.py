import numpy as np

def generate_ct_scan(shape=(64, 64, 64), seed=42,
                     tb_detected=True, confidence=0.5):
    np.random.seed(seed)
    volume = np.zeros(shape, dtype=np.float32)

    cx, cy, cz = shape[0]//2, shape[1]//2, shape[2]//2

    X, Y, Z = np.meshgrid(
        np.arange(shape[0]),
        np.arange(shape[1]),
        np.arange(shape[2]),
        indexing='ij'
    )

    # ── Real lung shapes ──
    # Lungs are tall, narrow, and tapered at top/bottom
    # Left lung (slightly smaller — heart pushes it)
    def lung_mask(cx, cy, cz, rx, ry, rz, taper=0.6):
        # Normalized coords
        nx = (X - cx) / rx
        ny = (Y - cy) / ry
        nz = (Z - cz) / rz
        # Taper: narrower at top and bottom
        taper_factor = 1.0 - taper * (ny ** 2)
        taper_factor = np.clip(taper_factor, 0.1, 1.0)
        dist = (nx / taper_factor)**2 + ny**2 + nz**2
        return dist <= 1.0

    # Left lung — taller, narrower
    left = lung_mask(
        cx - 13, cy, cz,
        rx=10, ry=20, rz=12, taper=0.5
    )
    # Right lung — slightly wider
    right = lung_mask(
        cx + 14, cy, cz,
        rx=11, ry=20, rz=12, taper=0.5
    )

    # Hollow out center (lungs are not solid)
    left_inner = lung_mask(
        cx - 13, cy, cz,
        rx=6, ry=16, rz=8, taper=0.4
    )
    right_inner = lung_mask(
        cx + 14, cy, cz,
        rx=7, ry=16, rz=8, taper=0.4
    )

    lung_tissue = (left | right) & ~(left_inner & right_inner)
    volume[lung_tissue] = 0.2

    # Add texture noise
    noise = np.random.normal(0, 0.025, shape).astype(np.float32)
    volume = np.clip(volume + noise, 0, 1)

    # ── Abnormal nodules (only if TB) ──
    abnormal_mask = np.zeros(shape, dtype=np.uint8)

    if tb_detected:
        # Scale nodule size with confidence
        base_radius = 3 + int(confidence * 4)
        nodules = [
            (cx - 13, cy - 6, cz + 4, base_radius),
            (cx + 14, cy + 5, cz - 3, max(2, base_radius - 1)),
        ]
        for nx2, ny2, nz2, radius in nodules:
            dist = np.sqrt(
                (X - nx2)**2 +
                (Y - ny2)**2 +
                (Z - nz2)**2
            )
            nodule_region = dist <= radius
            volume[nodule_region]        = 0.85
            abnormal_mask[nodule_region] = 1
    else:
        nodules = []

    return volume, abnormal_mask, nodules if tb_detected else []


def get_slice(volume, axis='axial', index=None):
    mid = volume.shape[0] // 2 if index is None else index
    if axis == 'axial':
        return volume[mid, :, :]
    elif axis == 'coronal':
        return volume[:, mid, :]
    elif axis == 'sagittal':
        return volume[:, :, mid]


def compute_abnormality_score(volume, abnormal_mask):
    lung_voxels     = np.sum(volume > 0.15)
    abnormal_voxels = np.sum(abnormal_mask)
    if lung_voxels == 0:
        return 0.0
    return round(float(abnormal_voxels / lung_voxels), 4)