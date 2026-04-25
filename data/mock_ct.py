import numpy as np

def generate_ct_scan(shape=(64, 64, 64), seed=42):
    np.random.seed(seed)

    volume = np.zeros(shape, dtype=np.float32)

    cx, cy, cz = shape[0]//2, shape[1]//2, shape[2]//2

    # Build coordinate grids (fast NumPy — no loops)
    x = np.arange(shape[0])
    y = np.arange(shape[1])
    z = np.arange(shape[2])
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # Left lung ellipsoid
    left = ((X - cx + 14)**2 / 18**2 +
            (Y - cy)**2      / 22**2 +
            (Z - cz)**2      / 20**2)

    # Right lung ellipsoid
    right = ((X - cx - 14)**2 / 18**2 +
             (Y - cy)**2       / 22**2 +
             (Z - cz)**2       / 20**2)

    # Fill lung tissue
    lung_mask = (left <= 1.0) | (right <= 1.0)
    volume[lung_mask] = 0.2

    # Add noise
    noise = np.random.normal(0, 0.03, shape).astype(np.float32)
    volume = np.clip(volume + noise, 0, 1)

    # Add abnormal nodules
    nodules = [
        (cx - 14, cy - 5, cz + 5, 4),
        (cx + 14, cy + 4, cz - 3, 3),
    ]

    abnormal_mask = np.zeros(shape, dtype=np.uint8)

    for nx, ny, nz, radius in nodules:
        dist = np.sqrt((X - nx)**2 + (Y - ny)**2 + (Z - nz)**2)
        nodule_region = dist <= radius
        volume[nodule_region]       = 0.85
        abnormal_mask[nodule_region] = 1

    return volume, abnormal_mask, nodules


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
    score = abnormal_voxels / lung_voxels
    return round(float(score), 4)