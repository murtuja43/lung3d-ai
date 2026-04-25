import numpy as np

def generate_ct_scan(shape=(64, 64, 64), seed=42):
    """
    Generate a mock 3D lung CT scan as a NumPy array.
    Values simulate Hounsfield Units (HU):
      - Air / background: ~ -1000 HU (near 0 in our scale)
      - Lung tissue:      ~ -500 HU  (near 0.2)
      - Abnormal nodule:  ~ +100 HU  (near 0.7–1.0)
    """
    np.random.seed(seed)

    # Start with background (air)
    volume = np.zeros(shape, dtype=np.float32)

    cx, cy, cz = shape[0]//2, shape[1]//2, shape[2]//2

    # --- Draw left lung ---
    for x in range(shape[0]):
        for y in range(shape[1]):
            for z in range(shape[2]):
                # Left lung ellipsoid
                left = ((x - cx + 14)**2 / 18**2 +
                        (y - cy)**2      / 22**2 +
                        (z - cz)**2      / 20**2)
                # Right lung ellipsoid
                right = ((x - cx - 14)**2 / 18**2 +
                         (y - cy)**2       / 22**2 +
                         (z - cz)**2       / 20**2)

                if left <= 1.0 or right <= 1.0:
                    volume[x, y, z] = 0.2  # healthy lung tissue

    # --- Add random noise (texture) ---
    noise = np.random.normal(0, 0.03, shape).astype(np.float32)
    volume = np.clip(volume + noise, 0, 1)

    # --- Add abnormal nodules (simulated TB lesions) ---
    nodules = [
        (cx - 14, cy - 5, cz + 5, 4),   # left lung nodule
        (cx + 14, cy + 4, cz - 3, 3),   # right lung nodule
    ]

    abnormal_mask = np.zeros(shape, dtype=np.uint8)

    for nx, ny, nz, radius in nodules:
        for x in range(shape[0]):
            for y in range(shape[1]):
                for z in range(shape[2]):
                    dist = np.sqrt((x-nx)**2 + (y-ny)**2 + (z-nz)**2)
                    if dist <= radius:
                        volume[x, y, z] = 0.85   # abnormal tissue (bright)
                        abnormal_mask[x, y, z] = 1

    return volume, abnormal_mask, nodules


def get_slice(volume, axis='axial', index=None):
    """
    Extract a 2D slice from the 3D volume.
    axis: 'axial', 'coronal', or 'sagittal'
    """
    mid = volume.shape[0] // 2 if index is None else index

    if axis == 'axial':
        return volume[mid, :, :]
    elif axis == 'coronal':
        return volume[:, mid, :]
    elif axis == 'sagittal':
        return volume[:, :, mid]


def compute_abnormality_score(volume, abnormal_mask):
    """
    Calculate what percentage of lung tissue is abnormal.
    Returns a score between 0 and 1.
    """
    lung_voxels = np.sum(volume > 0.15)
    abnormal_voxels = np.sum(abnormal_mask)

    if lung_voxels == 0:
        return 0.0

    score = abnormal_voxels / lung_voxels
    return round(float(score), 4)