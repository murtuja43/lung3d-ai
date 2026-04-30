import torch
import numpy as np
import base64
import io
from PIL import Image
from torchvision import transforms

# ─────────────────────────────────────────
# Image preprocessing for CNN input
# ─────────────────────────────────────────
def preprocess_image(image_file):
    """
    Convert uploaded image file → tensor for CNN.
    Works with file path or file-like object.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    if isinstance(image_file, str):
        img = Image.open(image_file).convert('RGB')
    else:
        img = Image.open(image_file).convert('RGB')

    tensor = transform(img).unsqueeze(0)  # (1, 3, 224, 224)
    return tensor, img


# ─────────────────────────────────────────
# Overlay heatmap on original image
# ─────────────────────────────────────────
def apply_heatmap(original_img, cam, alpha=0.5):
    """
    Overlay Grad-CAM heatmap on original image.

    Args:
        original_img : PIL Image
        cam          : numpy array (H, W) values 0-1
        alpha        : blend factor

    Returns:
        PIL Image with heatmap overlay
    """
    # Resize CAM to match original image
    cam_resized = np.array(
        Image.fromarray(
            (cam * 255).astype(np.uint8)
        ).resize(
            original_img.size,
            Image.BILINEAR
        )
    ) / 255.0

    # Create colormap (blue → green → red)
    heatmap_rgb = np.zeros(
        (*cam_resized.shape, 3), dtype=np.float32
    )

    # Blue for low activation
    heatmap_rgb[..., 2] = np.clip(1.5 - 4 * cam_resized, 0, 1)

    # Green for medium activation
    heatmap_rgb[..., 1] = np.clip(
        np.where(cam_resized < 0.5,
                 4 * cam_resized,
                 4 * (1 - cam_resized)), 0, 1
    )

    # Red for high activation
    heatmap_rgb[..., 0] = np.clip(4 * cam_resized - 1.5, 0, 1)

    heatmap_pil = Image.fromarray(
        (heatmap_rgb * 255).astype(np.uint8)
    )

    # Convert original to RGB numpy
    orig_array = np.array(
        original_img.resize(
            original_img.size
        ).convert('RGB')
    ).astype(np.float32) / 255.0

    # Blend
    blended = (
        alpha * heatmap_rgb +
        (1 - alpha) * orig_array
    )
    blended = np.clip(blended * 255, 0, 255).astype(np.uint8)

    return Image.fromarray(blended)


# ─────────────────────────────────────────
# Convert PIL image → base64 string
# ─────────────────────────────────────────
def image_to_base64(pil_image):
    """Convert PIL Image to base64 PNG string for frontend"""
    buffer = io.BytesIO()
    pil_image.save(buffer, format='PNG')
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')


# ─────────────────────────────────────────
# Full pipeline: image → heatmap base64
# ─────────────────────────────────────────
def generate_heatmap(image_file, predictor):
    """
    Full pipeline:
    1. Preprocess image
    2. Run CNN + Grad-CAM
    3. Overlay heatmap
    4. Return base64 images

    Returns dict with:
        original_b64  : original X-ray as base64
        heatmap_b64   : heatmap overlay as base64
        prediction    : TB / No TB
        confidence    : float
    """
    # Preprocess
    tensor, original_img = preprocess_image(image_file)

    # Resize original for display
    display_img = original_img.resize((224, 224)).convert('RGB')

    # Run prediction + get CAM
    tensor = tensor.to(predictor.device)
    cam, pred_class = predictor.cnn_model.get_gradcam(
        tensor, class_idx=None
    )

    with torch.no_grad():
        output = predictor.cnn_model(tensor)
        probs  = torch.softmax(output, dim=1)
        cnn_tb_prob = probs[0][1].item()

    # Apply heatmap overlay
    heatmap_img = apply_heatmap(display_img, cam, alpha=0.55)

    # Convert to base64
    original_b64 = image_to_base64(display_img)
    heatmap_b64  = image_to_base64(heatmap_img)

    return {
        'original_b64': original_b64,
        'heatmap_b64':  heatmap_b64,
        'cnn_tb_prob':  round(cnn_tb_prob, 3),
        'pred_class':   pred_class,
    }


# ─────────────────────────────────────────
# Test with a sample image
# ─────────────────────────────────────────
if __name__ == "__main__":
    import sys
    from pathlib import Path
    from datasets.fusion_model import TBPredictor

    print("🔍 Testing Grad-CAM...\n")

    # Load predictor
    predictor = TBPredictor(model_path='models/tb_cnn_best.pth')

    # Find a sample image
    sample_dirs = [
        Path("datasets/montgomery/CXR_png"),
        Path("datasets/shenzhen/CXR_png"),
    ]

    sample_img = None
    for d in sample_dirs:
        imgs = list(d.glob("*.png"))
        if imgs:
            sample_img = imgs[0]
            break

    if sample_img is None:
        print("⚠️  No sample image found!")
        print("   Creating dummy test instead...")
        dummy = Image.fromarray(
            np.random.randint(0, 255,
            (224, 224, 3), dtype=np.uint8)
        )
        dummy.save("/tmp/test_xray.png")
        sample_img = "/tmp/test_xray.png"

    print(f"📁 Using image: {sample_img}")

    result = generate_heatmap(str(sample_img), predictor)

    print(f"\n✅ Heatmap generated!")
    print(f"   CNN TB Probability : {result['cnn_tb_prob']}")
    print(f"   Prediction class   : {result['pred_class']}")
    print(f"   Original base64    : "
          f"{result['original_b64'][:30]}...")
    print(f"   Heatmap base64     : "
          f"{result['heatmap_b64'][:30]}...")

    # Save heatmap for visual inspection
    heatmap_bytes = base64.b64decode(result['heatmap_b64'])
    heatmap_pil   = Image.open(io.BytesIO(heatmap_bytes))
    heatmap_pil.save("models/sample_heatmap.png")
    print(f"\n💾 Heatmap saved to models/sample_heatmap.png")
    print(f"✅ Grad-CAM is ready!")