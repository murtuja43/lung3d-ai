import torch
import torch.nn as nn
from torchvision import models

# ─────────────────────────────────────────
# Device setup — use MPS (Mac GPU) if available
# ─────────────────────────────────────────
def get_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✅ Using Apple MPS (Mac GPU) for training")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("✅ Using CUDA GPU for training")
    else:
        device = torch.device("cpu")
        print("✅ Using CPU for training")
    return device


# ─────────────────────────────────────────
# CNN Model — ResNet18 with custom head
# ─────────────────────────────────────────
class TBClassifier(nn.Module):
    def __init__(self, num_classes=2, dropout=0.5):
        super(TBClassifier, self).__init__()

        # Load pre-trained ResNet18
        # (trained on ImageNet — already knows edges, shapes, textures)
        self.backbone = models.resnet18(weights='IMAGENET1K_V1')

        # Freeze early layers — keep pre-trained features
        for name, param in self.backbone.named_parameters():
            if 'layer4' not in name and 'fc' not in name:
                param.requires_grad = False

        # Replace the final classification head
        # ResNet18 original: fc(512 → 1000 classes)
        # Our version:       fc(512 → 2 classes: Normal / TB)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


# ─────────────────────────────────────────
# Feature extractor for Grad-CAM later
# ─────────────────────────────────────────
class TBClassifierWithFeatures(nn.Module):
    def __init__(self, num_classes=2, dropout=0.5):
        super(TBClassifierWithFeatures, self).__init__()

        self.backbone = models.resnet18(weights='IMAGENET1K_V1')

        for name, param in self.backbone.named_parameters():
            if 'layer4' not in name and 'fc' not in name:
                param.requires_grad = False

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

        # Store gradients for Grad-CAM
        self.gradients  = None
        self.activations = None

        # Hook into layer4 for feature maps
        self.backbone.layer4.register_forward_hook(self._save_activation)
        self.backbone.layer4.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def forward(self, x):
        return self.backbone(x)

    def get_gradcam(self, x, class_idx=None):
        """Generate Grad-CAM heatmap for input x"""
        self.eval()
        x = x.unsqueeze(0) if x.dim() == 3 else x

        # Forward pass
        output = self.forward(x)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        # Backward pass
        self.zero_grad()
        score = output[0, class_idx]
        score.backward()

        # Compute Grad-CAM
        gradients  = self.gradients[0].cpu()
        activations = self.activations[0].cpu()

        weights = gradients.mean(dim=(1, 2))
        cam     = torch.zeros(activations.shape[1:], dtype=torch.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = torch.clamp(cam, min=0)  # ReLU

        # Normalize to [0, 1]
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam.numpy(), class_idx


# ─────────────────────────────────────────
# Model summary
# ─────────────────────────────────────────
def print_model_summary(model):
    total_params    = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters()
                          if p.requires_grad)
    print(f"\n📊 Model Summary:")
    print(f"   Total parameters     : {total_params:,}")
    print(f"   Trainable parameters : {trainable_params:,}")
    print(f"   Frozen parameters    : {total_params - trainable_params:,}")


# ─────────────────────────────────────────
# Test the model
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("🔍 Testing CNN Model...\n")

    device = get_device()
    model  = TBClassifier(num_classes=2).to(device)

    print_model_summary(model)

    # Test forward pass with dummy input
    dummy_input  = torch.randn(4, 3, 224, 224).to(device)
    output       = model(dummy_input)

    print(f"\n✅ Forward pass successful!")
    print(f"   Input shape  : {dummy_input.shape}")
    print(f"   Output shape : {output.shape}")
    print(f"   Output sample: {output[0].detach().cpu().numpy()}")
    print(f"\n✅ CNN Model is ready for training!")