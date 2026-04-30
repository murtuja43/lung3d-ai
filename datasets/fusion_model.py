import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from datasets.model import TBClassifierWithFeatures, get_device
from datasets.tabular_model import (
    TabularModel, preprocess_patient_data,
    calculate_clinical_risk, NUM_FEATURES
)


# ─────────────────────────────────────────
# Fusion Model
# Combines CNN image features + tabular
# patient features for final prediction
# ─────────────────────────────────────────
class FusionModel(nn.Module):
    def __init__(self,
                 cnn_embedding_dim=512,
                 tabular_embedding_dim=32,
                 num_classes=2,
                 dropout=0.4):
        super(FusionModel, self).__init__()

        # CNN branch — ResNet18 backbone
        # We extract features before the final FC layer
        self.cnn_branch = TBClassifierWithFeatures(num_classes=2)

        # Tabular branch
        self.tabular_branch = TabularModel(
            input_dim   = NUM_FEATURES,
            hidden_dims = [64, 32],
            output_dim  = tabular_embedding_dim
        )

        # Fusion head — combines both embeddings
        fusion_input_dim = cnn_embedding_dim + tabular_embedding_dim

        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

        # Separate head for CNN-only prediction
        self.cnn_head = nn.Linear(cnn_embedding_dim, num_classes)

    def get_cnn_embedding(self, image):
        """Extract CNN features before classification layer"""
        x = self.cnn_branch.backbone.conv1(image)
        x = self.cnn_branch.backbone.bn1(x)
        x = self.cnn_branch.backbone.relu(x)
        x = self.cnn_branch.backbone.maxpool(x)
        x = self.cnn_branch.backbone.layer1(x)
        x = self.cnn_branch.backbone.layer2(x)
        x = self.cnn_branch.backbone.layer3(x)
        x = self.cnn_branch.backbone.layer4(x)
        x = self.cnn_branch.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        return x  # (batch, 512)

    def forward(self, image, tabular):
        # CNN embedding
        cnn_emb     = self.get_cnn_embedding(image)     # (B, 512)

        # Tabular embedding
        tabular_emb = self.tabular_branch(tabular)       # (B, 32)

        # Concatenate
        combined    = torch.cat([cnn_emb, tabular_emb], dim=1)  # (B, 544)

        # Final prediction
        output      = self.fusion_head(combined)         # (B, 2)
        return output

    def predict_with_gradcam(self, image, tabular):
        """Full prediction with Grad-CAM heatmap"""
        self.eval()

        with torch.no_grad():
            output = self.forward(image, tabular)
            probs  = torch.softmax(output, dim=1)
            pred   = output.argmax(dim=1).item()
            conf   = probs[0][pred].item()

        # Get Grad-CAM from CNN branch
        cam, _ = self.cnn_branch.get_gradcam(image, class_idx=pred)

        return pred, conf, cam


# ─────────────────────────────────────────
# Predictor class — used by Flask app
# ─────────────────────────────────────────
class TBPredictor:
    def __init__(self, model_path='models/tb_cnn_best.pth'):
        self.device     = get_device()
        self.model_path = model_path
        self.model      = None
        self._load_model()

    def _load_model(self):
        """Load the trained CNN model"""
        from datasets.model import TBClassifierWithFeatures
        self.cnn_model = TBClassifierWithFeatures(num_classes=2)

        model_path = Path(self.model_path)
        if model_path.exists():
            checkpoint = torch.load(
                model_path,
                map_location=self.device
            )
            self.cnn_model.load_state_dict(
                checkpoint['model_state_dict']
            )
            print(f"✅ CNN model loaded from {model_path}")
        else:
            print(f"⚠️  No trained model found at {model_path}")
            print("   Using untrained model (run training first)")

        self.cnn_model = self.cnn_model.to(self.device)
        self.cnn_model.eval()

        # Tabular model (rule-based, no training needed)
        self.tabular_model = TabularModel().to(self.device)
        self.tabular_model.eval()

    def predict(self, image_tensor, patient_data):
        """
        Full multimodal prediction.

        Args:
            image_tensor: preprocessed image tensor (1, 3, 224, 224)
            patient_data: dict with patient parameters

        Returns:
            dict with prediction, confidence, heatmap, explanation
        """
        image_tensor = image_tensor.to(self.device)

        # ── CNN prediction + Grad-CAM ──
        cam, pred_class = self.cnn_model.get_gradcam(
            image_tensor, class_idx=None
        )

        with torch.no_grad():
            cnn_output = self.cnn_model(image_tensor)
            cnn_probs  = torch.softmax(cnn_output, dim=1)
            cnn_conf   = cnn_probs[0][pred_class].item()

        # ── Clinical risk from patient data ──
        clinical_score, clinical_reasons = calculate_clinical_risk(
            patient_data
        )

        # ── Fusion: weighted combination ──
        # CNN gets 65% weight, clinical gets 35%
        cnn_tb_prob = cnn_probs[0][1].item()  # P(TB) from CNN
        fused_score = (0.65 * cnn_tb_prob) + (0.35 * clinical_score)
        fused_score = round(fused_score, 3)

        # ── Final prediction ──
        prediction  = "TB Detected" if fused_score >= 0.45 else "No TB Detected"
        confidence  = round(fused_score, 2)

        # ── Build explanation ──
        explanation = self._build_explanation(
            prediction, confidence,
            cnn_tb_prob, clinical_score,
            clinical_reasons, patient_data
        )

        return {
            'prediction':       prediction,
            'confidence':       confidence,
            'cnn_probability':  round(cnn_tb_prob, 3),
            'clinical_score':   clinical_score,
            'clinical_reasons': clinical_reasons,
            'explanation':      explanation,
            'heatmap':          cam,
        }

    def _build_explanation(self, prediction, confidence,
                           cnn_prob, clinical_score,
                           clinical_reasons, patient_data):
        lines = []
        lines.append(
            f"🔬 Prediction: {prediction} "
            f"(Confidence: {int(confidence*100)}%)"
        )
        lines.append("")
        lines.append("📊 Analysis Breakdown:")
        lines.append(
            f"  • CNN Image Analysis    : "
            f"{round(cnn_prob*100, 1)}% TB probability"
        )
        lines.append(
            f"  • Clinical Risk Score   : "
            f"{round(clinical_score*100, 1)}%"
        )
        lines.append(
            f"  • Fused Score           : "
            f"{round(confidence*100, 1)}%"
        )
        lines.append("")
        lines.append("📋 Clinical Reasoning:")
        if clinical_reasons:
            for r in clinical_reasons:
                lines.append(f"  ✅ {r}")
        else:
            lines.append("  ✅ No major clinical risk factors found")

        lines.append("")
        lines.append("👤 Patient Summary:")
        lines.append(
            f"  • Age    : {patient_data.get('age', 'N/A')} years"
        )
        lines.append(
            f"  • Sex    : {patient_data.get('sex', 'N/A').title()}"
        )
        lines.append(
            f"  • BMI    : {patient_data.get('bmi', 'N/A')}"
        )
        lines.append(
            f"  • Cough  : "
            f"{patient_data.get('cough_weeks', 0)} weeks"
        )
        lines.append("")
        lines.append(
            "⚠️  Disclaimer: This is a demo system only."
        )
        lines.append(
            "   Always consult a licensed medical professional."
        )
        return "\n".join(lines)


# ─────────────────────────────────────────
# Test the fusion model
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("🔍 Testing Fusion Model...\n")

    device = get_device()

    # Test FusionModel architecture
    model      = FusionModel().to(device)
    dummy_img  = torch.randn(2, 3, 224, 224).to(device)
    dummy_tab  = torch.randn(2, NUM_FEATURES).to(device)
    output     = model(dummy_img, dummy_tab)

    print(f"✅ Fusion Model forward pass!")
    print(f"   Image input  : {dummy_img.shape}")
    print(f"   Tabular input: {dummy_tab.shape}")
    print(f"   Output shape : {output.shape}")

    # Test TBPredictor
    print(f"\n🔍 Testing TBPredictor...\n")
    predictor = TBPredictor(model_path='models/tb_cnn_best.pth')

    sample_patient = {
        'age':          35,
        'sex':          'male',
        'bmi':          17.5,
        'cough_weeks':  4,
        'fever':        True,
        'night_sweats': True,
        'weight_loss':  True,
        'fatigue':      True,
        'chest_pain':   False,
        'tb_contact':   True,
        'prev_tb':      False,
    }

    # Dummy image tensor
    dummy_image = torch.randn(1, 3, 224, 224).to(predictor.device)
    result      = predictor.predict(dummy_image, sample_patient)

    print(f"\n✅ Prediction Result:")
    print(f"   Prediction  : {result['prediction']}")
    print(f"   Confidence  : {result['confidence']}")
    print(f"   CNN Prob    : {result['cnn_probability']}")
    print(f"   Clinical    : {result['clinical_score']}")
    print(f"\n{result['explanation']}")
    print(f"\n✅ Fusion Model is ready!")