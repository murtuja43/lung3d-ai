import torch
import torch.nn as nn
import numpy as np

# ─────────────────────────────────────────
# Patient parameter features (11 inputs)
# ─────────────────────────────────────────
FEATURE_NAMES = [
    'age',              # 0-1 normalized
    'sex',              # 0=Female, 1=Male
    'bmi',              # 0-1 normalized
    'cough_weeks',      # 0-1 normalized
    'fever',            # 0=No, 1=Yes
    'night_sweats',     # 0=No, 1=Yes
    'weight_loss',      # 0=No, 1=Yes
    'fatigue',          # 0=No, 1=Yes
    'chest_pain',       # 0=No, 1=Yes
    'tb_contact',       # 0=No, 1=Yes
    'prev_tb',          # 0=No, 1=Yes
]

NUM_FEATURES = len(FEATURE_NAMES)  # 11


# ─────────────────────────────────────────
# Tabular Neural Network
# ─────────────────────────────────────────
class TabularModel(nn.Module):
    def __init__(self, input_dim=NUM_FEATURES,
                 hidden_dims=[64, 32],
                 output_dim=32,
                 dropout=0.3):
        super(TabularModel, self).__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers += [
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]
            prev_dim = hidden_dim

        # Final output layer — produces embedding
        layers += [
            nn.Linear(prev_dim, output_dim),
            nn.ReLU()
        ]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# ─────────────────────────────────────────
# Patient data → normalized tensor
# ─────────────────────────────────────────
def preprocess_patient_data(patient_data):
    """
    Convert raw patient form data to normalized tensor.

    patient_data: dict with keys matching FEATURE_NAMES
    Returns: torch.Tensor of shape (1, NUM_FEATURES)
    """
    features = []

    # Age: normalize to [0, 1] assuming max age 100
    age = float(patient_data.get('age', 30))
    features.append(min(age / 100.0, 1.0))

    # Sex: 0=Female, 1=Male
    sex = 1.0 if str(patient_data.get('sex', 'male')).lower() == 'male' else 0.0
    features.append(sex)

    # BMI: normalize to [0, 1] assuming range 10-50
    bmi = float(patient_data.get('bmi', 22))
    features.append(min(max((bmi - 10) / 40.0, 0.0), 1.0))

    # Cough weeks: normalize to [0, 1] assuming max 52 weeks
    cough_weeks = float(patient_data.get('cough_weeks', 0))
    features.append(min(cough_weeks / 52.0, 1.0))

    # Binary symptoms
    binary_fields = [
        'fever', 'night_sweats', 'weight_loss',
        'fatigue', 'chest_pain', 'tb_contact', 'prev_tb'
    ]
    for field in binary_fields:
        val = patient_data.get(field, False)
        if isinstance(val, bool):
            features.append(1.0 if val else 0.0)
        elif isinstance(val, str):
            features.append(1.0 if val.lower() in
                           ['true', 'yes', '1'] else 0.0)
        else:
            features.append(float(val))

    tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    return tensor


# ─────────────────────────────────────────
# Rule-based TB risk from patient data
# (used as fallback + training signal)
# ─────────────────────────────────────────
def calculate_clinical_risk(patient_data):
    """
    Calculate TB risk score from patient parameters.
    Returns score between 0 and 1.
    """
    score = 0.0
    reasons = []

    age = float(patient_data.get('age', 30))
    if age < 5 or age > 65:
        score += 0.10
        reasons.append("Age is a risk factor")

    if patient_data.get('tb_contact'):
        score += 0.25
        reasons.append("Known TB contact")

    if patient_data.get('prev_tb'):
        score += 0.20
        reasons.append("Previous TB history")

    cough_weeks = float(patient_data.get('cough_weeks', 0))
    if cough_weeks >= 3:
        score += 0.15
        reasons.append(f"Chronic cough ({cough_weeks} weeks)")

    symptom_count = sum([
        bool(patient_data.get('fever')),
        bool(patient_data.get('night_sweats')),
        bool(patient_data.get('weight_loss')),
        bool(patient_data.get('fatigue')),
        bool(patient_data.get('chest_pain')),
    ])

    if symptom_count >= 3:
        score += 0.20
        reasons.append(f"{symptom_count} TB symptoms present")
    elif symptom_count >= 1:
        score += 0.10
        reasons.append(f"{symptom_count} TB symptom(s) present")

    bmi = float(patient_data.get('bmi', 22))
    if bmi < 18.5:
        score += 0.10
        reasons.append("Low BMI (underweight)")

    return round(min(score, 1.0), 3), reasons


# ─────────────────────────────────────────
# Test the tabular model
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("🔍 Testing Tabular Model...\n")

    # Sample patient
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

    print("📋 Sample Patient:")
    for k, v in sample_patient.items():
        print(f"   {k:15s}: {v}")

    # Clinical risk
    risk_score, reasons = calculate_clinical_risk(sample_patient)
    print(f"\n📊 Clinical Risk Score: {risk_score}")
    print(f"📋 Risk Reasons:")
    for r in reasons:
        print(f"   • {r}")

    # Preprocess
    tensor = preprocess_patient_data(sample_patient)
    print(f"\n✅ Feature tensor shape: {tensor.shape}")
    print(f"   Values: {tensor.numpy()}")

    # Forward pass
    model = TabularModel()
    output = model(tensor)
    print(f"\n✅ Model output shape: {output.shape}")
    print(f"   Embedding: {output.detach().numpy()}")
    print(f"\n✅ Tabular Model is ready!")