import numpy as np
from data.mock_ct import generate_ct_scan, compute_abnormality_score


def analyze_scan(seed=42):
    """
    Full pipeline:
    1. Generate mock CT scan
    2. Analyze abnormality
    3. Return prediction, confidence, explanation
    """
    volume, abnormal_mask, nodules = generate_ct_scan(seed=seed)

    # --- Feature extraction ---
    abnormality_score = compute_abnormality_score(volume, abnormal_mask)
    nodule_count      = len(nodules)
    max_nodule_size   = max([r for _, _, _, r in nodules]) if nodules else 0
    avg_intensity     = float(np.mean(volume[volume > 0.15]))

    # --- Rule-based scoring ---
    score = 0.0

    # Rule 1: High abnormality ratio
    if abnormality_score > 0.05:
        score += 0.40
    elif abnormality_score > 0.02:
        score += 0.20

    # Rule 2: Multiple nodules found
    if nodule_count >= 2:
        score += 0.25
    elif nodule_count == 1:
        score += 0.10

    # Rule 3: Large nodule size
    if max_nodule_size >= 4:
        score += 0.20
    elif max_nodule_size >= 2:
        score += 0.10

    # Rule 4: Elevated average tissue intensity
    if avg_intensity > 0.30:
        score += 0.15
    elif avg_intensity > 0.25:
        score += 0.07

    # Clamp score to [0, 1]
    confidence = round(min(score, 1.0), 2)

    # --- Prediction ---
    prediction = "TB Detected" if confidence >= 0.4 else "No TB Detected"

    # --- Explanation ---
    explanation = build_explanation(
        prediction,
        abnormality_score,
        nodule_count,
        max_nodule_size,
        avg_intensity,
        confidence
    )

    return {
        "prediction":        prediction,
        "confidence":        confidence,
        "abnormality_score": abnormality_score,
        "nodule_count":      nodule_count,
        "max_nodule_size":   max_nodule_size,
        "avg_intensity":     round(avg_intensity, 4),
        "explanation":       explanation
    }


def build_explanation(prediction, abnormality_score, nodule_count,
                      max_nodule_size, avg_intensity, confidence):
    """
    Build a human-readable explanation for the prediction.
    """
    lines = []

    lines.append(f"🔬 Prediction: {prediction} (Confidence: {int(confidence*100)}%)")
    lines.append("")

    lines.append("📊 Scan Analysis:")
    lines.append(f"  • Abnormal tissue ratio : {round(abnormality_score * 100, 2)}% of lung volume")
    lines.append(f"  • Suspicious nodules    : {nodule_count} detected")
    lines.append(f"  • Largest nodule size   : {max_nodule_size} units")
    lines.append(f"  • Avg tissue intensity  : {round(avg_intensity, 4)}")
    lines.append("")

    lines.append("📋 Reasoning:")

    if abnormality_score > 0.05:
        lines.append("  ✅ High proportion of abnormal tissue detected.")
    elif abnormality_score > 0.02:
        lines.append("  ⚠️  Moderate abnormal tissue detected.")
    else:
        lines.append("  ✅ Low abnormal tissue — within normal range.")

    if nodule_count >= 2:
        lines.append("  ✅ Multiple nodules found — consistent with TB patterns.")
    elif nodule_count == 1:
        lines.append("  ⚠️  Single nodule found — further investigation advised.")
    else:
        lines.append("  ✅ No nodules detected.")

    if max_nodule_size >= 4:
        lines.append("  ✅ Large nodule size detected — clinically significant.")
    elif max_nodule_size >= 2:
        lines.append("  ⚠️  Small nodule detected — monitoring recommended.")

    if avg_intensity > 0.30:
        lines.append("  ✅ Elevated tissue density — suggests active infection.")
    else:
        lines.append("  ✅ Tissue density within acceptable range.")

    lines.append("")
    lines.append("⚠️  Disclaimer: This is a simulated result for demo purposes only.")
    lines.append("   Always consult a licensed medical professional for diagnosis.")

    return "\n".join(lines)


if __name__ == "__main__":
    result = analyze_scan()
    print(result["explanation"])
    print("\nRaw result:", result)