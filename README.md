# 🫁 Lung3D AI — Multimodal TB Detection System

![Status](https://img.shields.io/badge/status-live-brightgreen)
![Python](https://img.shields.io/badge/python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2-red)
![Accuracy](https://img.shields.io/badge/accuracy-85%25-green)

> AI-powered Tuberculosis detection using chest X-rays + patient clinical data

🌐 **Live Demo:** [lung3d-ai.up.railway.app](https://lung3d-ai.up.railway.app)

---

## 🧠 What It Does

Lung3D AI is a multimodal medical AI system that:
- Analyzes chest X-ray images using a trained ResNet18 CNN
- Combines image analysis with 11 patient clinical parameters
- Generates Grad-CAM heatmaps showing suspicious regions
- Outputs TB / No TB prediction with confidence score
- Downloads a professional PDF clinical report

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Overall Accuracy | 85.12% |
| TB Sensitivity | 81.67% |
| Normal Specificity | 88.52% |
| Training Images | 800 |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python + Flask |
| Deep Learning | PyTorch + ResNet18 |
| Visualization | Three.js + Canvas API |
| Explainability | Grad-CAM |
| PDF Reports | ReportLab |
| Deployment | Railway |

---

## 🚀 Run Locally

```bash
# Clone the repo
git clone https://github.com/murtuja43/lung3d-ai.git
cd lung3d-ai

# Create environment
conda create -n tb-ai python=3.11 -y
conda activate tb-ai

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

---

## 📁 Project Structure


---

## ⚠️ Disclaimer

This is a demonstration system built for educational purposes.
Not intended for clinical use.
Always consult a licensed medical professional for diagnosis.