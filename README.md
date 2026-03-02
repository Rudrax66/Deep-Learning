# 🧠 Brain Tumor Detection — AI-Powered MRI Analysis

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-5.18+-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**An AI-powered web application that detects brain tumors from MRI scans using a pretrained ResNet50 deep learning model.**

[Features](#-features) · [Demo](#-demo) · [Installation](#-installation) · [Usage](#-usage) · [Model](#-model-details) · [Tech Stack](#-tech-stack)

</div>

---
Live Demo : https://deep-learning-vgvux97mkidnjrbenaosrr.streamlit.app/

## 📌 Overview

Brain Tumor Detection is a **Streamlit** web app powered by **PyTorch ResNet50** that:

- Accepts brain MRI scans (JPG, PNG, BMP)
- Predicts **Tumor Detected** or **No Tumor** with confidence score
- Displays interactive **Plotly charts** (gauge, bar, pixel intensity)
- Generates a **downloadable diagnosis report**
- Provides **health & dietary recommendations** based on result

> ⚠️ **Disclaimer:** This tool is for **educational and research purposes only**. It is NOT a substitute for professional medical diagnosis.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🧠 AI Diagnosis | ResNet50 detects tumor with confidence % |
| 📊 Confidence Gauge | Visual gauge chart showing prediction strength |
| 📈 Bar Chart | Class probability comparison (Plotly) |
| 🔬 Pixel Intensity | MRI scan brightness distribution graph |
| 🎯 Risk Meter | Color-coded tumor probability bar |
| 🥗 Health Tips | Dietary & medical recommendations |
| 📄 Report Export | Download full diagnosis as `.txt` file |
| ⚡ Fast Inference | Model cached after first load — instant results |
| 🖼️ RGB Channels | Optional R/G/B channel viewer |
| 🎛️ Threshold Control | Adjustable confidence threshold via sidebar |

---

## 🖥️ Demo

```
Upload MRI  →  Preprocess  →  ResNet50  →  Results + Charts  →  Report
   📤             🔧             🤖            📊                 📄
```

**Result looks like:**

```
⚠️  TUMOR DETECTED
Confidence: 87.4%
Risk: HIGH

✅  NO TUMOR FOUND  
Confidence: 94.1%
Risk: LOW
```

---

## 🚀 Installation

### Step 1 — Clone the Repository

```bash
git clone https://github.com/yourusername/brain-tumor-detection.git
cd brain-tumor-detection
```

### Step 2 — Create Virtual Environment

```bash
python -m venv myenv
```

### Step 3 — Activate Virtual Environment

**Windows:**
```bash
myenv\Scripts\activate
```

**Mac / Linux:**
```bash
source myenv/bin/activate
```

### Step 4 — Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install streamlit torch torchvision Pillow numpy plotly
```

### Step 5 — Run the App

```bash
python -m streamlit run app_short.py
```

Open browser at → **http://localhost:8501**

---

## 📋 Requirements

```txt
streamlit>=1.32.0
torch>=2.0.0
torchvision>=0.15.0
Pillow>=10.0.0
numpy>=1.24.0
plotly>=5.18.0
```

> **Python 3.8 or higher required**

---

## 📁 Project Structure

```
brain-tumor-detection/
│
├── app_short.py          # Main Streamlit application
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

---

## 🔬 Model Details

| Property | Value |
|---|---|
| Architecture | ResNet50 |
| Pretrained Weights | ImageNet (IMAGENET1K_V1) |
| Custom Head | Dropout(0.3) → Linear(2048 → 2) |
| Input Size | 224 × 224 px |
| Classes | 2 — `No Tumor` / `Tumor Detected` |
| Framework | PyTorch + torchvision |
| Inference Time | < 500ms (after first load) |
| Parameters | ~50 Million |

### Preprocessing Pipeline

```
Raw MRI Image
    ↓
Resize to 224×224
    ↓
Convert to 3-channel Grayscale
    ↓
ToTensor()
    ↓
Normalize (ImageNet mean/std)
    ↓
ResNet50 → Softmax → [No Tumor, Tumor]
```

---

## 🛠️ Tech Stack

| Library | Version | Purpose |
|---|---|---|
| `Python` | 3.8+ | Core language |
| `streamlit` | ≥1.32.0 | Web UI framework |
| `torch` | ≥2.0.0 | Deep learning engine |
| `torchvision` | ≥0.15.0 | ResNet50 + image transforms |
| `Pillow` | ≥10.0.0 | Image loading & processing |
| `numpy` | ≥1.24.0 | Numerical operations |
| `plotly` | ≥5.18.0 | Interactive charts & graphs |

---

## 📖 How to Use

1. **Open** the app at `http://localhost:8501`
2. **Upload** a brain MRI scan (JPG / PNG / BMP)
3. **Adjust** confidence threshold in the sidebar (default: 70%)
4. **Click** `🔬 ANALYZE SCAN` button
5. **View** result, charts, risk meter & health tips
6. **Download** the report using the download button

---

## ⚙️ Sidebar Options

| Option | Description |
|---|---|
| Confidence Threshold | Min % to flag as tumor (50–95%) |
| Enhance Contrast | Applies contrast enhancement to MRI |
| Show RGB Channels | Displays R, G, B channels separately |

---

## ⚠️ Common Errors & Fixes

| Error | Fix |
|---|---|
| `streamlit not recognized` | Use `python -m streamlit run app_short.py` |
| `No module named torch` | Run `pip install torch torchvision` |
| `Port 8501 already in use` | Use `--server.port 8502` flag |
| `Slow on first load` | Normal — ResNet50 weights download once (~100MB) |
| `myenv\Scripts\activate` fails | Run `Set-ExecutionPolicy Unrestricted` in PowerShell |

---

## 🏃 GPU Support (Optional)

For faster inference with NVIDIA GPU:

```bash
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Check GPU availability:
```python
import torch
print(torch.cuda.is_available())  # True = GPU ready
```

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first.

1. Fork the repo
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 📢 Disclaimer

> This AI tool is built for **educational and research purposes only**.
> It is **NOT** a substitute for professional medical diagnosis.
> Always consult a qualified **radiologist or neurologist** for any medical concerns.

---

<div align="center">

Made with ❤️ using Python · PyTorch · Streamlit

⭐ **Star this repo if you found it helpful!**

</div>
