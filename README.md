🧠 Brain Tumor Detection — README
🧠 BRAIN TUMOR DETECTION
AI-Powered MRI Analysis · ResNet50 · PyTorch · Streamlit

📌 Project Overview
Brain Tumor Detection is an AI-based web application built using Python, PyTorch (ResNet50), and Streamlit. It analyzes brain MRI scans and predicts whether a tumor is present or not with confidence scores, graphs, and downloadable reports.

✨ Features
•	ResNet50 pretrained model (ImageNet weights)
•	Upload MRI scan (JPG, PNG, BMP)
•	Confidence gauge chart (Plotly)
•	Class probability bar chart
•	Pixel intensity distribution graph
•	Tumor risk meter bar
•	Health & dietary recommendations
•	Downloadable .txt report
•	Contrast enhancement toggle
•	RGB channel viewer
•	Fast inference — model cached after first load

🛠️ Tech Stack
Library	Version	Purpose
streamlit	>=1.32.0	Web UI framework
torch	>=2.0.0	Deep learning engine
torchvision	>=0.15.0	ResNet50 + transforms
Pillow	>=10.0.0	Image loading & processing
numpy	>=1.24.0	Array operations
plotly	>=5.18.0	Interactive charts & graphs

📁 Folder Structure
D:\Deep Learning\
  ├── myenv\               (virtual environment)
  ├── app_short.py         (main Streamlit app)
  └── requirements.txt     (dependencies)

🚀 Installation & Setup
Step 1 — Open CMD
cd D:\Deep Learning
Step 2 — Create Virtual Environment
python -m venv myenv
Step 3 — Activate Environment
myenv\Scripts\activate
Step 4 — Install Libraries
pip install streamlit torch torchvision Pillow numpy plotly
Step 5 — Run the App
python -m streamlit run app_short.py
Then open your browser at: http://localhost:8501

📖 How to Use
•	Open the app in browser at http://localhost:8501
•	Upload a brain MRI scan (JPG / PNG / BMP)
•	Optionally adjust confidence threshold in the sidebar
•	Click the ANALYZE SCAN button
•	View result, charts, risk meter, and health tips
•	Click Download Report to save the analysis as .txt

🔬 Model Details
Property	Value
Architecture	ResNet50
Pretrained Weights	ImageNet (IMAGENET1K_V1)
Input Size	224 x 224 px
Classes	2 (No Tumor / Tumor Detected)
Framework	PyTorch + torchvision
Inference Speed	< 500ms (after first load)

⚠️ Common Errors & Fixes
Error	Fix
streamlit not recognized	Use: python -m streamlit run app_short.py
No module named torch	Run: pip install torch torchvision
Port 8501 in use	Use: --server.port 8502
Slow first load	Normal — model downloads ~100MB once only


📋 Disclaimer
This tool is built for EDUCATIONAL and RESEARCH purposes only.
It is NOT a substitute for professional medical diagnosis. Always consult a qualified radiologist or neurologist for any medical concerns.
