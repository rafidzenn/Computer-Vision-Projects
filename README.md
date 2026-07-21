# 👁️ Computer Vision & OCR Master Repository

Welcome to the central hub for my computer vision and Optical Character Recognition (OCR) projects. This repository serves as a portfolio containing various independent, production-ready mini-repositories. Each subfolder is a self-contained project with its own specific dependencies, datasets, and documentation.

---

## 🛠️ Global Technology Stack

While individual projects utilize unique specialized packages, the following core technologies are used across this repository:

- **Languages:** Python 3.10+, C++
- **Core Vision Libraries:** OpenCV, Pillow (PIL), Scikit-Image
- **OCR Engines:** Tesseract OCR, EasyOCR, Google Cloud Vision API, PaddleOCR
- **Deep Learning Frameworks:** PyTorch, TensorFlow, YOLO (v8/v11)
- **Deployment:** Docker, FastAPI, Streamlit

---

## 📁 Repository Directory & Projects

Below is an overview of the individual projects included in this repository. Click on a project name to jump directly to its subfolder and view its detailed `README.md`.

| Project Name | Technology Used | Key Features / Use Case | Status |
| :--- | :--- | :--- | :--- |
| **[📂 01-Invoice-Parser](./01-Invoice-Parser)** | Tesseract, Regex, OpenCV | Extracts structured billing data from scanned PDF invoices. | Done |
| **[📂 02-License-Plate-Reader](./02-License-Plate-Reader)** | YOLOv8, EasyOCR, PyTorch | Real-time vehicle plate detection and alphanumeric reading. | Done |
| **[📂 03-Handwritten-Text-Recognition](./03-Handwritten-Text-Recognition)** | TensorFlow, Keras, CNN-LSTM | Transcribes handwritten cursive notes into digital text. | In Progress |
| **[📂 04-Receipt-Scanner-API](./04-Receipt-Scanner-API)** | PaddleOCR, FastAPI, Docker | High-throughput cloud API for parsing retail store receipts. | Done |
| **[📂 05-Scene-Text-Spotter](./05-Scene-Text-Spotter)** | MMOCR, OpenCV | Detects and reads street signs and billboard text "in the wild." | Upcoming |

---

## 🚀 Getting Started

Because every project is independent, you do not need to install all dependencies at once. Follow these steps to run a specific project:

### 1. Clone the Master Repository
```bash
git clone https://github.com
cd your-cv-ocr-master-repo
```

### 2. Navigate to a Specific Project
Choose the mini-repo you want to run. For example:
```bash
cd 01-Invoice-Parser
```

### 3. Setup and Run
Every project folder contains its own setup instructions. Generally, you will create a virtual environment and install the localized `requirements.txt`:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```
*Note: Refer to the specific project's `README.md` for environmental variables, external weights download (e.g., YOLO `.pt` files), or system prerequisites like installing the Tesseract binary.*

---

## 🤝 How to Contribute

If you would like to add a new vision project to this master repo:
1. Create a new isolated subfolder using a clear, hyphenated naming convention (e.g., `06-Face-Blurrer`).
2. Ensure your project includes a standalone `requirements.txt` and an internal `README.md`.
3. Update the global directory table in this file with your project details.
4. Submit a Pull Request.

---

## 📄 License

This master repository is licensed under the MIT License. See individual subfolders for specific dataset licenses where applicable.
