# 🎯 Real-Time Object Detection with YOLO26

A GPU-accelerated, real-time object detection system using **Ultralytics YOLO26** and **Streamlit**. Designed for the NVIDIA RTX platform with CUDA support.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![YOLO26](https://img.shields.io/badge/YOLO-v26-blueviolet?logo=data:image/svg+xml;base64,)
![Streamlit](https://img.shields.io/badge/Streamlit-1.42%2B-ff4b4b?logo=streamlit)
![CUDA](https://img.shields.io/badge/CUDA-GPU%20Accelerated-76b900?logo=nvidia)

---

## ✨ Features

| Feature | Description |
|---|---|
| **YOLO26 Inference** | Latest end-to-end, NMS-free architecture for blazing-fast detection |
| **Model Toggle** | Switch between **Nano** (speed) and **Medium** (accuracy) variants |
| **Confidence Slider** | Adjust detection threshold in real-time from the sidebar |
| **FPS Counter** | Live frames-per-second overlay on the video feed |
| **GPU Acceleration** | Automatic CUDA detection and GPU inference |
| **Premium UI** | Dark-themed Streamlit interface with gradient accents and live stats |

---

## 🖥️ System Requirements

| Component | Minimum | Recommended |
|---|---|---|
| **GPU** | NVIDIA GTX 1060+ | RTX 3060+ / RTX 4060+ |
| **VRAM** | 4 GB | 8 GB |
| **CUDA** | 11.8+ | 12.x |
| **Python** | 3.10 | 3.12+ |
| **OS** | Windows 10 / Linux | Windows 11 / Ubuntu 22.04 |

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/prashplus/Real-Time-Object-Detection.git
cd Real-Time-Object-Detection
```

### 2. Create a Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install PyTorch with CUDA

Visit [pytorch.org](https://pytorch.org/get-started/locally/) to get the exact command for your CUDA version.  
For **CUDA 12.6** (common on RTX 40-series):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the App

```bash
streamlit run app.py
```

The app will open at **http://localhost:8501** in your browser.

---

## 🎮 Usage

1. **Start Detection** — Toggle the switch in the main area to begin webcam inference.
2. **Choose a Model** — Use the sidebar radio button to switch between:
   - 🟢 **Nano** (`yolo26n.pt`) — Fastest, ideal for real-time on any GPU
   - 🔵 **Medium** (`yolo26m.pt`) — Higher accuracy, slightly slower
3. **Adjust Confidence** — Drag the slider to filter out low-confidence detections.
4. **Monitor Performance** — Check the FPS counter on the video overlay and the stats panel.

---

## 📁 Project Structure

```
Real-Time-Object-Detection/
├── app.py               # Main Streamlit application
├── requirements.txt     # Python dependencies
├── README.md            # This file
└── venv/                # Virtual environment (created during setup)
```

---

## 🔧 Troubleshooting

<details>
<summary><b>CUDA not detected / running on CPU</b></summary>

Verify your PyTorch CUDA installation:

```python
import torch
print(torch.cuda.is_available())       # Should be True
print(torch.cuda.get_device_name(0))   # Should show your GPU
```

If `False`, reinstall PyTorch with the correct CUDA version from [pytorch.org](https://pytorch.org/get-started/locally/).
</details>

<details>
<summary><b>Webcam not opening</b></summary>

- Ensure no other app is using the camera.
- Try changing `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)` in `app.py` if you have multiple cameras.
- On Windows, grant camera permissions in **Settings → Privacy → Camera**.
</details>

<details>
<summary><b>Low FPS</b></summary>

- Switch to the **Nano** model for maximum speed.
- Increase the confidence threshold to reduce the number of rendered boxes.
- Close other GPU-intensive applications.
</details>

---

## 📄 License

This project is open-source under the [MIT License](LICENSE).

---

<p align="center">
  Built with ❤️ using <a href="https://ultralytics.com">Ultralytics YOLO26</a> and <a href="https://streamlit.io">Streamlit</a>
</p>
