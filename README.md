# 🛡️ AI-Based Enemy Movement Prediction & Threat Surveillance System

An intelligent defense surveillance system that detects, tracks, and predicts suspicious movement from drone/CCTV footage using computer vision and Transformer-based deep learning.

---

## 📌 Project Overview

Traditional surveillance systems only **record** activity. This system goes further — it **understands** and **predicts** movement behavior before threats occur.

Built for defense-oriented applications including:
- Border monitoring
- Smart surveillance
- Military intelligence
- Security analytics

---

## 🧠 How It Works

```
Video Input (CCTV / Drone)
        ↓
Object Detection — YOLOv8
  Detects people, cars, bikes, trucks in each frame
        ↓
Object Tracking — DeepSORT
  Assigns consistent IDs and builds movement history
        ↓
Trajectory Prediction — Transformer Model
  Takes last 8 positions → predicts next 12 future positions
        ↓
Threat Analysis (Phase 4)
  Flags loitering, restricted zone intrusions, erratic movement
        ↓
Dashboard + Alerts (Phase 5)
  Real-time visualization, heatmaps, alert logs
```

---

## 🗂️ Project Structure

```
defense_surveillance/
│
├── main.py                        # Pipeline runner — run this
├── requirements.txt               # All dependencies
├── README.md                      # You are here
├── .gitignore
│
├── configs/
│   └── config.yaml                # All settings (detection, tracking, model)
│
├── data/
│   └── raw/
│       ├── test.mp4               # Sample test video
│       └── eth_ucy/               # ETH/UCY trajectory dataset (download separately)
│
├── src/
│   ├── detection/
│   │   ├── __init__.py
│   │   └── detector.py            # YOLOv8 object detection
│   │
│   ├── tracking/
│   │   ├── __init__.py
│   │   └── tracker.py             # DeepSORT tracking + trail history
│   │
│   ├── prediction/
│   │   ├── __init__.py
│   │   ├── transformer_model.py   # Transformer architecture (353K parameters)
│   │   ├── trainer.py             # Training loop with ADE/FDE metrics
│   │   └── predictor.py           # Live inference on active tracks
│   │
│   └── utils/
│       ├── __init__.py
│       ├── logger.py              # Centralized logging
│       └── eth_ucy_loader.py      # ETH/UCY dataset loader
│
├── models/
│   └── predictor/
│       └── checkpoint_best.pt     # Saved after training (not in git)
│
└── outputs/
    ├── logs/                      # Runtime logs
    └── visualizations/            # Output videos + snapshots
```

---

## ⚙️ Setup

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/defense-surveillance-system.git
cd defense-surveillance-system
```

### 2. Create virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### Run detection + tracking on a video
```bash
python main.py --source data/raw/test.mp4
```

### Run and save output video
```bash
python main.py --source data/raw/test.mp4 --save
```

### Run on webcam
```bash
python main.py --source 0
```

### Run without prediction (faster)
```bash
python main.py --source data/raw/test.mp4 --no-predict --save
```

### Run headless (no display window)
```bash
python main.py --source data/raw/test.mp4 --no-display --save
```

**Keyboard shortcuts while running:**
- `q` — quit
- `s` — save snapshot of current frame

---

## 🤖 Training the Transformer Model

### Step 1 — Download ETH/UCY dataset
```
https://github.com/StanfordASL/Trajectron-plus-plus/tree/master/experiments/pedestrians/raw
```
Place .txt files in `data/raw/eth_ucy/`

### Step 2 — Train
```bash
python -m src.prediction.trainer --data data/raw/eth_ucy/
```

Training takes ~10 minutes on CPU, ~2 minutes on GPU.
Best model saved to `models/predictor/checkpoint_best.pt`

### Step 3 — Run with trained model
```bash
python main.py --source data/raw/test.mp4 --save
```

---

## 📊 Datasets Used

| Dataset | Purpose | Download |
|---|---|---|
| VisDrone | Fine-tune YOLOv8 for drone footage | github.com/VisDrone |
| ETH/UCY | Prototype trajectory model training | Trajectron++ repo |
| Stanford Drone (SDD) | Final trajectory model training | cvgl.stanford.edu |
| CUHK Avenue | Anomaly detection validation | CUHK dataset page |

---

## 🧩 Tech Stack

| Component | Technology |
|---|---|
| Object Detection | YOLOv8 (Ultralytics) |
| Object Tracking | DeepSORT |
| Trajectory Prediction | Transformer (PyTorch) |
| Video Processing | OpenCV |
| Dashboard | Streamlit (Phase 5) |
| Language | Python 3.11 |

---

## 📈 Model Architecture

```
Input: 8 observed (x, y) positions
          ↓
Linear Projection → 64 dimensions
          ↓
Positional Encoding (time awareness)
          ↓
Transformer Encoder (3 layers, 8 heads)
   learns movement patterns
          ↓
Transformer Decoder (3 layers, 8 heads)
   generates future positions
          ↓
Output: 12 predicted future (x, y) positions

Total parameters: 353,570
```

---

## 📋 Development Phases

| Phase | Description | Status |
|---|---|---|
| Phase 1 | Object Detection (YOLOv8) | ✅ Complete |
| Phase 2 | Object Tracking (DeepSORT) | ✅ Complete |
| Phase 3 | Transformer Prediction Model | ✅ Complete |
| Phase 4 | Threat Analysis + Alerts | 🔲 In Progress |
| Phase 5 | Dashboard + Visualization | 🔲 Upcoming |

---

## 📝 Results

After running on test.mp4 (430 frames):
- Total tracks detected: **38**
- Tracks ready for prediction (8+ points): **37**
- Average FPS on CPU: **~3.2**

---

## 👨‍💻 Author

Built as part of a defense AI internship project.
Demonstrates real-world application of computer vision, deep learning,
and Transformer architectures for intelligent surveillance systems.
