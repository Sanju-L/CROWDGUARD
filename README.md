# 🛡️ CrowdGuard

**AI-based Real-Time Crowd Safety System**

CrowdGuard detects crowd density, analyses movement flow, flags abnormal
behaviour, computes an explainable risk score, and dynamically recommends
safer evacuation routes — all in real time through an operator dashboard.

---

## 🏗️ Project Status

| Phase | Module | Status |
|-------|--------|--------|
| 1 | Project scaffold & environment | ✅ Done |
| 2 | Person detection (YOLO) & tracking (ByteTrack) | ✅ Implemented |
| 3 | Density, flow, bottleneck & anomaly analytics | ✅ Implemented |
| 4 | Explainable risk scoring engine | ✅ Implemented |
| 5 | Dynamic evacuation routing (NetworkX) | ✅ Implemented |
| 6 | HTML/CSS/JavaScript operator dashboard | ✅ Implemented |

The dashboard is served as HTML, CSS, and JavaScript by the Flask backend. Flask
provides the API and video stream used by the Python AI pipeline; Streamlit is
not used.

Gate localization uses persistent doorway-shaped line geometry from the uploaded
footage. Repeated geometric candidates, confidence, orientation, and stable IDs
are available in `/api/state` as `gates`. The map overlay projects normalized image
coordinates onto the existing map viewBox to preserve relative camera-frame
position and orientation. This is a camera-relative 2D projection, not a
calibrated or reconstructed metric 3D floor plan; monocular video alone cannot
establish real-world dimensions or depth without camera calibration. The
confidence field describes structural geometry and repeated observations, not a
calibrated semantic probability; review candidate gates on unfamiliar camera
views before relying on them for safety decisions.

### Live multi-camera monitoring

The dashboard can connect multiple backend-accessible webcam/device indexes or
RTSP(S)/HTTP(S) streams. Add each source in the live camera panel and assign it
to its venue area; the system shows a separate analyzed feed and crowd metrics
for each camera. Per-camera direction suggestions use that assigned area's
crowd level and the configured venue route graph. The selected area is an
operator-provided camera-to-map association; image analysis cannot determine
which real-world venue zone a camera is installed in. Camera model inference is
shared and serialized across feeds, so achievable frame rate depends on camera
count, resolution, and hardware. Detection runs at a 512-pixel inference size
by default to improve CPU throughput while retaining the full-resolution
camera frames for display. Tune `DETECTION_IMAGE_SIZE` in `config.py` if the
scene requires more detection detail or the hardware supports a larger size.
Network stream URLs should only be configured for trusted cameras.

---

## 📂 Project Structure

```
CrowdGuard/
├── app.py                  # Flask entry-point
├── config.py               # Central configuration
├── requirements.txt
├── README.md
│
├── detection/              # YOLO person detection
│   └── detector.py
├── tracking/               # ByteTrack multi-object tracking
│   └── tracker.py
├── analytics/              # Crowd analytics modules
│   ├── density.py
│   ├── flow.py
│   ├── bottleneck.py
│   └── anomaly.py
├── risk/                   # Explainable risk scoring
│   └── risk_engine.py
├── routing/                # Evacuation route engine
│   └── route_engine.py
├── dashboard/              # HTML/CSS/JavaScript dashboard
│   ├── index.html
│   ├── dashboard.css
│   └── dashboard.js
│
├── data/
│   ├── videos/             # Sample crowd footage
│   └── maps/               # Venue graph definitions
├── models/                 # YOLO model weights
└── tests/                  # Test suite
```

---

## ⚡ Quick Start

```bash
# 1. Create & activate virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the dashboard
python app.py
```

---

## 🧰 Tech Stack

| Layer | Technology |
|-------|-----------|
| Detection | Ultralytics YOLO |
| Tracking | ByteTrack |
| Vision | OpenCV |
| Analytics | NumPy, Pandas |
| Routing | NetworkX |
| Frontend | HTML, CSS, JavaScript |
| API / AI backend | Flask, Python, Ultralytics YOLO, ByteTrack |

---

## 📄 License

This project is for educational / prototype purposes.
