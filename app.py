# ──────────────────────────────────────────────
# CrowdGuard – Flask Backend + AI Pipeline
# ──────────────────────────────────────────────

"""
Main entry point for the CrowdGuard dashboard.

Run:
    cd CrowdGuard
    .venv\\Scripts\\python app.py
"""

from __future__ import annotations

import atexit
import math
import os
import tempfile
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify, request

# ── CrowdGuard modules ──────────────────────
from config import BASE_DIR, VIDEO_DIR, MAP_DIR
from tracking.tracker import MultiObjectTracker
from analytics.density import DensityAnalyser
from analytics.flow import FlowAnalyser
from analytics.anomaly import CounterFlowDetector
from analytics.bottleneck import BottleneckDetector, assign_zone
from analytics.cpwp import CrowdPressureWavePredictor
from risk.risk_engine import RiskEngine
from routing.route_engine import RouteEngine
from alerts.alert_manager import AlertManager, AlertType, AlertSeverity
from detection.gate_detector import GateDetector
from tracking.camera_manager import CameraManager

app = Flask(__name__, static_folder='dashboard', static_url_path='')

# ── CORS ─────────────────────────────────────
@app.after_request
def add_cors(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    return response

# ── Timestamp Utility ────────────────────────

def ts_now() -> str:
    return datetime.now().strftime("%H:%M:%S")

# ── Global State ───────────────────────────────
pipeline = {
    'running': False,
    'cap': None,
    'tracker': None,
    'gate_detector': None,
    'density': None,
    'flow': None,
    'cf_detector': None,
    'bn_detector': None,
    'cpwp': None,
    'risk': None,
    'route': None,
    'alert_manager': AlertManager(),
    'session_start': None,
    'session_duration': 0,
    'peak_people': 0,
    'peak_density': 0.0,
    'peak_risk': 0.0,
    'total_alerts': 0,
    'cf_events': 0,
    'bn_events': 0,
    'route_changes': 0,
    'timeline': [],
    'latest_frame_encoded': None,
    'fps': 24.0,
}

app_state = {
    'metrics': {
        'people': 0,
        'density': 0,
        'speed': 0,
        'risk_level': 'SAFE',
        'risk_score': 0,
        'dominant_flow': 'STATIONARY',
        'cf_flag': False
    },
    'zones': [],
    'gates': [],
    'exits': [],
    'tracks': [],
    'person_flows': [],
    'route': None,
    'alerts': [],
    'timeline': [],
    'summary': {},
    'frame_info': {
        'frame_idx': 0,
        'fps': 0,
        'time': '00:00',
        'total_time': '00:00'
    }
}

camera_manager = CameraManager()
atexit.register(camera_manager.close)

# ── Event Recording ──────────────────────────

def add_timeline_event(event_type, zone="", detail="", risk_score=0):
    ev = {
        "timestamp": ts_now(),
        "event_type": event_type,
        "zone": zone,
        "detail": detail,
        "risk_score": risk_score
    }
    pipeline['timeline'].insert(0, ev)
    if len(pipeline['timeline']) > 100:
        pipeline['timeline'] = pipeline['timeline'][:100]


# ── Pipeline Initialisation ──────────────────

def initialize_pipeline(source):
    """Initialize all AI modules and open the video source."""
    # Release old capture if any
    if pipeline['cap'] is not None:
        try:
            pipeline['cap'].release()
        except Exception:
            pass

    # Open video source
    if source == 'camera':
        pipeline['cap'] = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    else:
        video_path = str(VIDEO_DIR / source)
        print(f"[INIT] Opening video: {video_path}")
        pipeline['cap'] = cv2.VideoCapture(video_path)

    if not pipeline['cap'].isOpened():
        print(f"[INIT] ERROR: Could not open video source: {source}")
        return False

    width = int(pipeline['cap'].get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(pipeline['cap'].get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = pipeline['cap'].get(cv2.CAP_PROP_FPS) or 24.0

    if width == 0 or height == 0:
        width, height = 1280, 720

    print(f"[INIT] Video opened: {width}x{height} @ {fps:.1f} FPS")

    frame_shape = (height, width)
    pipeline['fps'] = fps

    # Initialize AI modules
    # MultiObjectTracker takes no fps arg
    pipeline['tracker'] = MultiObjectTracker()
    pipeline['gate_detector'] = GateDetector()

    # DensityAnalyser takes optional zones/thresholds dicts, NOT frame_shape
    pipeline['density'] = DensityAnalyser()

    # FlowAnalyser takes fps
    pipeline['flow'] = FlowAnalyser(fps=fps)

    # CounterFlowDetector takes no required args
    pipeline['cf_detector'] = CounterFlowDetector()

    # BottleneckDetector takes no required args  
    pipeline['bn_detector'] = BottleneckDetector()

    # Pressure Wave Predictor
    pipeline['cpwp'] = CrowdPressureWavePredictor(fps=fps)

    # RiskEngine takes no required args
    pipeline['risk'] = RiskEngine()

    # RouteEngine takes a map filename (relative to MAP_DIR), NOT a full path
    # Default is "venue_map.json" inside data/maps/
    pipeline['route'] = RouteEngine()
    pipeline['route'].load_graph()  # loads venue_map.json

    pipeline['alert_manager'] = AlertManager()

    app_state['gates'] = []
    app_state['exits'] = []
    pipeline['session_start'] = time.time()
    pipeline['timeline'] = []

    # Reset stats
    pipeline['peak_people'] = 0
    pipeline['peak_density'] = 0.0
    pipeline['peak_risk'] = 0.0
    pipeline['total_alerts'] = 0
    pipeline['cf_events'] = 0
    pipeline['bn_events'] = 0
    pipeline['route_changes'] = 0

    add_timeline_event("SESSION STARTED", detail=f"Source: {source}")
    print("[INIT] Pipeline ready.")
    return True


# ── AI Processing Loop ──────────────────────

def ai_loop():
    frame_idx = 0
    last_fps_time = time.time()
    frames_since_fps = 0
    current_fps = 0
    last_route_path = []

    while True:
        try:
            if not pipeline['running'] or pipeline['cap'] is None:
                time.sleep(0.1)
                continue

            ret, frame = pipeline['cap'].read()
            if not ret:
                # Loop video files back to start
                pipeline['cap'].set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = pipeline['cap'].read()
                if not ret:
                    time.sleep(0.1)
                    continue

            h, w = frame.shape[:2]

            # Ensure every frame is annotated to avoid bounding box flickering

            # 1. Tracking & Gate Detection
            tracker_state = pipeline['tracker'].update(frame, frame_number=frame_idx)
            pipeline['gate_detector'].update(frame, tracker_state.tracks)

            # 2. Density – expects list of (cx, cy) and (height, width)
            centroids = [(t.center_x, t.center_y) for t in tracker_state.tracks]
            zone_reports = pipeline['density'].compute(centroids, (h, w))

            # 3. Flow
            person_flows = pipeline['flow'].compute_person_flows(tracker_state.history)
            zone_flows = pipeline['flow'].compute_zone_flows(person_flows, (h, w))

            # 4. Counter-flow – expects list of ZoneFlow objects
            cf_reports = pipeline['cf_detector'].analyse(zone_flows)

            # 5. Bottleneck – update() expects {track_id: zone_name}, frame_number
            track_zones = {}
            for t in tracker_state.tracks:
                z = assign_zone(t.center_x, t.center_y, (h, w))
                track_zones[t.track_id] = z
            pipeline['bn_detector'].update(track_zones, frame_idx)
            gate_reports = pipeline['bn_detector'].get_reports()

            # 6. Risk – evaluate() expects zone_reports, cf_reports, gate_reports, zone_flows
            risk_report = pipeline['risk'].evaluate(
                zone_reports=zone_reports,
                counter_flow_reports=cf_reports,
                gate_reports=gate_reports,
                zone_flows=zone_flows
            )

            # 7. Route
            # Update risk state from zone reports
            risk_state = {}
            for zr in zone_reports:
                # Map zone name (e.g. "Zone A") to graph node ID (e.g. "ZONE_A")
                node_id = zr.name.upper().replace(" ", "_")
                risk_state[node_id] = zr.status
            pipeline['route'].update_risk_state(risk_state)

            route_result = pipeline['route'].find_safest_route("ZONE_C")

            # 8. Pressure Wave Prediction
            wave_predictions = pipeline['cpwp'].predict(zone_reports, zone_flows)

            # 9. Update FPS counter
            now = time.time()
            frames_since_fps += 1
            if now - last_fps_time >= 1.0:
                current_fps = frames_since_fps
                frames_since_fps = 0
                last_fps_time = now

            # Stats
            ppl = len(tracker_state.tracks)
            # Compute global density as average of zone densities
            den = sum(zr.density_pct for zr in zone_reports) / max(1, len(zone_reports))
            if ppl > pipeline['peak_people']:
                pipeline['peak_people'] = ppl
            if den > pipeline['peak_density']:
                pipeline['peak_density'] = den
            if risk_report.risk_score > pipeline['peak_risk']:
                pipeline['peak_risk'] = risk_report.risk_score

            # Timeline events
            for cf in cf_reports:
                if cf.counter_flow and cf.opposing_pct > 40:
                    pipeline['cf_events'] += 1
                    add_timeline_event(
                        "COUNTER-FLOW",
                        zone=cf.zone_name,
                        detail=f"Opposing: {cf.opposing_pct:.0f}%"
                    )

            for gr in gate_reports:
                if gr.bottleneck:
                    pipeline['bn_events'] += 1
                    add_timeline_event(
                        "BOTTLENECK",
                        zone=gr.gate_name,
                        detail=f"Accum rate: {gr.accumulation_rate:.1f}/min"
                    )

            for wp in wave_predictions:
                # Add a timeline event for new warnings (avoid spamming every frame by only logging if we are not logging a lot)
                # We'll just log CRITICAL ones for timeline
                if wp.severity == 'CRITICAL' and frame_idx % int(pipeline['fps'] * 5) == 0:
                    add_timeline_event(
                        "PRESSURE WAVE",
                        zone=wp.target_zone,
                        detail=f"From {wp.source_zone} in ~{int(wp.time_to_impact_sec)}s"
                    )

            if route_result.route_nodes != last_route_path:
                if last_route_path:  # don't record the initial route
                    pipeline['route_changes'] += 1
                    add_timeline_event(
                        "ROUTE CHANGED",
                        detail=f"New: {' → '.join(route_result.route_nodes)}"
                    )
                last_route_path = route_result.route_nodes

            # Build JS-compatible state
            js_tracks = []
            for t in tracker_state.tracks:
                js_tracks.append({
                    'id': t.track_id,
                    'bbox': t.bbox.tolist(),
                    'cx': t.center_x,
                    'cy': t.center_y
                })

            js_person_flows = []
            for pf in person_flows:
                js_person_flows.append({
                    'id': pf.track_id,
                    'dx': pf.dx,
                    'dy': pf.dy,
                    'speed': pf.speed,
                    'direction': pf.direction,
                    'cx': pf.center[0],
                    'cy': pf.center[1]
                })

            # Dominant flow direction
            if person_flows:
                dirs = [pf.direction for pf in person_flows if pf.direction != 'STATIONARY']
                if dirs:
                    dom_flow = max(set(dirs), key=dirs.count)
                else:
                    dom_flow = 'STATIONARY'
            else:
                dom_flow = 'STATIONARY'

            app_state['metrics'] = {
                'people': ppl,
                'density': round(den, 1),
                'speed': round(sum(pf.speed for pf in person_flows) / max(1, len(person_flows)), 1),
                'risk_level': risk_report.risk_level,
                'risk_score': round(risk_report.risk_score, 1),
                'dominant_flow': dom_flow,
                'cf_flag': any(cf.counter_flow for cf in cf_reports)
            }

            app_state['zones'] = [{
                'name': zr.name,
                'density': zr.density_pct,
                'status': zr.status,
                'bounds': zr.bounds_px
            } for zr in zone_reports]

            app_state['tracks'] = js_tracks
            app_state['person_flows'] = js_person_flows

            # Gate coordinates stay in the source camera's normalized image space.
            gates = pipeline['gate_detector'].to_api_dict(w, h)

            app_state['route'] = {
                'path': route_result.route_nodes,
                'cost': route_result.total_cost,
                'exit': route_result.recommended_exit,
                'status': route_result.explanation[:60] if route_result.explanation else ''
            }
            app_state['gates'] = gates
            app_state['exits'] = gates

            js_waves = []
            for wp in wave_predictions:
                js_waves.append({
                    'source': wp.source_zone,
                    'target': wp.target_zone,
                    'time': wp.time_to_impact_sec,
                    'msg': wp.warning_message,
                    'severity': wp.severity,
                    'recs': wp.recommendations
                })
            app_state['pressure_waves'] = js_waves

            app_state['timeline'] = pipeline['timeline']

            sess_dur = now - pipeline['session_start'] if pipeline['session_start'] else 0
            app_state['summary'] = {
                'duration': f"{int(sess_dur // 60):02d}:{int(sess_dur % 60):02d}",
                'peak_people': pipeline['peak_people'],
                'peak_density': round(pipeline['peak_density'], 1),
                'peak_risk': round(pipeline['peak_risk'], 1),
                'cf_events': pipeline['cf_events'],
                'bn_events': pipeline['bn_events'],
                'route_changes': pipeline['route_changes']
            }

            # Video time info
            total_frames = int(pipeline['cap'].get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            vid_fps = pipeline['fps']
            cur_time_s = frame_idx / max(vid_fps, 1)
            total_time_s = total_frames / max(vid_fps, 1)

            app_state['frame_info'] = {
                'frame_idx': frame_idx,
                'fps': current_fps,
                'time': f"{int(cur_time_s // 60):02d}:{int(cur_time_s % 60):02d}",
                'total_time': f"{int(total_time_s // 60):02d}:{int(total_time_s % 60):02d}",
                'progress': min(100, int(frame_idx / max(total_frames, 1) * 100)),
                'width': w,
                'height': h
            }

            # Draw tracking boxes on the frame before encoding
            # Draw gates first (underneath tracking boxes)
            annotated = pipeline['gate_detector'].draw(frame)
            flow_by_id = {pf.track_id: pf for pf in person_flows}
            
            for t in tracker_state.tracks:
                x1, y1, x2, y2 = t.bbox.astype(int)
                
                # Box color: cyan for normal, red for anomaly
                color = (255, 212, 0)  # BGR cyan
                
                # Draw bounding box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                
                # Draw corner accents
                cl = min(15, (x2 - x1) // 4)
                cv2.line(annotated, (x1, y1), (x1 + cl, y1), color, 3)
                cv2.line(annotated, (x1, y1), (x1, y1 + cl), color, 3)
                cv2.line(annotated, (x2, y1), (x2 - cl, y1), color, 3)
                cv2.line(annotated, (x2, y1), (x2, y1 + cl), color, 3)
                cv2.line(annotated, (x1, y2), (x1 + cl, y2), color, 3)
                cv2.line(annotated, (x1, y2), (x1, y2 - cl), color, 3)
                cv2.line(annotated, (x2, y2), (x2 - cl, y2), color, 3)
                cv2.line(annotated, (x2, y2), (x2, y2 - cl), color, 3)
                
                # Draw center dot
                cx_pt = int(t.center_x)
                cy_pt = int(t.center_y)
                cv2.circle(annotated, (cx_pt, cy_pt), 3, (0, 0, 255), -1)
                
                # Draw ID label
                label = f"ID-{t.track_id}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 6, y1), (0, 0, 0), -1)
                cv2.putText(annotated, label, (x1 + 3, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

                # Mark each tracked person's current movement direction. A
                # short vector keeps arrows readable even in dense scenes.
                flow = flow_by_id.get(t.track_id)
                if flow is not None and flow.speed > 0:
                    vector_length = max(24, min(70, int(flow.speed * 2.5)))
                    magnitude = max(math.hypot(flow.dx, flow.dy), 1e-6)
                    end = (
                        cx_pt + int(flow.dx / magnitude * vector_length),
                        cy_pt + int(flow.dy / magnitude * vector_length),
                    )
                    direction_color = (0, 230, 255) if flow.direction != 'STATIONARY' else (180, 180, 180)
                    cv2.arrowedLine(
                        annotated, (cx_pt, cy_pt), end,
                        direction_color, 2, cv2.LINE_AA, tipLength=0.3,
                    )
                    cv2.putText(
                        annotated, flow.direction, (end[0] + 3, end[1] - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, direction_color, 1, cv2.LINE_AA,
                    )
            
            # Draw people count overlay
            count_text = f"TRACKED: {len(tracker_state.tracks)}"
            cv2.rectangle(annotated, (8, 8), (200, 36), (0, 0, 0), -1)
            cv2.putText(annotated, count_text, (14, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 212, 0), 1, cv2.LINE_AA)

            # Encode annotated frame as JPEG for streaming
            ret_jpg, jpeg = cv2.imencode('.jpg', annotated)
            if ret_jpg:
                pipeline['latest_frame_encoded'] = jpeg.tobytes()

            frame_idx += 1

        except Exception as e:
            print(f"[AI LOOP ERROR] {e}")
            import traceback
            traceback.print_exc()
            time.sleep(1)


threading.Thread(target=ai_loop, daemon=True).start()


# ── Flask Routes ─────────────────────────────

@app.route('/')
def index():
    return app.send_static_file('index.html')

def gen_frames():
    while True:
        if pipeline['latest_frame_encoded']:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + pipeline['latest_frame_encoded'] + b'\r\n')
        time.sleep(0.016)  # ~60fps max

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/state')
def get_state():
    return jsonify(app_state)


@app.route('/api/health')
def health():
    return jsonify({
        'status': 'ok',
        'ai_running': pipeline['running'] or camera_manager.is_running()
    })


@app.route('/api/cameras', methods=['GET', 'POST'])
def cameras():
    if request.method == 'GET':
        return jsonify(camera_manager.dashboard_state())

    data = request.get_json(silent=True) or {}
    source_type = data.get('source_type')
    raw_source = data.get('source')
    try:
        if source_type == 'device':
            source = int(raw_source)
            if source < 0:
                raise ValueError("Camera device index must be zero or greater.")
        elif source_type == 'stream':
            source = str(raw_source or '').strip()
            from urllib.parse import urlparse

            parsed = urlparse(source)
            if parsed.scheme not in {'rtsp', 'rtsps', 'http', 'https'} or not parsed.netloc:
                raise ValueError("Enter a valid RTSP, RTSPS, HTTP, or HTTPS camera stream URL.")
        else:
            raise ValueError("Select a camera device or network stream.")

        camera = camera_manager.add_camera(
            name=str(data.get('name') or ''),
            source=source,
            area=str(data.get('area') or ''),
        )
        return jsonify({'status': 'ok', 'camera': camera}), 201
    except (TypeError, ValueError) as error:
        return jsonify({'status': 'error', 'msg': str(error)}), 400
    except Exception as error:
        app.logger.exception("Could not initialize live camera")
        return jsonify({'status': 'error', 'msg': f'Could not initialize camera: {error}'}), 500


@app.route('/api/cameras/<camera_id>', methods=['DELETE'])
def remove_camera(camera_id):
    if not camera_manager.remove_camera(camera_id):
        return jsonify({'status': 'error', 'msg': 'Camera not found.'}), 404
    return jsonify({'status': 'ok'})


@app.route('/api/cameras/<camera_id>/feed')
def camera_feed(camera_id):
    if not any(camera['id'] == camera_id for camera in camera_manager.list_cameras()):
        return jsonify({'status': 'error', 'msg': 'Camera not found.'}), 404

    def generate_camera_frames():
        while True:
            frame = camera_manager.get_frame(camera_id)
            if frame is None:
                if not any(camera['id'] == camera_id for camera in camera_manager.list_cameras()):
                    break
                time.sleep(0.1)
                continue
            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
            )
            time.sleep(0.03)

    return Response(
        generate_camera_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame',
    )


@app.route('/api/command', methods=['POST'])
def command():
    data = request.json or {}
    cmd = data.get('command')

    if cmd == 'start':
        source = data.get('source', 'test_crowd.mp4')
        print(f"[CMD] Start requested, source={source}")
        try:
            if initialize_pipeline(source):
                pipeline['running'] = True
                return jsonify({'status': 'ok'})
            else:
                return jsonify({'status': 'error', 'msg': f'Could not open video: {source}'})
        except Exception as e:
            print(f"[CMD] Start error: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({'status': 'error', 'msg': str(e)})

    elif cmd == 'stop':
        pipeline['running'] = False
        if pipeline['cap'] is not None:
            try:
                pipeline['cap'].release()
            except Exception:
                pass
            pipeline['cap'] = None
        pipeline['latest_frame_encoded'] = None
        return jsonify({'status': 'ok'})

    return jsonify({'status': 'unknown'})


@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'msg': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'status': 'error', 'msg': 'No selected file'})

    filename = file.filename
    save_path = VIDEO_DIR / filename

    # Ensure directory exists
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    file.save(str(save_path))
    print(f"[UPLOAD] Saved: {save_path}")

    return jsonify({'status': 'ok', 'filename': filename})


@app.route('/api/videos')
def list_videos():
    """List available video files."""
    videos = []
    if VIDEO_DIR.exists():
        for f in VIDEO_DIR.iterdir():
            if f.suffix.lower() in ('.mp4', '.avi', '.mov', '.mkv'):
                videos.append(f.name)
    return jsonify({'videos': videos})


if __name__ == '__main__':
    print(f"[STARTUP] VIDEO_DIR = {VIDEO_DIR}")
    print(f"[STARTUP] MAP_DIR   = {MAP_DIR}")
    if VIDEO_DIR.exists():
        print(f"[STARTUP] Videos: {[f.name for f in VIDEO_DIR.iterdir() if f.suffix.lower() in ('.mp4','.avi','.mov')]}")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
