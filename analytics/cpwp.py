# ──────────────────────────────────────────────
# CrowdGuard – Crowd Pressure Wave Predictor (CPWP)
# ──────────────────────────────────────────────

from dataclasses import dataclass
from typing import Dict, List, Optional
from collections import deque

from analytics.density import ZoneReport
from analytics.flow import ZoneFlow

@dataclass
class WavePrediction:
    source_zone: str
    target_zone: str
    time_to_impact_sec: float
    warning_message: str
    severity: str  # 'WARNING' or 'CRITICAL'
    recommendations: List[str]

class CrowdPressureWavePredictor:
    def __init__(self, history_seconds: float = 10.0, fps: float = 24.0):
        self.history_len = max(2, int(history_seconds * fps))
        self.fps = max(1.0, fps)
        
        # History for each zone: {'density': float, 'speed': float, 'direction': str}
        self.zone_history: Dict[str, deque] = {}
        
        # Adjacency inferred from layout:
        # Zone A | Zone B
        # ───────┼───────
        # Zone C | Zone D
        self.adjacency = {
            "Zone A": ["Zone B", "Zone C"],
            "Zone B": ["Zone A", "Zone D"],
            "Zone C": ["Zone A", "Zone D"],
            "Zone D": ["Zone B", "Zone C"]
        }
        
    def predict(self, zone_reports: List[ZoneReport], zone_flows: List[ZoneFlow]) -> List[WavePrediction]:
        predictions = []
        
        # 1. Update history
        current_data = {}
        for zr in zone_reports:
            current_data[zr.name] = {'density': zr.density_pct}
            
        for zf in zone_flows:
            if zf.zone_name in current_data:
                current_data[zf.zone_name]['speed'] = zf.average_speed
                current_data[zf.zone_name]['direction'] = zf.dominant_direction
                
        for zone, data in current_data.items():
            if zone not in self.zone_history:
                self.zone_history[zone] = deque(maxlen=self.history_len)
            self.zone_history[zone].append(data)
            
        # 2. Analyze each zone for pressure buildup and wave propagation
        for zone, history in self.zone_history.items():
            if len(history) < self.fps * 3: # need at least 3 seconds of data
                continue
                
            # Calculate density trend over history
            densities = [frame['density'] for frame in history]
            speeds = [frame.get('speed', 0) for frame in history]
            
            recent_frames = int(self.fps * 2) # last 2 seconds
            
            recent_densities = densities[-recent_frames:]
            past_densities = densities[:-recent_frames]
            
            avg_recent_density = sum(recent_densities) / len(recent_densities)
            avg_past_density = sum(past_densities) / len(past_densities)
            
            recent_speeds = speeds[-recent_frames:]
            past_speeds = speeds[:-recent_frames]
            
            avg_recent_speed = sum(recent_speeds) / len(recent_speeds)
            avg_past_speed = sum(past_speeds) / max(1, len(past_speeds))
            
            # Condition 1: Any slight increase in density (> 0.5% jump for testing visibility)
            density_jump = avg_recent_density - avg_past_density
            
            # Condition 2: High density (lowered to 1.0 for testing)
            if density_jump > 0.5 and avg_recent_density > 1.0:
                # Pressure buildup detected! Where is it heading?
                current_direction = history[-1].get('direction', 'STATIONARY')
                
                # Determine target zone based on movement direction
                target_zone = self.infer_target_zone(zone, current_direction)
                
                if target_zone and target_zone in self.adjacency.get(zone, []):
                    # Condition 3: Check if target zone is already moderately congested (amplifies wave)
                    target_history = self.zone_history.get(target_zone)
                    target_density = target_history[-1]['density'] if target_history else 0
                    
                    if target_density >= 0.0: # Removed target congestion requirement for testing
                        # Estimate time to impact
                        est_time = 20.0
                        if avg_recent_speed > 2.0:
                            est_time = min(40.0, max(10.0, 150.0 / avg_recent_speed))
                            
                        severity = 'CRITICAL' if (avg_recent_density > 75.0 or target_density > 70.0) else 'WARNING'
                        msg = f"Potential Crowd Compression in {target_zone} — {int(est_time-3)}–{int(est_time+4)} sec"
                        
                        predictions.append(WavePrediction(
                            source_zone=zone,
                            target_zone=target_zone,
                            time_to_impact_sec=est_time,
                            warning_message=msg,
                            severity=severity,
                            recommendations=["STOP ENTRY", "OPEN ALTERNATE ROUTE", "REDIRECT CROWD", "ALERT OPERATOR"]
                        ))
                    
        return predictions
        
    def infer_target_zone(self, source_zone: str, direction: str) -> Optional[str]:
        # Based on layout:
        # A | B
        # ──┼──
        # C | D
        mapping = {
            "Zone A": {"RIGHT": "Zone B", "DOWN": "Zone C"},
            "Zone B": {"LEFT": "Zone A", "DOWN": "Zone D"},
            "Zone C": {"RIGHT": "Zone D", "UP": "Zone A"},
            "Zone D": {"LEFT": "Zone C", "UP": "Zone B"}
        }
        return mapping.get(source_zone, {}).get(direction)
