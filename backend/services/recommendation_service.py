"""
recommendation_service.py
Generates data-grounded, actionable energy-saving recommendations
based on predicted hourly load profiles, diurnal peak windows,
and sub-metering appliance dynamics.
"""

import sys
import os
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd

class RecommendationService:
    @staticmethod
    def generate_recommendations(predictions, current_hour=None):
        """
        predictions: list of dicts with 'timestamp', 'predicted_consumption' (kWh), 'level', etc.
        """
        if not predictions or len(predictions) == 0:
            return [{
                "category": "General",
                "priority": "Low",
                "recommendation": "Maintain standard energy conservation practices by turning off idle electronics."
            }]
            
        recs = []
        
        # 1. Peak-Hour Analysis (18:00 to 21:00)
        peak_preds = [
            p for p in predictions 
            if pd.to_datetime(p['timestamp']).hour in [18, 19, 20, 21]
        ]
        
        if peak_preds:
            max_peak = max(peak_preds, key=lambda x: x['predicted_consumption'])
            peak_val = max_peak['predicted_consumption']
            peak_time = pd.to_datetime(max_peak['timestamp']).strftime("%I %p").lstrip("0")
            
            if peak_val > 1.8:
                recs.append({
                    "category": "Peak Load Shifting",
                    "priority": "High",
                    "title": "Shift Appliance Usage Away from Evening Peak",
                    "recommendation": f"High consumption of ~{peak_val:.2f} kWh is forecast around {peak_time}. Consider scheduling high-power appliances (washing machine, dishwasher, dryer) after 9:30 PM to avoid peak-demand surcharges."
                })
            else:
                recs.append({
                    "category": "Peak Management",
                    "priority": "Medium",
                    "title": "Moderate Evening Peak Expected",
                    "recommendation": f"Evening peak consumption is projected to remain moderate (~{peak_val:.2f} kWh). Keep lighting and climate control at eco-settings."
                })
                
        # 2. Nighttime Base Load Check (01:00 to 05:00)
        night_preds = [
            p for p in predictions 
            if pd.to_datetime(p['timestamp']).hour in [1, 2, 3, 4, 5]
        ]
        if night_preds:
            avg_night = sum(p['predicted_consumption'] for p in night_preds) / len(night_preds)
            if avg_night > 0.65:
                recs.append({
                    "category": "Vampire Power",
                    "priority": "Medium",
                    "title": "Elevated Standby Power Detected",
                    "recommendation": f"Projected overnight baseline consumption ({avg_night:.2f} kWh/h) is higher than normal (typically < 0.35 kWh). Check for appliances left on standby or space heating cycling continuously."
                })
            else:
                recs.append({
                    "category": "Base Load",
                    "priority": "Low",
                    "title": "Efficient Standby Baseline",
                    "recommendation": f"Overnight standby load is healthy at ~{avg_night:.2f} kWh/h, indicating minimal phantom energy loss."
                })

        # 3. Overall 24-Hour Highest Consumption Window
        max_overall = max(predictions, key=lambda x: x['predicted_consumption'])
        max_ts = pd.to_datetime(max_overall['timestamp']).strftime("%a %I:%M %p")
        if max_overall['predicted_consumption'] >= 3.0:
            recs.append({
                "category": "High Demand Alert",
                "priority": "High",
                "title": f"Anticipated Energy Surge at {max_ts}",
                "recommendation": f"Consumption is expected to surge to {max_overall['predicted_consumption']:.2f} kWh. Avoid concurrent operation of electric water heaters, ovens, and AC units during this hour."
            })

        # 4. Actionable Efficiency Tip
        recs.append({
            "category": "Smart Automation",
            "priority": "Low",
            "title": "Smart Thermostat & Timer Scheduling",
            "recommendation": "Pre-cool or pre-heat living spaces 30 minutes before projected peak hours, then let thermal inertia maintain comfortable temperatures without drawing peak power."
        })
        
        return recs
