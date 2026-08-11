"""Dashboard-side thin client for the hands lane.

The hands container (hands_service.py) writes geometry to shm; this class
centralizes every read/write of that file so route handlers in
robot_web_controller.py never touch shm paths directly.
"""
import json
import os
import time


class HandsClient:
    def __init__(self, tracks_path, demand_path, ttl=2.5):
        self.tracks_path = tracks_path
        self.demand_path = demand_path
        self.ttl = ttl

    def is_live(self):
        """True if tracks were written within ttl seconds."""
        try:
            return (time.time() - os.path.getmtime(self.tracks_path)) < self.ttl
        except OSError:
            return False

    def demand(self):
        """Heartbeat: hands container only infers while watched."""
        try:
            with open(self.demand_path, "wb") as f:
                f.write(b"1")
        except OSError:
            pass

    def tracks(self):
        """{w, h, items:[{hand, score, landmarks:[[x,y,z]x21]}]}, or an empty default."""
        try:
            with open(self.tracks_path) as f:
                return json.load(f)
        except (OSError, ValueError):
            return {"w": 0, "h": 0, "items": []}
