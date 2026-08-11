"""Dashboard-side thin client for the pose lane.

The pose container (pose_service.py) writes geometry + labels to shm; this
class centralizes every read/write of those files so route handlers in
robot_web_controller.py never touch shm paths directly.
"""
import json
import os
import time


class PoseClient:
    def __init__(self, tracks_path, labels_path, demand_path, ttl=2.5):
        self.tracks_path = tracks_path
        self.labels_path = labels_path
        self.demand_path = demand_path
        self.ttl = ttl

    def is_live(self):
        """True if tracks were written within ttl seconds."""
        try:
            return (time.time() - os.path.getmtime(self.tracks_path)) < self.ttl
        except OSError:
            return False

    def demand(self):
        """Heartbeat: pose container only infers while watched."""
        try:
            with open(self.demand_path, "wb") as f:
                f.write(b"1")
        except OSError:
            pass

    def tracks(self):
        """{w, h, items:[{id, name, box, kpts}]}, or an empty default."""
        try:
            with open(self.tracks_path) as f:
                return json.load(f)
        except (OSError, ValueError):
            return {"w": 0, "h": 0, "items": []}

    def set_label(self, track_id, name):
        """Map a track id to an operator-chosen name (empty name clears it)."""
        labels = {}
        try:
            with open(self.labels_path) as f:
                labels = json.load(f) or {}
        except (OSError, ValueError):
            pass
        if name:
            labels[track_id] = name
        else:
            labels.pop(track_id, None)
        tmp = self.labels_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(labels, f)
        os.replace(tmp, self.labels_path)   # atomic -> pose_service never reads a partial file
        return labels
