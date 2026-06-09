"""On-demand FAST-LIO map builder for the G1 dashboard.

Replaces the old odometry+depth voxel accumulator. Mapping now runs a real SLAM
backend (FAST-LIO on the head Mid-360) as an on-demand subprocess: start() spawns
~/g1_mapping_ws/run_mapping.sh (Mid-360 driver -> FAST-LIO -> map_bridge), stop()
SIGINTs it. The bridge writes the drift-corrected, floor-aligned map to a shared
-memory file; we read it here. The public API (start/stop/clear/add_scan/get_map/
has_points/save/load/list_maps/status/active) is unchanged, so the web server,
the /ws/lidar stream, the three.js renderer and the .npy/.ply files all keep
working exactly as before.

shm format (/dev/shm/g1_map.bin):  <uint32 N> <N*3 float32>  (written by map_bridge)
"""

import glob
import os
import signal
import struct
import subprocess
import threading

import numpy as np

MAP_SHM = "/dev/shm/g1_map.bin"
MAPPING_LOG = "/tmp/g1_mapping.log"


class MapBuilder:
    def __init__(self, map_dir, run_cmd, shm_path=MAP_SHM, max_points=300000):
        self.map_dir = map_dir
        self.run_cmd = run_cmd          # path to ~/g1_mapping_ws/run_mapping.sh
        self.shm_path = shm_path
        self.max_points = max_points
        self.active = False
        self.loaded = None              # name of a loaded map, or None
        self.saved = True               # False once unsaved points exist
        self._cloud = np.zeros((0, 3), np.float32)  # frozen map (loaded or last-recorded)
        self._proc = None
        self._lock = threading.Lock()

    # --- mapping (on-demand FAST-LIO subprocess) --------------------------

    def start(self):
        if self.active:
            return
        # fresh recording
        self._cloud = np.zeros((0, 3), np.float32)
        self.loaded = None
        try:
            os.remove(self.shm_path)
        except OSError:
            pass
        log = open(MAPPING_LOG, "wb")
        self._proc = subprocess.Popen(
            ["/bin/bash", self.run_cmd],
            start_new_session=True,     # own process group -> SIGINT tears the whole tree down
            stdout=log, stderr=subprocess.STDOUT)
        self.active = True
        self.saved = False
        print(f"[MAP] FAST-LIO mapping started (pid {self._proc.pid}); log {MAPPING_LOG}",
              flush=True)

    def stop(self):
        if not self.active and self._proc is None:
            return
        self.active = False
        p, self._proc = self._proc, None
        if p is not None and p.poll() is None:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGINT)   # graceful: FAST-LIO/bridge flush
            except ProcessLookupError:
                pass
            try:
                p.wait(timeout=12)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
        final = self._read_shm()        # freeze the final map so it survives for save()
        if len(final):
            self._cloud = final
        print(f"[MAP] mapping stopped ({len(self._cloud)} pts)", flush=True)

    def clear(self):
        with self._lock:
            self._cloud = np.zeros((0, 3), np.float32)
        self.loaded = None
        self.saved = True
        try:
            os.remove(self.shm_path)
        except OSError:
            pass

    def add_scan(self, cloud_body, pose):
        """No-op. FAST-LIO is the map engine now; the RealSense no longer feeds the map."""
        return

    # --- map access -------------------------------------------------------

    def _read_shm(self):
        try:
            with open(self.shm_path, "rb") as f:
                d = f.read()
            n = struct.unpack("<I", d[:4])[0]
            if n == 0:
                return np.zeros((0, 3), np.float32)
            return np.frombuffer(d[4:4 + n * 12], "<f4").reshape(-1, 3).copy()
        except (OSError, struct.error, ValueError):
            return np.zeros((0, 3), np.float32)

    def get_map(self):
        if self.active:
            live = self._read_shm()     # the growing FAST-LIO map
            return live if len(live) else self._cloud
        return self._cloud

    def has_points(self):
        return len(self.get_map()) > 0

    # --- persistence (explicit) ------------------------------------------

    def _path(self, name, ext):
        safe = "".join(ch for ch in name if ch.isalnum() or ch in "-_")
        return os.path.join(self.map_dir, f"{safe}.{ext}")

    def save(self, name):
        if not name:
            return False
        os.makedirs(self.map_dir, exist_ok=True)
        pts = self.get_map()
        np.save(self._path(name, "npy"), pts)
        _write_ply(self._path(name, "ply"), pts)
        self.loaded = name
        self.saved = True
        print(f"[MAP] saved '{name}' ({len(pts)} pts)", flush=True)
        return True

    def load(self, name):
        path = self._path(name, "npy")
        if not os.path.exists(path):
            return False
        if self.active:
            self.stop()
        with self._lock:
            self._cloud = np.load(path).astype(np.float32)
        self.loaded = name
        self.saved = True
        print(f"[MAP] loaded '{name}' ({len(self._cloud)} pts)", flush=True)
        return True

    def list_maps(self):
        if not os.path.isdir(self.map_dir):
            return []
        return sorted(os.path.splitext(os.path.basename(p))[0]
                      for p in glob.glob(os.path.join(self.map_dir, "*.npy")))

    def status(self):
        return {
            "type": "map_status",
            "active": self.active,
            "points": len(self.get_map()),
            "loaded": self.loaded,
            "saved": self.saved,
            "maps": self.list_maps(),
        }


def _write_ply(path, pts):
    """Minimal ASCII PLY export for external viewers."""
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(pts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("end_header\n")
        for x, y, z in pts:
            f.write(f"{x:.4f} {y:.4f} {z:.4f}\n")
