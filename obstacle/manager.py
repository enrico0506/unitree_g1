"""ObstacleManager -- start/stop the obstacle perception node subprocess.

The obstacle ROS2 node (obstacle_node.py) needs the sourced ROS2 Foxy + Livox
environment on its own isolated DDS domain (99), which the dashboard process
(domain 0, Unitree SDK) cannot share. So, exactly like the mapping stack, we run
it as an on-demand subprocess via a bash launcher (run_obstacle.sh).

This mirrors the subprocess lifecycle of scripts/map_builder.py (MapBuilder):
  - Popen(["/bin/bash", run_cmd], start_new_session=True, ...) so the whole
    process group can be torn down with one SIGINT;
  - stop() SIGINTs the group, waits, then SIGKILLs as a fallback;
  - start() is idempotent; is_alive() polls the process; an `active` flag and a
    status() dict feed the dashboard.

Imported by the dashboard (domain 0). No rclpy here.
"""

import os
import signal
import subprocess
import threading

OBSTACLE_LOG = "/tmp/g1_obstacle.log"


class ObstacleManager:
    def __init__(self, run_cmd, log_path=OBSTACLE_LOG):
        self.run_cmd = run_cmd          # path to obstacle/run_obstacle.sh
        self.log_path = log_path
        self.active = False
        self._proc = None
        self._lock = threading.Lock()

    # --- lifecycle --------------------------------------------------------

    def start(self):
        """Spawn the obstacle node subprocess. Idempotent: a no-op if already
        running."""
        with self._lock:
            if self.active and self._proc is not None and self._proc.poll() is None:
                return
            log = open(self.log_path, "wb")
            self._proc = subprocess.Popen(
                ["/bin/bash", self.run_cmd],
                start_new_session=True,   # own process group -> SIGINT tears the whole tree down
                stdout=log, stderr=subprocess.STDOUT)
            self.active = True
            print(f"[OBSTACLE] node started (pid {self._proc.pid}); log {self.log_path}",
                  flush=True)

    def stop(self):
        """Gracefully stop the subprocess group, falling back to SIGKILL."""
        with self._lock:
            if not self.active and self._proc is None:
                return
            self.active = False
            p, self._proc = self._proc, None
        if p is not None and p.poll() is None:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGINT)   # graceful tree teardown
            except ProcessLookupError:
                pass
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
        print("[OBSTACLE] node stopped", flush=True)

    # --- status -----------------------------------------------------------

    def is_alive(self):
        p = self._proc
        return p is not None and p.poll() is None

    def status(self):
        return {
            "type": "obstacle_manager",
            "active": self.active,
            "alive": self.is_alive(),
        }
