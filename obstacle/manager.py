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

import glob
import os
import signal
import subprocess
import threading
import time

OBSTACLE_LOG = "/tmp/g1_obstacle.log"


# Command-line signatures of our obstacle subprocess tree (bash wrapper -> ros2
# launch -> obstacle_node.py). Used to find and reap an ORPHANED tree left behind
# when a prior controller was hard-killed / crashed so its lifespan stop() never ran:
# start_new_session=True detaches the tree, so it survives, keeps holding the
# Mid-360's fixed UDP ports, and silently starves a freshly-spawned node (which then
# fails to bind the LiDAR while the orphan keeps writing the shm with stale code).
# Matched narrowly (path token AND the expected interpreter) so an editor/grep that
# merely NAMES one of these files is never mistaken for a running node.
def _is_obstacle_cmd(cmd):
    if "obstacle_node.py" in cmd and "python" in cmd:
        return True
    if "obstacle.launch.py" in cmd and "launch" in cmd:
        return True
    if "run_obstacle.sh" in cmd and "bash" in cmd:
        return True
    return False


def _pgid_alive(pgid):
    """True if any process in process-group `pgid` still exists (signal-0 probe)."""
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:                 # exists but not ours (same-user: unexpected)
        return True


class ObstacleManager:
    def __init__(self, run_cmd, log_path=OBSTACLE_LOG):
        self.run_cmd = run_cmd          # path to obstacle/run_obstacle.sh
        self.log_path = log_path
        self.active = False
        self._proc = None
        self._lock = threading.Lock()

    # --- lifecycle --------------------------------------------------------

    def start(self):
        """Spawn the obstacle node subprocess. Idempotent: a no-op if OUR node is
        already alive. Before spawning a fresh one, reap any ORPHANED obstacle tree
        left by a hard-killed / crashed prior controller, so the new node can bind
        the Mid-360's fixed UDP ports instead of silently losing to the orphan (which
        is why a plain `restart g1-web` used to leave the stale node running)."""
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                self.active = True          # already up -> reconcile a stale flag
                return
            # Our handle is dead/None, so anything still matching our signature is a
            # stale orphan. Tear it down first so the LiDAR ports are free.
            reaped = self._reap_orphans()
            if reaped:
                print(f"[OBSTACLE] reaped {reaped} orphaned obstacle process "
                      f"group(s) before start", flush=True)
            log = open(self.log_path, "wb")
            self._proc = subprocess.Popen(
                ["/bin/bash", self.run_cmd],
                start_new_session=True,   # own process group -> SIGINT tears the whole tree down
                stdout=log, stderr=subprocess.STDOUT)
            self.active = True
            print(f"[OBSTACLE] node started (pid {self._proc.pid}); log {self.log_path}",
                  flush=True)

    def _reap_orphans(self):
        """SIGINT (then SIGKILL) any process group whose command matches our obstacle
        subprocess signature. Returns the number of groups signalled (0 = none stale).

        Scans /proc (dependency-free) and tears down each DISTINCT process group, so
        the whole bash -> ros2 launch -> node tree (and the node's worker threads)
        goes down together. Only ever called from start() when our own _proc is
        dead/None, so every match is by definition a stale orphan -- we never reap a
        node we mean to keep. Same-user processes, so no privileges are needed; any
        per-group signal failure is swallowed so start() still proceeds."""
        groups = set()
        for entry in glob.glob("/proc/[0-9]*/cmdline"):
            try:
                with open(entry, "rb") as fh:
                    cmd = fh.read().replace(b"\x00", b" ").decode("utf-8", "replace")
            except OSError:
                continue                    # process vanished mid-scan
            if not cmd or not _is_obstacle_cmd(cmd):
                continue
            try:
                pid = int(entry.rsplit("/", 2)[1])
                groups.add(os.getpgid(pid))
            except (ValueError, IndexError, ProcessLookupError, PermissionError):
                continue
        if not groups:
            return 0
        for pgid in groups:                 # graceful first: let ROS free the sockets
            try:
                os.killpg(pgid, signal.SIGINT)
            except (ProcessLookupError, PermissionError):
                pass
        deadline = time.time() + 3.0        # wait for a clean exit, then hard-kill
        while time.time() < deadline:
            if not any(_pgid_alive(g) for g in groups):
                break
            time.sleep(0.1)
        for pgid in groups:
            if _pgid_alive(pgid):
                try:
                    os.killpg(pgid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
        return len(groups)

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
