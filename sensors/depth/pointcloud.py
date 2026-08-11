"""3D point cloud for the G1 dashboard, from the RealSense depth camera.

The MID-360 lidar isn't publishing on this robot, so we unproject the RealSense
depth stream (/dev/video0, Z16) into a dense 3D cloud. Output is (N, 3) float32
in a world frame where Z is UP, so the browser can colour points by height.

/dev/video0 (depth) is read with raw V4L2 (no pyrealsense2 needed) and coexists
with the videohub daemon, which only holds the color node.

WIRE FORMAT (pack_cloud): uint32 N, uint8 stride(=3), 3 pad, then N*3 float32.
"""

import ctypes
import fcntl
import mmap
import os
import select
import struct
import threading
import time

import numpy as np


# --- Depth unprojection (RealSense D435 depth @ 640x480, approx intrinsics) ---
DEPTH_W, DEPTH_H = 640, 480
DEPTH_SCALE = 0.001            # Z16 unit -> metres
FX = FY = 385.0
CX, CY = DEPTH_W / 2.0, DEPTH_H / 2.0
PIXEL_STEP = 3                 # subsample stride
DEPTH_MIN, DEPTH_MAX = 0.3, 8.0
MOUNT_HEIGHT = 1.3             # camera height (m); shifts floor near grid z=0
MAX_POINTS = 25000


# Raw V4L2 depth reader

_u32 = ctypes.c_uint32
_u8 = ctypes.c_uint8


class _Pix(ctypes.Structure):
    _fields_ = [(n, _u32) for n in (
        "width", "height", "pixelformat", "field", "bytesperline",
        "sizeimage", "colorspace", "priv", "flags", "enc",
        "quantization", "xfer_func")]


class _FmtU(ctypes.Union):
    _fields_ = [("pix", _Pix), ("raw", _u8 * 200), ("_a", ctypes.c_void_p)]


class _Format(ctypes.Structure):
    _fields_ = [("type", _u32), ("fmt", _FmtU)]


class _ReqBufs(ctypes.Structure):
    _fields_ = [("count", _u32), ("type", _u32), ("memory", _u32),
                ("capabilities", _u32), ("flags", _u8), ("reserved", _u8 * 3)]


class _TV(ctypes.Structure):
    _fields_ = [("s", ctypes.c_long), ("u", ctypes.c_long)]


class _TC(ctypes.Structure):
    _fields_ = [("type", _u32), ("flags", _u32), ("frames", _u8),
                ("seconds", _u8), ("minutes", _u8), ("hours", _u8),
                ("userbits", _u8 * 4)]


class _BM(ctypes.Union):
    _fields_ = [("offset", _u32), ("userptr", ctypes.c_ulong), ("fd", ctypes.c_int32)]


class _Buffer(ctypes.Structure):
    _fields_ = [("index", _u32), ("type", _u32), ("bytesused", _u32),
                ("flags", _u32), ("field", _u32), ("timestamp", _TV),
                ("timecode", _TC), ("sequence", _u32), ("memory", _u32),
                ("m", _BM), ("length", _u32), ("reserved2", _u32),
                ("request_fd", ctypes.c_int32)]


# canonical 64-bit V4L2 ioctl numbers
_S_FMT = 0xc0d05605
_REQBUFS = 0xc0145608
_QUERYBUF = 0xc0585609
_QBUF = 0xc058560f
_DQBUF = 0xc0585611
_STREAMON = 0x40045612
_STREAMOFF = 0x40045613
_CAP = 1
_MMAP = 1


def _fourcc(s):
    s = (s + "    ")[:4]
    return ord(s[0]) | ord(s[1]) << 8 | ord(s[2]) << 16 | ord(s[3]) << 24


class _DepthReader:
    """Minimal mmap V4L2 grabber for the RealSense Z16 depth node."""

    def __init__(self, dev="/dev/video0", w=DEPTH_W, h=DEPTH_H):
        self.dev, self.w, self.h = dev, w, h
        self.fd = None
        self.bufs = []

    def open(self):
        fd = os.open(self.dev, os.O_RDWR)
        f = _Format()
        f.type = _CAP
        f.fmt.pix.width = self.w
        f.fmt.pix.height = self.h
        f.fmt.pix.pixelformat = _fourcc("Z16")
        f.fmt.pix.field = 1
        fcntl.ioctl(fd, _S_FMT, f)
        self.w, self.h = f.fmt.pix.width, f.fmt.pix.height
        r = _ReqBufs()
        r.count = 4
        r.type = _CAP
        r.memory = _MMAP
        fcntl.ioctl(fd, _REQBUFS, r)
        for i in range(r.count):
            b = _Buffer()
            b.type = _CAP
            b.memory = _MMAP
            b.index = i
            fcntl.ioctl(fd, _QUERYBUF, b)
            mm = mmap.mmap(fd, b.length, mmap.MAP_SHARED,
                           mmap.PROT_READ | mmap.PROT_WRITE, offset=b.m.offset)
            self.bufs.append(mm)
            fcntl.ioctl(fd, _QBUF, b)
        fcntl.ioctl(fd, _STREAMON, ctypes.c_int(_CAP))
        self.fd = fd

    def read(self):
        """Return an (H, W) float32 depth image in metres, or None."""
        if not select.select([self.fd], [], [], 1.0)[0]:
            return None
        b = _Buffer()
        b.type = _CAP
        b.memory = _MMAP
        fcntl.ioctl(self.fd, _DQBUF, b)
        mm = self.bufs[b.index]
        mm.seek(0)
        raw = mm.read(self.w * self.h * 2)
        fcntl.ioctl(self.fd, _QBUF, b)
        depth = np.frombuffer(raw, dtype=np.uint16).reshape(self.h, self.w)
        return depth.astype(np.float32) * DEPTH_SCALE

    def close(self):
        if self.fd is not None:
            try:
                fcntl.ioctl(self.fd, _STREAMOFF, ctypes.c_int(_CAP))
                os.close(self.fd)
            except Exception:
                pass
            self.fd = None
        self.bufs = []


# ---------------------------------------------------------------------------
# Depth -> 3D cloud
# ---------------------------------------------------------------------------

# Precomputed (subsampled) pixel grid for unprojection.
_VS, _US = np.meshgrid(
    np.arange(0, DEPTH_H, PIXEL_STEP, dtype=np.float32),
    np.arange(0, DEPTH_W, PIXEL_STEP, dtype=np.float32),
    indexing="ij",
)
_UU = (_US - CX) / FX
_VV = (_VS - CY) / FY


def _depth_to_cloud(depth, mount_height):
    """Unproject a depth image (metres) to a world-frame (Z up) cloud."""
    d = depth[::PIXEL_STEP, ::PIXEL_STEP]
    m = (d > DEPTH_MIN) & (d < DEPTH_MAX)
    if not m.any():
        return np.zeros((0, 3), np.float32)
    z = d[m]
    # optical -> world: forward=z, left=-x_opt, up=-y_opt
    wx = z
    wy = -_UU[m] * z
    wz = -_VV[m] * z + mount_height
    return np.stack([wx, wy, wz], axis=1).astype(np.float32)


def pack_cloud(cloud):
    """Pack (N, 3) float32 -> uint32 N, uint8 stride, 3 pad, N*3 float32."""
    if cloud is None or len(cloud) == 0:
        return struct.pack("<IB3x", 0, 3)
    cloud = np.ascontiguousarray(cloud[:, :3], dtype="<f4")
    return struct.pack("<IB3x", cloud.shape[0], 3) + cloud.tobytes()


def _cap(cloud, n):
    if len(cloud) <= n:
        return cloud
    return cloud[np.random.choice(len(cloud), n, replace=False)]


# ---------------------------------------------------------------------------
# Odometry (robot pose for mapping)
# ---------------------------------------------------------------------------

class OdomReader:
    """Latest robot pose (x, y, yaw) from rt/odommodestate."""

    def __init__(self):
        self._lock = threading.Lock()
        self._pose = (0.0, 0.0, 0.0)
        self._live = False
        self._sub = None

    def start(self):
        from unitree_sdk2py.core.channel import ChannelSubscriber
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
        self._sub = ChannelSubscriber("rt/odommodestate", SportModeState_)
        self._sub.Init(self._on_odom, 10)
        print("[ODOM] subscribed rt/odommodestate", flush=True)

    def stop(self):
        if self._sub is not None:
            try:
                self._sub.Close()
            except Exception:
                pass

    def _on_odom(self, msg):
        p = msg.position
        yaw = msg.imu_state.rpy[2]
        with self._lock:
            self._pose = (float(p[0]), float(p[1]), float(yaw))
            self._live = True

    def get_pose(self):
        with self._lock:
            return self._pose

    def is_live(self):
        return self._live


# ---------------------------------------------------------------------------
# Source
# ---------------------------------------------------------------------------

class LidarSource:
    def __init__(self, max_points=MAX_POINTS, mount_height=MOUNT_HEIGHT,
                 odom=None, mapper=None):
        self.max_points = max_points
        self.mount_height = mount_height
        self.odom = odom
        self.mapper = mapper
        self.backend = "depth"
        self._lock = threading.Lock()
        self._latest = None
        self._live = False
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def get_cloud(self):
        with self._lock:
            return self._latest

    def is_live(self):
        return self._live

    def _run(self):
        reader = _DepthReader()
        try:
            reader.open()
        except Exception as e:
            print(f"[LIDAR] depth camera unavailable: {e}", flush=True)
            return
        print("[LIDAR] RealSense depth -> 3D cloud", flush=True)
        period = 0.1
        while not self._stop.is_set():
            t0 = time.time()
            try:
                depth = reader.read()
            except Exception as e:
                print(f"[LIDAR] read error: {e}", flush=True)
                depth = None
            if depth is not None:
                self._live = True
                cloud = _cap(_depth_to_cloud(depth, self.mount_height), self.max_points)
                with self._lock:
                    self._latest = cloud
                # Feed the map builder (no-op unless mapping is active).
                if self.mapper is not None and self.mapper.active and self.odom is not None:
                    self.mapper.add_scan(cloud, self.odom.get_pose())
            dt = time.time() - t0
            if dt < period:
                time.sleep(period - dt)
        reader.close()
