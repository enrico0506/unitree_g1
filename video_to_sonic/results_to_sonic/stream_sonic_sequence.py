from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import zmq


THIS_DIR = Path(__file__).resolve().parent

if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from sonic_protocol import build_command_message, pack_pose_message  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream a SONIC-ready pose sequence over ZMQ.")
    parser.add_argument("--input_npz", type=str, required=True)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5556)
    parser.add_argument("--topic", type=str, default="pose")
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--window_size", type=int, default=5)
    parser.add_argument("--warmup_s", type=float, default=0.1)
    parser.add_argument("--repeat", action="store_true")
    parser.add_argument("--send_start", action="store_true")
    parser.add_argument("--send_stop", action="store_true")
    parser.add_argument("--conflate", action="store_true")
    return parser.parse_args()


def _windowed(arr: np.ndarray, start: int, size: int) -> np.ndarray:
    end = min(arr.shape[0], start + size)
    chunk = arr[start:end]
    if chunk.shape[0] == size:
        return chunk
    pad = np.repeat(chunk[-1:], size - chunk.shape[0], axis=0)
    return np.concatenate([chunk, pad], axis=0)


def _load_sequence(input_npz: Path) -> dict[str, np.ndarray]:
    with np.load(input_npz, allow_pickle=False) as data:
        required = ("joint_pos", "joint_vel", "smpl_joints", "smpl_pose")
        missing = [key for key in required if key not in data]
        if missing:
            raise KeyError(f"Missing required SONIC fields in {input_npz}: {missing}")
        loaded = {
            "joint_pos": data["joint_pos"].astype(np.float32),
            "joint_vel": data["joint_vel"].astype(np.float32),
            "smpl_joints": data["smpl_joints"].astype(np.float32),
            "smpl_pose": data["smpl_pose"].astype(np.float32),
        }
        if "frame_index" in data:
            loaded["frame_index"] = data["frame_index"].astype(np.int64)
        return loaded


def _send_sequence(
    socket: zmq.Socket,
    sequence: dict[str, np.ndarray],
    fps: float,
    window_size: int,
    topic: str,
) -> None:
    frame_count = sequence["smpl_pose"].shape[0]
    frame_period = 1.0 / max(fps, 1e-6)

    for start in range(frame_count):
        frame_index_value = int(sequence["frame_index"][start]) if "frame_index" in sequence else start
        pose_data = {
            "joint_pos": _windowed(sequence["joint_pos"], start, window_size),
            "joint_vel": _windowed(sequence["joint_vel"], start, window_size),
            "smpl_joints": _windowed(sequence["smpl_joints"], start, window_size),
            "smpl_pose": _windowed(sequence["smpl_pose"], start, window_size),
            "frame_index": np.array([frame_index_value], dtype=np.int64),
            "catch_up": np.array([False], dtype=bool),
        }
        socket.send(pack_pose_message(pose_data, topic=topic, version=3))
        print(f"Sent frame {start + 1}/{frame_count}")
        time.sleep(frame_period)


def main() -> None:
    args = _parse_args()
    input_npz = Path(args.input_npz).resolve()
    if not input_npz.exists():
        raise FileNotFoundError(input_npz)

    sequence = _load_sequence(input_npz)
    frame_count = sequence["smpl_pose"].shape[0]
    if frame_count == 0:
        raise RuntimeError(f"Sequence {input_npz} is empty")

    ctx = zmq.Context()
    socket = ctx.socket(zmq.PUB)
    if args.conflate:
        socket.setsockopt(zmq.CONFLATE, 1)
    socket.bind(f"tcp://{args.host}:{args.port}")
    time.sleep(args.warmup_s)
    print(f"ZMQ pose streamer bound to tcp://{args.host}:{args.port}")
    print(f"Loaded {frame_count} frames from {input_npz}")

    try:
        if args.send_start:
            socket.send(build_command_message(start=True, stop=False, planner=False))
            print("Sent start command for pose streaming")

        while True:
            _send_sequence(
                socket,
                sequence,
                fps=args.fps,
                window_size=args.window_size,
                topic=args.topic,
            )
            if not args.repeat:
                break

        if args.send_stop:
            socket.send(build_command_message(start=False, stop=True, planner=False))
            print("Sent stop command")
    finally:
        socket.close()
        ctx.term()


if __name__ == "__main__":
    main()
