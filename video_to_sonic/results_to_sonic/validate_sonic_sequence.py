from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


THIS_DIR = Path(__file__).resolve().parent

if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from sonic_protocol import pack_pose_message  # noqa: E402


HEADER_SIZE = 1280
REQUIRED_SHAPES = {
    "joint_pos": (None, 29),
    "joint_vel": (None, 29),
    "smpl_joints": (None, 24, 3),
    "smpl_pose": (None, 21, 3),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a SONIC-ready v3 sequence file.")
    parser.add_argument("--input_npz", type=str, required=True)
    parser.add_argument("--topic", type=str, default="pose")
    return parser.parse_args()


def _unpack_message(packed_data: bytes, topic: str) -> dict[str, np.ndarray]:
    topic_bytes = topic.encode("utf-8")
    if not packed_data.startswith(topic_bytes):
        raise ValueError(f"Message does not start with expected topic {topic!r}")

    offset = len(topic_bytes)
    if len(packed_data) < offset + HEADER_SIZE:
        raise ValueError("Packed data too small to contain SONIC header")

    header_bytes = packed_data[offset : offset + HEADER_SIZE]
    null_idx = header_bytes.find(b"\x00")
    if null_idx >= 0:
        header_bytes = header_bytes[:null_idx]
    header = json.loads(header_bytes.decode("utf-8"))

    dtype_map = {
        "f32": np.float32,
        "f64": np.float64,
        "i32": np.int32,
        "i64": np.int64,
        "bool": bool,
    }
    result: dict[str, np.ndarray] = {
        "version": np.array([header.get("v", 0)], dtype=np.int32),
    }

    current_offset = offset + HEADER_SIZE
    for field in header.get("fields", []):
        dtype = dtype_map[field["dtype"]]
        shape = tuple(field["shape"])
        n_bytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
        arr = np.frombuffer(packed_data[current_offset : current_offset + n_bytes], dtype=dtype)
        result[field["name"]] = arr.reshape(shape).copy()
        current_offset += n_bytes

    return result


def _validate_required_fields(data: dict[str, np.ndarray]) -> list[str]:
    errors: list[str] = []
    frame_count: int | None = None

    for key, expected_shape in REQUIRED_SHAPES.items():
        if key not in data:
            errors.append(f"Missing required field: {key}")
            continue

        arr = data[key]
        if arr.ndim != len(expected_shape):
            errors.append(f"{key}: expected {len(expected_shape)} dims, got shape {arr.shape}")
            continue

        if frame_count is None:
            frame_count = arr.shape[0]
        elif arr.shape[0] != frame_count:
            errors.append(f"Frame count mismatch: {key} has {arr.shape[0]} frames, expected {frame_count}")

        for axis, expected in enumerate(expected_shape[1:], start=1):
            if arr.shape[axis] != expected:
                errors.append(f"{key}: expected axis {axis} size {expected}, got {arr.shape[axis]}")

        if arr.dtype.kind not in "fiu":
            errors.append(f"{key}: expected numeric dtype, got {arr.dtype}")
        elif not np.isfinite(arr).all():
            errors.append(f"{key}: contains NaN or Inf")

    return errors


def _roundtrip_validate(data: dict[str, np.ndarray], topic: str) -> list[str]:
    errors: list[str] = []
    packed = pack_pose_message(
        {
            "joint_pos": data["joint_pos"],
            "joint_vel": data["joint_vel"],
            "smpl_joints": data["smpl_joints"],
            "smpl_pose": data["smpl_pose"],
        },
        topic=topic,
        version=3,
    )
    decoded = _unpack_message(packed, topic=topic)

    version = int(decoded["version"][0])
    if version != 3:
        errors.append(f"Packed message version mismatch: expected 3, got {version}")

    for key in ("joint_pos", "joint_vel", "smpl_joints", "smpl_pose"):
        if key not in decoded:
            errors.append(f"Decoded message missing field: {key}")
            continue
        if decoded[key].shape != data[key].shape:
            errors.append(f"Decoded shape mismatch for {key}: {decoded[key].shape} vs {data[key].shape}")
            continue
        if not np.allclose(decoded[key], data[key]):
            errors.append(f"Decoded values differ for {key}")

    return errors


def main() -> None:
    args = _parse_args()
    input_npz = Path(args.input_npz).resolve()
    if not input_npz.exists():
        raise FileNotFoundError(input_npz)

    with np.load(input_npz, allow_pickle=False) as loaded:
        data = {key: loaded[key] for key in loaded.files}

    errors = _validate_required_fields(data)
    if not errors:
        errors.extend(_roundtrip_validate(data, topic=args.topic))

    frame_count = int(data["smpl_pose"].shape[0]) if "smpl_pose" in data else -1

    print(f"File: {input_npz}")
    print(f"Frames: {frame_count}")
    for key in ("joint_pos", "joint_vel", "smpl_joints", "smpl_pose"):
        if key in data:
            arr = data[key]
            arr_min = float(np.min(arr))
            arr_max = float(np.max(arr))
            print(f"{key}: shape={arr.shape} dtype={arr.dtype} min={arr_min:.6f} max={arr_max:.6f}")

    if errors:
        print("Validation: FAILED")
        for err in errors:
            print(f"- {err}")
        raise SystemExit(1)

    print("Validation: OK")
    print("This means the file is structurally valid for SONIC protocol v3.")
    print("It does not guarantee good robot behavior, but it strongly reduces format-level failure risk.")


if __name__ == "__main__":
    main()
