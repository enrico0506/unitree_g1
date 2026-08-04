from __future__ import annotations

import json
import struct

import numpy as np


HEADER_SIZE = 1280


def _build_header(fields: list[dict[str, object]], version: int = 1, count: int = 1) -> bytes:
    header = {
        "v": version,
        "endian": "le",
        "count": count,
        "fields": fields,
    }
    header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")
    if len(header_json) > HEADER_SIZE:
        raise ValueError(f"Header too large: {len(header_json)} > {HEADER_SIZE}")
    return header_json.ljust(HEADER_SIZE, b"\x00")


def build_command_message(
    start: bool,
    stop: bool,
    planner: bool,
    delta_heading: float | None = None,
) -> bytes:
    fields: list[dict[str, object]] = [
        {"name": "start", "dtype": "u8", "shape": [1]},
        {"name": "stop", "dtype": "u8", "shape": [1]},
        {"name": "planner", "dtype": "u8", "shape": [1]},
    ]
    payload = b"".join(
        (
            struct.pack("B", 1 if start else 0),
            struct.pack("B", 1 if stop else 0),
            struct.pack("B", 1 if planner else 0),
        )
    )

    if delta_heading is not None:
        fields.append({"name": "delta_heading", "dtype": "f32", "shape": [1]})
        payload += struct.pack("<f", float(delta_heading))

    return b"command" + _build_header(fields, version=1, count=1) + payload


def pack_pose_message(pose_data: dict[str, np.ndarray], topic: str = "pose", version: int = 3) -> bytes:
    fields: list[dict[str, object]] = []
    binary_data: list[bytes] = []

    for key, value in pose_data.items():
        if not isinstance(value, np.ndarray):
            continue

        if value.dtype == np.float32:
            dtype_str = "f32"
        elif value.dtype == np.float64:
            dtype_str = "f64"
        elif value.dtype == np.int32:
            dtype_str = "i32"
        elif value.dtype == np.int64:
            dtype_str = "i64"
        elif value.dtype == np.uint8:
            dtype_str = "u8"
        elif value.dtype == bool:
            dtype_str = "bool"
        else:
            dtype_str = "f32"
            value = value.astype(np.float32)

        fields.append({"name": key, "dtype": dtype_str, "shape": list(value.shape)})

        if not value.flags["C_CONTIGUOUS"]:
            value = np.ascontiguousarray(value)
        if value.dtype.byteorder == ">":
            value = value.astype(value.dtype.newbyteorder("<"))

        binary_data.append(value.tobytes())

    header_bytes = _build_header(fields, version=version, count=1)
    return topic.encode("utf-8") + header_bytes + b"".join(binary_data)
