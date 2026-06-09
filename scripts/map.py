"""Discover what each Unitree remote button/combo maps to.

Subscribes to rt/wirelesscontroller and prints whenever ANY input changes
(buttons, sticks). Each new button combo is logged with:
  - the raw hex bitmask
  - the individual bit indices that are set
  - a guessed name (from the working bit map) -- edit BIT_NAMES below as you
    discover the true mapping

USAGE
    python3 scripts/remote_mapper.py

Workflow:
    1. Run this script.
    2. Press ONE button at a time (release between each).
    3. Note the bit index that lit up -- that IS your mapping.
    4. Edit BIT_NAMES at the top of this file to match.
    5. Now press combos (L2+A etc.) and verify the decoded names look right.

Ctrl+C to stop.
"""

import os
import signal
import socket
import sys
import time
import fcntl
import struct

from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_go.msg.dds_ import WirelessController_


# ---------------------------------------------------------------------------
# EDIT THIS AS YOU DISCOVER THE MAPPING
# ---------------------------------------------------------------------------
# Best guess starting point. Press one button at a time and correct as needed.
BIT_NAMES = {
    0:  "R1",
    1:  "L1",
    2:  "start",
    3:  "select",
    4:  "R2",
    5:  "L2",
    6:  "F1",
    7:  "F2",
    8:  "A",
    9:  "B",
    10: "X",
    11: "Y",
    12: "up",
    13: "right",
    14: "down",
    15: "left",
}

DEFAULT_INTERFACE = "eth0"
ROBOT_SUBNET_PREFIX = "192.168.123."
STICK_DEADZONE = 0.15  # ignore tiny stick noise


# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------

_stop = False
_sigint_count = 0

_last_keys = 0
_last_lx = 0.0
_last_ly = 0.0
_last_rx = 0.0
_last_ry = 0.0
_seen_combos = {}   # bitmask -> count
_start_time = None


# ---------------------------------------------------------------------------
# Signal handling
# ---------------------------------------------------------------------------

def _handle_sigint(signum, frame):
    global _stop, _sigint_count
    _sigint_count += 1
    if _sigint_count == 1:
        _stop = True
        print("\n[Ctrl+C -- printing summary and exiting...]", flush=True)
    else:
        os._exit(130)


# ---------------------------------------------------------------------------
# Interface auto-detection
# ---------------------------------------------------------------------------

def _iface_ipv4(iface):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        ip = socket.inet_ntoa(fcntl.ioctl(
            s.fileno(),
            0x8915,
            struct.pack('256s', iface[:15].encode('utf-8'))
        )[20:24])
        return ip
    except OSError:
        return None


def _list_interfaces():
    try:
        return os.listdir('/sys/class/net/')
    except OSError:
        return []


def autodetect_interface(preferred=DEFAULT_INTERFACE):
    ip = _iface_ipv4(preferred)
    if ip and ip.startswith(ROBOT_SUBNET_PREFIX):
        print(f"[auto] using {preferred} ({ip})", flush=True)
        return preferred
    for iface in _list_interfaces():
        if iface == "lo":
            continue
        ip = _iface_ipv4(iface)
        if ip and ip.startswith(ROBOT_SUBNET_PREFIX):
            print(f"[auto] using {iface} ({ip})", flush=True)
            return iface
    print(f"[auto] no robot subnet found, falling back to {preferred}",
          flush=True)
    return preferred


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------

def bits_set(mask):
    """Return list of bit indices that are set in mask."""
    return [i for i in range(16) if mask & (1 << i)]


def decode_combo(mask):
    """Return (bit_list, name_list, name_str)."""
    bits = bits_set(mask)
    names = [BIT_NAMES.get(b, f"bit{b}") for b in bits]
    name_str = "+".join(names) if names else "-"
    return bits, names, name_str


# ---------------------------------------------------------------------------
# DDS callback
# ---------------------------------------------------------------------------

def _on_wireless(msg):
    global _last_keys, _last_lx, _last_ly, _last_rx, _last_ry
    global _seen_combos, _start_time

    if _start_time is None:
        _start_time = time.time()

    t = time.time() - _start_time

    # Stick changes (with deadzone)
    lx, ly, rx, ry = msg.lx, msg.ly, msg.rx, msg.ry
    stick_changed = (
        abs(lx - _last_lx) > 0.05 or abs(ly - _last_ly) > 0.05 or
        abs(rx - _last_rx) > 0.05 or abs(ry - _last_ry) > 0.05
    )
    significant_stick = (
        abs(lx) > STICK_DEADZONE or abs(ly) > STICK_DEADZONE or
        abs(rx) > STICK_DEADZONE or abs(ry) > STICK_DEADZONE
    )

    # Button change
    if msg.keys != _last_keys:
        bits, names, name_str = decode_combo(msg.keys)
        _seen_combos[msg.keys] = _seen_combos.get(msg.keys, 0) + 1

        if msg.keys == 0:
            print(f"[{t:6.2f}s] released  (prev=0x{_last_keys:04x})",
                  flush=True)
        else:
            print(f"[{t:6.2f}s] keys=0x{msg.keys:04x}  bits={bits}  "
                  f"-> {name_str}", flush=True)
        _last_keys = msg.keys

    # Stick crossing into significant range
    if stick_changed and significant_stick:
        print(f"[{t:6.2f}s] sticks  L=({lx:+.2f},{ly:+.2f})  "
              f"R=({rx:+.2f},{ry:+.2f})", flush=True)

    _last_lx, _last_ly, _last_rx, _last_ry = lx, ly, rx, ry


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    signal.signal(signal.SIGINT, _handle_sigint)

    iface = autodetect_interface()
    print(f"[init] ChannelFactoryInitialize(0, '{iface}')", flush=True)
    ChannelFactoryInitialize(0, iface)

    sub = ChannelSubscriber("rt/wirelesscontroller", WirelessController_)
    sub.Init(_on_wireless, 10)

    print("", flush=True)
    print("=" * 60, flush=True)
    print(" REMOTE MAPPER ready.", flush=True)
    print(" Press ONE button at a time and note the bit index.", flush=True)
    print(" Then try combos. Ctrl+C when done for a summary.", flush=True)
    print("=" * 60, flush=True)
    print("", flush=True)

    while not _stop:
        time.sleep(0.05)

    # Summary
    print("", flush=True)
    print("=" * 60, flush=True)
    print(" SUMMARY -- unique combos seen", flush=True)
    print("=" * 60, flush=True)
    if not _seen_combos:
        print(" (none -- did the remote produce any input?)", flush=True)
    else:
        for mask in sorted(_seen_combos.keys()):
            if mask == 0:
                continue
            count = _seen_combos[mask]
            bits, names, name_str = decode_combo(mask)
            print(f"  0x{mask:04x}  bits={str(bits):20s}  "
                  f"x{count:<3d}  -> {name_str}", flush=True)
    print("", flush=True)
    print(" Copy the bit indices above into BIT_NAMES at the top of",
          flush=True)
    print(" this file to correct the mapping.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)