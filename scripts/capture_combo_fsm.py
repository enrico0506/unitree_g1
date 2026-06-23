"""Capture which whole-body FSM a remote combo triggers.

Some remote combos (R2+B = climb, R1+B = dance) switch the robot into a
whole-body mode driven by an FSM id we don't know until we watch it on the
real robot. This tool listens to the remote (rt/wirelesscontroller) and polls
LocoClient.GetFsmId(), printing the FSM id the robot lands in after a combo.

READ-ONLY: it sends nothing to the robot, it only observes.

USAGE
    python3 scripts/capture_combo_fsm.py [interface]

    1. Run it (robot standing, clear space, e-stop in reach).
    2. Press the combo on the remote (e.g. R2+B for climb).
    3. Read the printed "FSM <old> -> <new>" line — that <new> is the id.
    4. Put it in config/mapping.yaml under mode_combos.<name>.fsm_id and tell
       the dashboard to wire a SetFsmId(<new>) button.

Ctrl+C to stop.
"""

import signal
import sys
import threading
import time
from pathlib import Path

import yaml

from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_go.msg.dds_ import WirelessController_
from unitree_sdk2py.h2.loco.h2_loco_client import LocoClient as H2LocoClient


CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "robot.yaml"

# Bit -> name (matches config/mapping.yaml remote_keys).
BIT_NAMES = {
    0: "R1", 1: "L1", 2: "start", 3: "select", 4: "R2", 5: "L2",
    6: "F1", 7: "F2", 8: "A", 9: "B", 10: "X", 11: "Y",
    12: "up", 13: "left", 14: "down", 15: "right",
}

# Known FSM ids (extend as you learn more) -- purely for nicer printouts.
FSM_NAMES = {
    0: "zero_torque", 1: "damping", 4: "ready_stand", 802: "main_control",
}

POLL_HZ = 10.0
RPC_TIMEOUT = 1.0
COMBO_WINDOW_S = 2.0   # an FSM change within this long of a combo is attributed to it

_stop = False
_last_combo = {"mask": 0, "names": "-", "t": 0.0}


def _interface_from_config(default="eth0"):
    try:
        with open(CONFIG_PATH) as f:
            cfg = yaml.safe_load(f) or {}
        return cfg.get("network", {}).get("interface", default)
    except OSError:
        return default


def _bits(mask):
    return [i for i in range(16) if mask & (1 << i)]


def _combo_name(mask):
    names = [BIT_NAMES.get(b, f"bit{b}") for b in _bits(mask)]
    return "+".join(names) if names else "-"


def _fsm_name(fsm_id):
    if fsm_id is None:
        return "unknown"
    return FSM_NAMES.get(fsm_id, f"fsm_{fsm_id}")


def _on_wireless(msg):
    # Remember the most recent NON-zero button combo and when it was pressed.
    if msg.keys:
        _last_combo["mask"] = msg.keys
        _last_combo["names"] = _combo_name(msg.keys)
        _last_combo["t"] = time.time()


def _get_fsm(reader):
    """GetFsmId with a wall-clock timeout (the RPC can hang)."""
    result = {"v": None}

    def run():
        try:
            result["v"] = reader.GetFsmId()
        except Exception:
            result["v"] = None

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(RPC_TIMEOUT)
    if t.is_alive() or result["v"] is None:
        return None
    code, fsm = result["v"]
    return fsm if code == 0 else None


def _handle_sigint(signum, frame):
    global _stop
    _stop = True


def main():
    signal.signal(signal.SIGINT, _handle_sigint)

    iface = sys.argv[1] if len(sys.argv) > 1 else _interface_from_config()
    print(f"[init] ChannelFactoryInitialize(0, '{iface}')", flush=True)
    ChannelFactoryInitialize(0, iface)

    reader = H2LocoClient()
    reader.SetTimeout(RPC_TIMEOUT)
    reader.Init()

    sub = ChannelSubscriber("rt/wirelesscontroller", WirelessController_)
    sub.Init(_on_wireless, 10)

    print("=" * 64, flush=True)
    print(" COMBO -> FSM capture (read-only). Press a combo on the remote.", flush=True)
    print(" e.g. R2+B (climb) or R1+B (dance). Ctrl+C to stop.", flush=True)
    print("=" * 64, flush=True)

    last_fsm = _get_fsm(reader)
    print(f"[fsm] start: {last_fsm} ({_fsm_name(last_fsm)})", flush=True)

    period = 1.0 / POLL_HZ
    while not _stop:
        fsm = _get_fsm(reader)
        if fsm is not None and fsm != last_fsm:
            age = time.time() - _last_combo["t"]
            if _last_combo["t"] > 0 and age <= COMBO_WINDOW_S:
                cause = (f"combo {_last_combo['names']} "
                         f"(0x{_last_combo['mask']:04x}) {age:.2f}s ago")
            else:
                cause = "no recent combo (manual / timeout?)"
            print(f"[fsm] {last_fsm} ({_fsm_name(last_fsm)}) -> "
                  f"{fsm} ({_fsm_name(fsm)})   <- {cause}", flush=True)
            last_fsm = fsm
        time.sleep(period)

    print("\n[done]", flush=True)


if __name__ == "__main__":
    main()
