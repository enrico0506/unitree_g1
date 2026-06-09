"""Continuously read the G1 FSM, map IDs to names, and log transitions.

Verified FSM IDs (Enrico's G1, MyBotShop IEA, 2026-05-28):
    0   = zero_torque
    1   = damping
    4   = ready_stand     (initial state at boot -- needs final naming)
    802 = main_control    (walk mode -- needs final naming)

Unknown IDs are shown as 'fsm_???'. Add new mappings to FSM_NAMES below as
you discover them.

Ctrl+C to stop.
"""

import threading
import time
from datetime import datetime

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.h2.loco.h2_loco_client import LocoClient as H2LocoClient


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

INTERFACE = "eth0"
POLL_HZ = 10.0          # FSM poll rate
PRINT_HEARTBEAT_S = 5.0 # print a heartbeat every N seconds even if no change
RPC_TIMEOUT = 1.0       # per-call wall-clock timeout

# Verified FSM ID -> human name. Extend as you discover more.
FSM_NAMES = {
    0:   "zero_torque",
    1:   "damping",
    4:   "ready_stand",
    802: "main_control",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fsm_name(fsm_id):
    return FSM_NAMES.get(fsm_id, f"fsm_???_{fsm_id}")


def call_with_timeout(fn, timeout):
    """Call fn() but give up after `timeout` seconds. Returns the result or
    None on timeout/exception."""
    result = {"value": None, "exc": None}

    def runner():
        try:
            result["value"] = fn()
        except Exception as e:
            result["exc"] = e

    t = threading.Thread(target=runner, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive() or result["exc"] is not None:
        return None
    return result["value"]


def now_str():
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("init DDS...", flush=True)
    ChannelFactoryInitialize(0, INTERFACE)

    print("init H2 LocoClient...", flush=True)
    h2 = H2LocoClient()
    h2.SetTimeout(RPC_TIMEOUT)
    h2.Init()
    print(f"ready. polling at {POLL_HZ:.0f} Hz.\n", flush=True)
    print("Press remote combos. Transitions print immediately.", flush=True)
    print("Heartbeat prints every "
          f"{PRINT_HEARTBEAT_S:.0f}s with current state.\n", flush=True)

    last_fsm = None
    last_print = 0.0
    miss_count = 0
    period = 1.0 / POLL_HZ

    while True:
        result = call_with_timeout(h2.GetFsmId, RPC_TIMEOUT)

        if result is None:
            miss_count += 1
            if miss_count == 1 or miss_count % 50 == 0:
                print(f"[{now_str()}]  WARN: GetFsmId failed "
                      f"(miss #{miss_count})", flush=True)
            time.sleep(period)
            continue

        code, fsm = result
        if code != 0:
            print(f"[{now_str()}]  WARN: GetFsmId returned code={code}",
                  flush=True)
            time.sleep(period)
            continue

        # Reset miss counter on any success
        if miss_count > 0:
            print(f"[{now_str()}]  recovered after {miss_count} misses",
                  flush=True)
            miss_count = 0

        # Transition: print immediately
        if fsm != last_fsm:
            if last_fsm is None:
                print(f"[{now_str()}]  initial state: "
                      f"{fsm} ({fsm_name(fsm)})", flush=True)
            else:
                print(f"[{now_str()}]  transition: "
                      f"{last_fsm} ({fsm_name(last_fsm)}) -> "
                      f"{fsm} ({fsm_name(fsm)})", flush=True)
            last_fsm = fsm
            last_print = time.time()

        # Heartbeat: print current state every N seconds
        elif time.time() - last_print >= PRINT_HEARTBEAT_S:
            print(f"[{now_str()}]  steady: {fsm} ({fsm_name(fsm)})",
                  flush=True)
            last_print = time.time()

        time.sleep(period)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nstopped.")