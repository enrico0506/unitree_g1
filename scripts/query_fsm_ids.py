"""Read-only: ask the robot which FSM ids it accepts + its current fsm/mode/phase.

Run it at 802 (walk), then again after entering dance mode (503) on the dashboard,
to see whether 801 is an addressable id or a sub-task of 503.

    python3 scripts/query_fsm_ids.py [interface]

Sends nothing that moves the robot -- only Get* RPCs.
"""
import sys
from pathlib import Path
import yaml

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.h2.loco.h2_loco_client import LocoClient as H2LocoClient

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "robot.yaml"


def _iface(default="eth0"):
    try:
        with open(CONFIG_PATH) as f:
            return (yaml.safe_load(f) or {}).get("network", {}).get("interface", default)
    except OSError:
        return default


def _try(label, fn):
    try:
        print(f"  {label:22} = {fn()}", flush=True)
    except Exception as e:
        print(f"  {label:22} ! {type(e).__name__}: {e}", flush=True)


def main():
    iface = sys.argv[1] if len(sys.argv) > 1 else _iface()
    ChannelFactoryInitialize(0, iface)
    c = H2LocoClient()
    c.SetTimeout(2.0)
    c.Init()
    print("=== robot FSM query (read-only) ===", flush=True)
    _try("GetFsmId", c.GetFsmId)
    _try("GetFsmMode", c.GetFsmMode)
    _try("GetAvailableFsmIds", c.GetAvailableFsmIds)
    _try("GetBalanceMode", c.GetBalanceMode)
    _try("GetPhase", c.GetPhase)


if __name__ == "__main__":
    main()
