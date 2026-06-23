#!/usr/bin/env python3
"""g1_cmd_vel_bridge -- Nav2 /cmd_vel (domain 99) -> Unitree LocoClient (domain 0).

WHY A BRIDGE
------------
Two DDS worlds that must not see each other:
  * The Livox / FAST-LIO / Nav2 stack runs on ROS_DOMAIN_ID=99 (CycloneDDS),
    sourced via /home/unitree/g1_mapping_ws/env.sh. Nav2's controller publishes
    geometry_msgs/Twist on /cmd_vel here.
  * The Unitree LocoClient talks to the robot on DDS domain 0, bound to eth0.
This bridge subscribes /cmd_vel on domain 99 and forwards each Twist as a Unitree
SetVelocity call on domain 0 -- the only component that legitimately straddles
both domains.

DESIGN NOTES
  * The G1 is HOLONOMIC for our purposes: it accepts vx, vy AND vyaw. The DWB
    controller is configured holonomic (see nav2_params.yaml) so /cmd_vel carries
    all three. We forward linear.x, linear.y, angular.z.
  * Velocity is sent via the no-reply RPC path (_CallNoReply) exactly like
    scripts/robot_web_controller.py::send_velocity -- a blocking Move() at Nav2's
    controller rate would saturate the DDS RPC layer and stall movement.
  * Caps from config/robot.yaml are re-applied here as a final safety clamp:
    max_vx=1.5, max_vy=0.7, max_vyaw=2.0. Nav2 also limits velocity, but the
    bridge is the last line before the robot.
  * 'duration' is the robot's deadman hold for one command. We use a short hold
    (move_duration_s) and rely on Nav2's steady stream to keep it alive; a
    watchdog stops the robot if /cmd_vel goes silent.

  !! SAFETY: This node DRIVES THE REAL ROBOT. Only run it once the full Nav2 stack
     is validated in sim/teleop. It is OFF in the default bringup launch (commented
     out) and must be enabled deliberately. -- <ASK ENRICO: e-stop / spotting plan
     before first autonomous /cmd_vel forward>

ChannelFactoryInitialize binds domain 0 / eth0 -- run this node WITHOUT sourcing
env.sh's domain-99 export into the SDK init, OR ensure the SDK init wins. In
practice: rclpy uses ROS_DOMAIN_ID=99 for the /cmd_vel subscription, and the
Unitree SDK is initialized explicitly to (0, "eth0"), so the two coexist in one
process. Confirm on the robot. -- <ASK ENRICO: verify single-process dual-domain>
"""

import json

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Twist

# Unitree SDK -- domain 0 / eth0. Same imports as scripts/robot_web_controller.py.
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
from unitree_sdk2py.g1.loco.g1_loco_api import ROBOT_API_ID_LOCO_SET_VELOCITY


# Safety caps (config/robot.yaml). The bridge is the last clamp before the robot.
MAX_VX = 1.5      # m/s
MAX_VY = 0.7      # m/s
MAX_VYAW = 2.0    # rad/s
MOVE_DURATION_S = 1.0   # robot deadman hold per command
WATCHDOG_S = 0.5        # stop if no /cmd_vel within this window


def _clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


class G1CmdVelBridge(Node):
    def __init__(self):
        super().__init__("g1_cmd_vel_bridge")

        # --- Unitree LocoClient on domain 0 / eth0 (drives the real robot) ----
        ChannelFactoryInitialize(0, "eth0")
        self.client = LocoClient()
        self.client.SetTimeout(1.0)
        self.client.Init()
        self.get_logger().warn(
            "g1_cmd_vel_bridge: LocoClient initialized on DDS domain 0 (eth0). "
            "This node DRIVES THE ROBOT from Nav2 /cmd_vel."
        )

        # --- Nav2 /cmd_vel on domain 99 (rclpy uses ROS_DOMAIN_ID=99) ---------
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.sub = self.create_subscription(Twist, "/cmd_vel", self._on_cmd_vel, qos)

        # Watchdog: zero the robot if /cmd_vel goes quiet.
        self._last_cmd_t = self.get_clock().now()
        self._moving = False
        self.create_timer(0.1, self._watchdog)

        self.get_logger().info(
            f"bridging /cmd_vel(d99) -> SetVelocity(d0) | caps vx={MAX_VX} vy={MAX_VY} "
            f"vyaw={MAX_VYAW} | watchdog={WATCHDOG_S}s"
        )

    def _on_cmd_vel(self, msg):
        vx = _clamp(float(msg.linear.x), -MAX_VX, MAX_VX)
        vy = _clamp(float(msg.linear.y), -MAX_VY, MAX_VY)
        vyaw = _clamp(float(msg.angular.z), -MAX_VYAW, MAX_VYAW)
        self._send_velocity(vx, vy, vyaw, MOVE_DURATION_S)
        self._last_cmd_t = self.get_clock().now()
        self._moving = not (vx == 0.0 and vy == 0.0 and vyaw == 0.0)

    def _watchdog(self):
        if not self._moving:
            return
        dt = (self.get_clock().now() - self._last_cmd_t).nanoseconds * 1e-9
        if dt > WATCHDOG_S:
            self.get_logger().warn(f"/cmd_vel silent {dt:.2f}s -> stopping robot")
            self._send_velocity(0.0, 0.0, 0.0, MOVE_DURATION_S)
            self._moving = False

    def _send_velocity(self, vx, vy, vyaw, duration):
        """Fire-and-forget SetVelocity (no-reply RPC) -- never blocks the executor."""
        payload = json.dumps({"velocity": [vx, vy, vyaw], "duration": duration})
        self.client._CallNoReply(ROBOT_API_ID_LOCO_SET_VELOCITY, payload)


def main(args=None):
    rclpy.init(args=args)
    node = G1CmdVelBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node._send_velocity(0.0, 0.0, 0.0, MOVE_DURATION_S)  # stop on exit
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
