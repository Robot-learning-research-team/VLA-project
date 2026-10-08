#!/usr/bin/env python3
"""
dry_run_logger.py — karakterizacija ROS2 tokova PRE snimanja VLA dataseta.

Šta radi (ništa ne snima, ne komanduje robotom):
  1. Meri stvarne frekvencije /joint_states, /fb_c_pos i teleop komandi.
  2. Vrti fiksni takt na TARGET_FPS (isti mehanizam kojim će snimati dataset)
     i meri STAROST keša u tom trenutku — ako je velika, uparivanje slike i
     stanja bilo bi loše.
  3. Loguje šta teleop objavljuje na /manipulator_controller/joint_trajectory
     (broj tačaka, zglobovi, pozicije, time_from_start) — da odlučimo da li
     logujemo komandu direktno ili idemo state-as-action.

Pokretanje:
  source /opt/ros/<distro>/setup.bash  (+ tvoj workspace)
  python3 dry_run_logger.py
  ...pa TELEOPERIŠI ruku telefonom da uhvatiš komande.
  Ctrl+C za kraj.
"""

import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory

TARGET_FPS = 10.0          # ciljni fps dataseta — probaj i 15 kasnije
STALE_WARN_MS = 1000.0 / TARGET_FPS * 1.5   # prag za upozorenje o zastareloj poruci


def _round(vals, nd=1):
    return [round(v, nd) for v in vals]


class DryRunLogger(Node):
    def __init__(self):
        super().__init__("dry_run_logger")

        # keš poslednjih poruka + vreme prijema (monotoni sat)
        self.last_js = None
        self.last_js_t = None
        self.last_cpos = None
        self.last_cpos_t = None
        self._js_names_printed = False

        # brojači za merenje frekvencije
        self.counts = {"joint_states": 0, "fb_c_pos": 0, "cmd_traj": 0}
        self.t0 = time.monotonic()

        # statistika starosti keša u okviru prozora izveštaja
        self.max_js_age = 0.0
        self.max_cp_age = 0.0

        # publisheri su RELIABLE/VOLATILE -> default (RELIABLE, depth 50) se poklapa
        self.create_subscription(JointState, "/joint_states", self.on_js, 50)
        self.create_subscription(JointState, "/fb_c_pos", self.on_cpos, 50)
        self.create_subscription(
            JointTrajectory,
            "/manipulator_controller/joint_trajectory",
            self.on_cmd, 50,
        )

        # fiksni takt (isti kao budući recorder) + periodični izveštaj
        self.create_timer(1.0 / TARGET_FPS, self.on_tick)
        self.create_timer(2.0, self.report)

        self.get_logger().info(
            f"Logger pokrenut na {TARGET_FPS:.0f} Hz. Teleoperiši ruku da vidiš komande."
        )

    # ---- callbackovi: samo keširaju poslednju vrednost ----
    def on_js(self, msg):
        if not self._js_names_printed:
            self.get_logger().info(f"[/joint_states zglobovi] {list(msg.name)}")
            self._js_names_printed = True
        self.last_js = msg
        self.last_js_t = time.monotonic()
        self.counts["joint_states"] += 1

    def on_cpos(self, msg):
        self.last_cpos = msg
        self.last_cpos_t = time.monotonic()
        self.counts["fb_c_pos"] += 1

    def on_cmd(self, msg):
        # event-driven: štampa se samo kad teleop stvarno pošalje komandu
        self.counts["cmd_traj"] += 1
        n = len(msg.points)
        first = list(msg.points[0].positions) if n else []
        last = list(msg.points[-1].positions) if n else []
        tfs = 0.0
        if n:
            d = msg.points[-1].time_from_start
            tfs = d.sec + d.nanosec * 1e-9
        self.get_logger().info(
            f"[TELEOP CMD] tacaka={n} zglobovi={list(msg.joint_names)} "
            f"prvi={_round(first)} poslednji={_round(last)} time_from_start={tfs:.3f}s"
        )

    # ---- fiksni takt: meri starost keša (validacija sync-a) ----
    def on_tick(self):
        if self.last_js_t is None or self.last_cpos_t is None:
            return
        now = time.monotonic()
        js_age = (now - self.last_js_t) * 1e3
        cp_age = (now - self.last_cpos_t) * 1e3
        self.max_js_age = max(self.max_js_age, js_age)
        self.max_cp_age = max(self.max_cp_age, cp_age)
        if js_age > STALE_WARN_MS or cp_age > STALE_WARN_MS:
            self.get_logger().warn(
                f"[STALE] js={js_age:.0f}ms cart={cp_age:.0f}ms (prag {STALE_WARN_MS:.0f}ms)"
            )

    # ---- periodični izveštaj o frekvencijama i starosti ----
    def report(self):
        dt = time.monotonic() - self.t0
        r = {k: v / dt for k, v in self.counts.items()}
        cart = ""
        if self.last_cpos is not None:
            p = self.last_cpos.position
            cart = f" | Cart XY=({p[0]:.1f},{p[1]:.1f}) Z={p[2]:.1f} Rz={p[5]:.1f}"
        self.get_logger().info(
            f"[RATE] joint_states={r['joint_states']:.1f}Hz fb_c_pos={r['fb_c_pos']:.1f}Hz "
            f"cmd_traj={r['cmd_traj']:.2f}Hz ({self.counts['cmd_traj']} kom.) | "
            f"max starost: js={self.max_js_age:.0f}ms cart={self.max_cp_age:.0f}ms{cart}"
        )
        self.max_js_age = 0.0
        self.max_cp_age = 0.0


def main():
    rclpy.init()
    node = DryRunLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
