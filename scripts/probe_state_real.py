#!/usr/bin/env python3
# probe_state_real.py — ispisuje STATE koji model stvarno dobija na realnom robotu:
#   js (6 zglobova, rad) + cart iz /fb_c_pos: cp[0]=x_mm, cp[1]=y_mm, cp[5]=rz.
# Nema Zivid/MoveIt/policy — samo cita ROS topike. Isti venv/ROS kao faza3_rollout_ik.py.
#
# Upotreba: pokreni, jog-uj robota u 2-3 poze iznad stola, za svaku prepisi ispisane
# js (tih 6 brojeva) i cart. js posle ubacis u rollout_sim.py --probe (TEST_QS).
import threading
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


def dbg_state(tag, js, cx_mm, cy_mm, rz):
    print(f"[{tag}] js=[" + ", ".join(f"{v:+.4f}" for v in js) + "]  "
          f"cart_x={cx_mm:+.1f}mm  cart_y={cy_mm:+.1f}mm  rz={rz:+.5f}  "
          f"(rz-(-2.375018)={rz-(-2.375018):+.5f})", flush=True)


class Probe(Node):
    def __init__(self):
        super().__init__("probe_state_real")
        self._lock = threading.Lock()
        self.js = None
        self.cp = None
        self.create_subscription(JointState, "/joint_states", self._on_js, 50)
        self.create_subscription(JointState, "/fb_c_pos", self._on_cp, 50)

    def _on_js(self, m):
        with self._lock:
            self.js = list(m.position)

    def _on_cp(self, m):
        with self._lock:
            self.cp = list(m.position)

    def snap(self):
        with self._lock:
            return self.js, self.cp


def main():
    rclpy.init()
    node = Probe()
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
    print("[probe] Cekam /joint_states i /fb_c_pos... (Ctrl+C za kraj)")
    try:
        while True:
            js, cp = node.snap()
            if js is not None and cp is not None and len(js) >= 6 and len(cp) >= 6:
                dbg_state("REAL", js[:6], cp[0], cp[1], cp[5])
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
