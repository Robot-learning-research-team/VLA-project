#!/usr/bin/env python3
"""
faza3_rollout.py — ROS + Zivid KLIJENT (pokretati u rollout venv, Python 3.10).

Nema lerobot/torch zavisnosti. Po tiku: uslika Zivid + procita stanje -> posalje
observaciju policy serveru (policy_server.py, u condi) -> dobije 6 joint targeta
-> clamp + z-floor -> posalje robotu.

FEEDBACK/REAL-TIME:
  - Svaki recv ima timeout (--timeout). Petlja NIKAD ne visi.
  - Timeout ili puknuta veza -> NE salje se komanda (robot drzi poziciju),
    jasan ispis, reconnect sledeci tik.
  - Serverova greska stize kao poruka i ispise se.
BEZBEDNOST: --max-joint-step (rad/tik), --z-floor (mm), ruka na e-stop-u.

Kontrola: ENTER = start/stop rollout-a; q ENTER = kraj.

Preduslov: PRVO pokreni policy_server.py u condi, pa ovo u rollout venv-u.
           ROS source; Zivid Studio zatvoren; WiFi adresa sklonjena sa 192.168.10.x.
"""
import argparse
import json
import socket
import struct
import sys
import threading
import time
from pathlib import Path

import numpy as np
import zivid

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

JOINT_NAMES = ["J1", "J2", "J3", "J4", "J5", "J6"]


# ---------- framed protokol (isti kao na serveru) ----------
def recv_all(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("veza zatvorena tokom citanja")
        buf.extend(chunk)
    return bytes(buf)


def recv_msg(sock):
    hlen, blen = struct.unpack(">II", recv_all(sock, 8))
    header = json.loads(recv_all(sock, hlen).decode("utf-8"))
    blob = recv_all(sock, blen) if blen else b""
    return header, blob


def send_msg(sock, header, blob=b""):
    hj = json.dumps(header).encode("utf-8")
    sock.sendall(struct.pack(">II", len(hj), len(blob)) + hj + blob)


class PolicyClient:
    """Persistentna veza ka serveru; sam se reconnectuje; svaki poziv je time-bounded."""

    def __init__(self, host, port, timeout):
        self.host, self.port, self.timeout = host, port, timeout
        self.sock = None

    def _connect(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        s.settimeout(self.timeout)
        s.connect((self.host, self.port))
        self.sock = s
        print(f"[klijent] Povezan na server {self.host}:{self.port}.")

    def _ensure(self):
        if self.sock is None:
            self._connect()

    def _drop(self):
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass
            self.sock = None

    def reset(self):
        """Javi serveru novu epizodu. Vraca True ako je uspelo."""
        try:
            self._ensure()
            send_msg(self.sock, {"type": "reset"})
            recv_msg(self.sock)
            return True
        except socket.timeout:
            print(f"[klijent] [TIMEOUT] reset nije potvrdjen za {self.timeout}s.")
            self._drop(); return False
        except (ConnectionError, OSError) as e:
            print(f"[klijent] [SOCKET] reset nije prosao: {e}")
            self._drop(); return False

    def infer(self, img, state, task, robot_type):
        """Vraca (action np.ndarray(6), infer_ms) ili (None, None) ako je pao/istekao."""
        try:
            self._ensure()
            img = np.ascontiguousarray(img)
            header = {"type": "infer", "shape": list(img.shape),
                      "state": [float(x) for x in state], "task": task, "robot_type": robot_type}
            send_msg(self.sock, header, img.tobytes())
            rheader, _ = recv_msg(self.sock)
            if rheader.get("status") == "ok":
                return np.array(rheader["action"], dtype=np.float64), rheader.get("infer_ms")
            print(f"[klijent] [SERVER ERROR] {rheader.get('msg')}")
            return None, None
        except socket.timeout:
            print(f"[klijent] [TIMEOUT] server nije odgovorio za {self.timeout}s — "
                  f"drzim poziciju, reconnect...")
            self._drop(); return None, None
        except (ConnectionError, OSError) as e:
            print(f"[klijent] [SOCKET] veza pukla: {e} — drzim poziciju, reconnect...")
            self._drop(); return None, None


class Caches(Node):
    def __init__(self):
        super().__init__("faza3_rollout")
        self._lock = threading.Lock()
        self.js = None
        self.cp = None
        self.create_subscription(JointState, "/joint_states", self._on_js, 50)
        self.create_subscription(JointState, "/fb_c_pos", self._on_cp, 50)
        self.pub = self.create_publisher(JointTrajectory, "/manipulator_controller/joint_trajectory", 10)

    def _on_js(self, m):
        with self._lock:
            self.js = list(m.position)

    def _on_cp(self, m):
        with self._lock:
            self.cp = list(m.position)

    def snapshot(self):
        with self._lock:
            return self.js, self.cp

    def send(self, target, dt):
        jt = JointTrajectory()
        jt.joint_names = JOINT_NAMES
        pt = JointTrajectoryPoint()
        pt.positions = [float(x) for x in target]
        pt.time_from_start = Duration(sec=0, nanosec=int(dt * 1e9))
        jt.points = [pt]
        self.pub.publish(jt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--preset", default="preset_2d.yml")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--timeout", type=float, default=2.0,
                    help="max cekanje na server po pozivu (s). Detektor mrtvog servera, ne po-tik budzet.")
    ap.add_argument("--max-joint-step", type=float, default=0.015, help="rad/zglob/tik (SIGURNOST)")
    ap.add_argument("--z-floor", type=float, default=None,
                    help="mm; ako trenutni Z padne na/ispod, komanda se ne salje. Prazno = iskljuceno.")
    args = ap.parse_args()

    period = 1.0 / args.fps

    # --- Zivid ---
    app = zivid.Application()
    camera = app.connect_camera()
    print("[klijent] Kamera:", camera.info.model, "SN:", camera.info.serial_number)
    settings_2d = zivid.Settings2D.load(Path(__file__).parent / args.preset)

    # --- policy klijent (probaj odmah da se povezes, da rano vidis ako server ne radi) ---
    client = PolicyClient(args.host, args.port, args.timeout)
    try:
        client._connect()
    except OSError as e:
        print(f"[klijent] [SOCKET] Ne mogu da se povezem na server ({e}). "
              f"Da li si pokrenuo policy_server.py u condi na {args.host}:{args.port}? "
              f"Nastavicu i pokusavati reconnect.")

    # --- ROS ---
    rclpy.init()
    node = Caches()
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()

    toggle = threading.Event()
    quit_ev = threading.Event()

    def kb():
        while not quit_ev.is_set():
            line = sys.stdin.readline()
            if not line or line.strip().lower() == "q":
                quit_ev.set(); toggle.set(); break
            toggle.set()

    threading.Thread(target=kb, daemon=True).start()

    print("[klijent] Cekam prve ROS poruke (pokreni robota)...")
    while not quit_ev.is_set():
        js, cp = node.snapshot()
        if js is not None and cp is not None:
            break
        time.sleep(0.1)
    if quit_ev.is_set():
        rclpy.shutdown(); return

    _, cp0 = node.snapshot()
    print(f"[klijent] Spremno. Task: '{args.task}'. ENTER = start/stop, q = kraj.")
    print(f"[klijent] SIGURNOST: max {args.max_joint_step} rad/zglob/tik, timeout {args.timeout}s. Ruka na e-stop!")
    print(f"[klijent] Trenutni Z = {cp0[2]:.1f} mm.", end=" ")
    if args.z_floor is not None:
        print(f"Z-floor = {args.z_floor:.1f} mm.")
        if cp0[2] <= args.z_floor:
            print("[klijent] UPOZORENJE: vec si NA/ISPOD Z-floor-a — blokirace odmah.")
    else:
        print("Z-floor iskljucen.")

    while not quit_ev.is_set():
        print("\n[klijent] [rollout] ENTER za start (q za kraj)...")
        toggle.wait(); toggle.clear()
        if quit_ev.is_set():
            break

        if not client.reset():
            print("[klijent] reset nije prosao — proveri server pa ponovo ENTER.")
            continue
        print("[klijent] ROLLOUT AKTIVAN... ENTER za stop.")
        next_t = time.monotonic()
        blocked_warned = False
        tick = 0

        while not toggle.is_set() and not quit_ev.is_set():
            js, cp = node.snapshot()
            if js is None or cp is None:
                continue

            # Z-FLOOR: tvrda granica visine
            if args.z_floor is not None and cp[2] <= args.z_floor:
                if not blocked_warned:
                    print(f"[klijent] [Z-FLOOR] Z={cp[2]:.1f} <= {args.z_floor:.1f} — blokiram, drzim poziciju.")
                    blocked_warned = True
                next_t += period
                s = next_t - time.monotonic()
                if s > 0: time.sleep(s)
                else: next_t = time.monotonic()
                continue
            blocked_warned = False

            rgb = np.ascontiguousarray(
                camera.capture_2d(settings_2d).image_rgba_srgb().copy_data()[..., :3]).astype(np.uint8)
            state = np.array(list(js) + [cp[0], cp[1], cp[5]], dtype=np.float32)  # x, y, rz

            target, infer_ms = client.infer(rgb, state, args.task, "fanuc_crx10ia")

            if target is None:
                # timeout/greska vec ispisani -> drzi poziciju, samo odrzi ritam
                next_t += period
                s = next_t - time.monotonic()
                if s > 0: time.sleep(s)
                else: next_t = time.monotonic()
                continue

            # SIGURNOSNI CLAMP
            cur = np.array(js, dtype=np.float64)
            delta = np.clip(target - cur, -args.max_joint_step, args.max_joint_step)
            node.send(cur + delta, period)

            tick += 1
            if tick % 20 == 0 and infer_ms is not None:
                print(f"[klijent] tik {tick}: infer {infer_ms} ms | Z={cp[2]:.1f}")

            next_t += period
            s = next_t - time.monotonic()
            if s > 0: time.sleep(s)
            else: next_t = time.monotonic()

        toggle.clear()
        print("[klijent] rollout zaustavljen (robot drzi zadnju poziciju).")

    node.destroy_node()
    rclpy.shutdown()
    print("[klijent] Kraj.")


if __name__ == "__main__":
    main()
