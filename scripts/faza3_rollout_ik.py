#!/usr/bin/env python3
"""
faza3_rollout_ik.py — FK->lock->IK rollout (venv 3.10), ISPRAVLJENA verzija.

Kljucna promena u odnosu na prvu verziju: clamp u JOINT prostoru je razarao
kartezijanski lock (odsecen joint pomak ne ostvaruje zakljucani Z -> Z curi).
Sada:
  - brzina se ogranicava u KARTEZIJANSKOM prostoru (--max-xy-step, m/tik) PRE IK-a
  - salje se PUN IK rezultat (bez joint-clamp-a) -> Z i orijentacija tacno zakljucani
  - loose --joint-safety-cap samo da uhvati katastrofalan IK skok (inace ne dira)

Frame cinjenice (izmereno): FANUC X,Y == MoveIt X,Y; FANUC_Z = MoveIt_z - 245mm
(konstantan wbase ofset). Zato lock MoveIt z-a drzi FANUC Z konstantnim, a trenutni
XY citamo direktno iz /fb_c_pos (mm->m).

Sigurnost: --z-floor (na /fb_c_pos, mm) i dalje kao nezavisna kocnica; e-stop.
Kontrola/feedback: timeout, hold-on-fail, jasan ispis (kao pre).
"""
import argparse
import csv
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
from geometry_msgs.msg import PoseStamped
from moveit_msgs.srv import GetPositionFK, GetPositionIK

JOINT_NAMES = ["J1", "J2", "J3", "J4", "J5", "J6"]
MOVEIT_SUCCESS = 1


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
        try:
            self._ensure()
            send_msg(self.sock, {"type": "reset"})
            recv_msg(self.sock)
            return True
        except (socket.timeout, ConnectionError, OSError) as e:
            print(f"[klijent] reset nije prosao: {e}")
            self._drop(); return False

    def infer(self, img, state, task, robot_type):
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
            print(f"[klijent] [TIMEOUT] server nije odgovorio za {self.timeout}s — drzim, reconnect...")
            self._drop(); return None, None
        except (ConnectionError, OSError) as e:
            print(f"[klijent] [SOCKET] veza pukla: {e} — drzim, reconnect...")
            self._drop(); return None, None


class RobotIO(Node):
    def __init__(self):
        super().__init__("faza3_rollout_ik")
        self._lock = threading.Lock()
        self.js = None
        self.cp = None
        self.create_subscription(JointState, "/joint_states", self._on_js, 50)
        self.create_subscription(JointState, "/fb_c_pos", self._on_cp, 50)
        self.pub = self.create_publisher(JointTrajectory, "/manipulator_controller/joint_trajectory", 10)
        self.fk_cli = self.create_client(GetPositionFK, "/compute_fk")
        self.ik_cli = self.create_client(GetPositionIK, "/compute_ik")

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

    def _call(self, cli, req, timeout):
        fut = cli.call_async(req)
        t0 = time.monotonic()
        while not fut.done():
            if time.monotonic() - t0 > timeout:
                return None
            time.sleep(0.002)
        return fut.result()

    def fk(self, joints, ee_link, base_frame, timeout):
        req = GetPositionFK.Request()
        req.header.frame_id = base_frame
        req.fk_link_names = [ee_link]
        req.robot_state.joint_state.name = JOINT_NAMES
        req.robot_state.joint_state.position = [float(x) for x in joints]
        res = self._call(self.fk_cli, req, timeout)
        if res is None:
            return None, "FK timeout"
        if res.error_code.val != MOVEIT_SUCCESS:
            return None, f"FK error_code={res.error_code.val}"
        return res.pose_stamped[0].pose, None

    def ik(self, pose, seed_joints, group, ee_link, base_frame, ik_timeout, timeout):
        req = GetPositionIK.Request()
        req.ik_request.group_name = group
        req.ik_request.ik_link_name = ee_link
        req.ik_request.robot_state.joint_state.name = JOINT_NAMES
        req.ik_request.robot_state.joint_state.position = [float(x) for x in seed_joints]
        ps = PoseStamped()
        ps.header.frame_id = base_frame
        ps.pose = pose
        req.ik_request.pose_stamped = ps
        req.ik_request.avoid_collisions = False
        req.ik_request.timeout = Duration(sec=0, nanosec=int(ik_timeout * 1e9))
        res = self._call(self.ik_cli, req, timeout)
        if res is None:
            return None, "IK timeout"
        if res.error_code.val != MOVEIT_SUCCESS:
            return None, f"IK error_code={res.error_code.val} (nema resenja / van dohvata?)"
        name = list(res.solution.joint_state.name)
        pos = list(res.solution.joint_state.position)
        try:
            out = [pos[name.index(j)] for j in JOINT_NAMES]
        except ValueError as e:
            return None, f"IK solution ne sadrzi sve zglobove: {e}"
        return np.array(out, dtype=np.float64), None


def next_run_index(root):
    idx = [int(p.name.split("_")[1]) for p in root.glob("run_*") if p.is_dir() and p.name.split("_")[1].isdigit()]
    return max(idx) + 1 if idx else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--preset", default="preset_2d.yml")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--timeout", type=float, default=2.0)
    ap.add_argument("--z-floor", type=float, default=None, help="mm (/fb_c_pos); prazno = off")
    # brzina se sada ogranicava u KARTEZIJANSKOM prostoru:
    ap.add_argument("--max-xy-step", type=float, default=0.005,
                    help="max pomak X/Y po tiku (m). Kontrola brzine BEZ razaranja lock-a.")
    ap.add_argument("--joint-safety-cap", type=float, default=0.15,
                    help="ako bi IK trazio pomak veci od ovoga po zglobu/tik -> odbaci (drzi). Samo protiv skokova.")
    ap.add_argument("--group", default="manipulator")
    ap.add_argument("--ee-link", default="tcp")
    ap.add_argument("--base-frame", default="base_link")
    ap.add_argument("--ik-timeout", type=float, default=0.05)
    ap.add_argument("--srv-timeout", type=float, default=0.5)
    ap.add_argument("--record-dir", default=None,
                    help="Ako je zadato, svaki rollout se snima u <dir>/run_XXXXXX/ "
                    "(slike + log.csv + task.txt) za kasniju procenu uspesnosti. Prazno = ne snima.")
    args = ap.parse_args()

    period = 1.0 / args.fps

    app = zivid.Application()
    camera = app.connect_camera()
    print("[klijent] Kamera:", camera.info.model, "SN:", camera.info.serial_number)
    settings_2d = zivid.Settings2D.load(Path(__file__).parent / args.preset)

    client = PolicyClient(args.host, args.port, args.timeout)
    try:
        client._connect()
    except OSError as e:
        print(f"[klijent] [SOCKET] server nedostupan ({e}). Pokusavacu reconnect.")

    rclpy.init()
    node = RobotIO()
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()

    print("[klijent] Cekam MoveIt servise /compute_fk i /compute_ik...")
    if not node.fk_cli.wait_for_service(timeout_sec=10.0) or not node.ik_cli.wait_for_service(timeout_sec=10.0):
        print("[klijent] GRESKA: /compute_fk ili /compute_ik nisu dostupni. move_group radi?")
        rclpy.shutdown(); return
    print("[klijent] MoveIt servisi OK.")

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

    print(f"[klijent] Spremno. Task: '{args.task}'. group='{args.group}', ee='{args.ee_link}', frame='{args.base_frame}'.")
    print(f"[klijent] max XY korak={args.max_xy_step*1000:.1f} mm/tik, joint safety cap={args.joint_safety_cap} rad. "
          f"Z-floor={args.z_floor if args.z_floor is not None else 'off'}. Ruka na e-stop!")

    while not quit_ev.is_set():
        print("\n[klijent] [rollout] ENTER za start (q za kraj)...")
        toggle.wait(); toggle.clear()
        if quit_ev.is_set():
            break

        js, cp = node.snapshot()
        start_pose, err = node.fk(js, args.ee_link, args.base_frame, args.srv_timeout)
        if start_pose is None:
            print(f"[klijent] Ne mogu startnu pozu ({err}). Proveri group/ee/frame. Ponovi ENTER.")
            continue
        z_lock = start_pose.position.z         # MoveIt z (m) -> drzi FANUC Z konstantnim
        q_lock = start_pose.orientation        # zakljucana orijentacija
        print(f"[klijent] LOCK: MoveIt z={z_lock*1000:.1f} mm (FANUC Z~{z_lock*1000-245:.1f}), "
              f"quat=({q_lock.x:.3f},{q_lock.y:.3f},{q_lock.z:.3f},{q_lock.w:.3f})")

        if not client.reset():
            print("[klijent] reset nije prosao. Ponovi ENTER.")
            continue

        # --- snimanje eval run-a (ako je --record-dir zadato) ---
        rec_writer = rec_file = rec_img_dir = None
        rec_count = 0
        if args.record_dir is not None:
            root = Path(args.record_dir).expanduser()
            root.mkdir(parents=True, exist_ok=True)
            run_idx = next_run_index(root)
            run_dir = root / f"run_{run_idx:06d}"
            rec_img_dir = run_dir / "images"
            rec_img_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "task.txt").write_text(args.task + "\n")
            (run_dir / "meta.txt").write_text(
                f"task={args.task}\ngroup={args.group} ee={args.ee_link} frame={args.base_frame}\n"
                f"lock_moveit_z_m={z_lock}\nmax_xy_step_m={args.max_xy_step}\nz_floor_mm={args.z_floor}\n")
            rec_file = open(run_dir / "log.csv", "w", newline="")
            rec_writer = csv.writer(rec_file)
            rec_writer.writerow(
                ["tick", "t_wall"]
                + [f"js_{i}" for i in range(6)]
                + ["cp_x", "cp_y", "cp_z", "cp_rx", "cp_ry", "cp_rz"]
                + [f"model_{i}" for i in range(6)]
                + [f"sent_{i}" for i in range(6)]
                + ["infer_ms", "image_path"])
            print(f"[klijent] Snimam eval run u {run_dir}")

        print("[klijent] ROLLOUT AKTIVAN... ENTER za stop.")
        next_t = time.monotonic()
        blocked_warned = False
        tick = 0

        while not toggle.is_set() and not quit_ev.is_set():
            js, cp = node.snapshot()
            if js is None or cp is None:
                continue

            def hold():
                nonlocal next_t
                next_t += period
                s = next_t - time.monotonic()
                if s > 0: time.sleep(s)
                else: next_t = time.monotonic()

            if args.z_floor is not None and cp[2] <= args.z_floor:
                if not blocked_warned:
                    print(f"[klijent] [Z-FLOOR] Z={cp[2]:.1f} <= {args.z_floor:.1f} — drzim.")
                    blocked_warned = True
                hold(); continue
            blocked_warned = False

            img_obj = camera.capture_2d(settings_2d).image_rgba_srgb()
            rgb = np.ascontiguousarray(img_obj.copy_data()[..., :3]).astype(np.uint8)
            state = np.array(list(js) + [cp[0], cp[1], cp[5]], dtype=np.float32)

            model_joints, infer_ms = client.infer(rgb, state, args.task, "fanuc_crx10ia")
            if model_joints is None:
                hold(); continue

            # gde model hoce vrh (X,Y) -> FK
            pm, err = node.fk(model_joints, args.ee_link, args.base_frame, args.srv_timeout)
            if pm is None:
                print(f"[klijent] [FK] {err} — drzim."); hold(); continue

            # trenutni XY (m): FANUC X,Y == MoveIt X,Y -> citamo iz /fb_c_pos (mm->m)
            cur_x, cur_y = cp[0] / 1000.0, cp[1] / 1000.0
            # ogranici XY korak (kontrola brzine, NE dira Z ni orijentaciju)
            tx = cur_x + float(np.clip(pm.position.x - cur_x, -args.max_xy_step, args.max_xy_step))
            ty = cur_y + float(np.clip(pm.position.y - cur_y, -args.max_xy_step, args.max_xy_step))

            # zakljucana poza: ogranicen XY, fiksni Z i orijentacija
            locked = PoseStamped().pose
            locked.position.x = tx
            locked.position.y = ty
            locked.position.z = z_lock
            locked.orientation = q_lock

            sol, err = node.ik(locked, js, args.group, args.ee_link, args.base_frame,
                               args.ik_timeout, args.srv_timeout)
            if sol is None:
                print(f"[klijent] [IK] {err} — drzim."); hold(); continue

            # SAMO sigurnosna provera protiv skoka; NE seci normalan pomak (to bi razorilo lock)
            max_dj = float(np.max(np.abs(sol - np.array(js, dtype=np.float64))))
            if max_dj > args.joint_safety_cap:
                print(f"[klijent] [SAFETY] IK skok {max_dj:.3f} rad > {args.joint_safety_cap} — drzim (moguc IK flip).")
                hold(); continue

            node.send(sol, period)  # PUN IK rezultat -> Z i orijentacija tacno zakljucani

            if rec_writer is not None:
                fname = f"frame_{rec_count:06d}.png"
                try:
                    img_obj.save(str(rec_img_dir / fname))
                    rec_writer.writerow(
                        [rec_count, time.time()]
                        + [float(x) for x in js]
                        + [float(cp[0]), float(cp[1]), float(cp[2]),
                           float(cp[3]), float(cp[4]), float(cp[5])]
                        + [float(x) for x in model_joints]
                        + [float(x) for x in sol]
                        + [infer_ms if infer_ms is not None else "", f"images/{fname}"])
                    rec_count += 1
                except Exception as e:
                    print(f"[klijent] [SNIMANJE] greska pri cuvanju frejma: {e}")

            tick += 1
            if tick % 20 == 0:
                print(f"[klijent] tik {tick}: infer {infer_ms} ms | FANUC Z={cp[2]:.1f} mm (treba ~const)")

            next_t += period
            s = next_t - time.monotonic()
            if s > 0: time.sleep(s)
            else: next_t = time.monotonic()

        toggle.clear()
        if rec_file is not None:
            rec_file.close()
            print(f"[klijent] Snimak sacuvan: {rec_count} frejmova u {rec_img_dir.parent}")
        print("[klijent] rollout zaustavljen (robot drzi zadnju poziciju).")

    node.destroy_node()
    rclpy.shutdown()
    print("[klijent] Kraj.")


if __name__ == "__main__":
    main()
