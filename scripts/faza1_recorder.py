#!/usr/bin/env python3
"""
faza1_recorder.py — sirovo snimanje VLA epizoda (Faza 1).

Po epizodi pravi folder:
  episode_XXXXXX/
    images/frame_YYYYYY.png   — Zivid 2D RGB (preset iz .yml, By2x2)
    data.csv                  — po frejmu: timestampovi, feedback joints,
                                cartesian, teleop komanda (buduca akcija)
    task.txt                  — jezicki opis zadatka (isti za celu epizodu)

Sync: rclpy spin u POZADINSKOJ niti drzi kesove svezim; glavna petlja na FPS
snepsotuje kesove -> Zivid capture (blokira ~55ms, zato je van ROS niti) ->
upis. Konverzija u LeRobotDataset ide kasnije (Faza 2, u condi/CUDA).

Preduslov: source ROS + aktiviran faza1 venv (rclpy + zivid + numpy).
            Zivid Studio ZATVOREN, WiFi adresa sklonjena sa 192.168.10.x.

Kontrola (u terminalu recordera):
  ENTER   -> start / stop epizode (stop = SACUVAJ)
  x ENTER -> tokom snimanja: ODBACI tekucu epizodu (obrisi, ne cuvaj)
  q ENTER -> kraj snimanja
"""
import argparse
import csv
import shutil
import sys
import threading
import time
from pathlib import Path

import numpy as np
import zivid

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory


class Caches(Node):
    """Drzi poslednju vrednost svakog toka + vreme prijema. Spin ide u drugoj niti."""

    def __init__(self):
        super().__init__("faza1_recorder")
        self._lock = threading.Lock()
        self.js = None
        self.js_t = None
        self.cp = None
        self.cp_t = None
        self.cmd = None
        self.cmd_t = None
        self.create_subscription(JointState, "/joint_states", self._on_js, 50)
        self.create_subscription(JointState, "/fb_c_pos", self._on_cp, 50)
        self.create_subscription(
            JointTrajectory,
            "/manipulator_controller/joint_trajectory",
            self._on_cmd, 50,
        )

    def _on_js(self, m):
        with self._lock:
            self.js = list(m.position)
            self.js_t = time.monotonic()

    def _on_cp(self, m):
        with self._lock:
            self.cp = list(m.position)
            self.cp_t = time.monotonic()

    def _on_cmd(self, m):
        if not m.points:
            return
        with self._lock:
            self.cmd = list(m.points[-1].positions)  # jedna tacka po poruci
            self.cmd_t = time.monotonic()

    def snapshot(self):
        with self._lock:
            return (self.js, self.js_t, self.cp, self.cp_t, self.cmd, self.cmd_t)


def next_episode_index(root):
    idx = [int(p.name.split("_")[1]) for p in root.glob("episode_*") if p.is_dir()]
    return max(idx) + 1 if idx else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="~/vla_dataset_raw", help="Root za sirove epizode")
    ap.add_argument("--fps", type=float, default=10.0, help="Ciljni fps snimanja")
    ap.add_argument("--preset", default="preset_2d.yml", help=".yml pored ove skripte")
    ap.add_argument("--task", default="", help="Jezicki opis zadatka za ovaj run")
    args = ap.parse_args()

    period = 1.0 / args.fps
    stale_ms = 1000.0 / args.fps * 1.5  # prag da je joint stanje "prestaro" za sliku

    # --- Zivid ---
    app = zivid.Application()
    camera = app.connect_camera()
    print("Kamera:", camera.info.model, "SN:", camera.info.serial_number)
    preset_path = Path(__file__).parent / args.preset
    settings_2d = zivid.Settings2D.load(preset_path)
    print(f"Preset ucitan: {preset_path}")

    # --- ROS (spin u pozadini) ---
    rclpy.init()
    node = Caches()
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()

    out_root = Path(args.out).expanduser()
    out_root.mkdir(parents=True, exist_ok=True)

    # --- kontrola tastaturom ---
    toggle = threading.Event()   # start/stop epizode
    abort = threading.Event()    # odbaci tekucu epizodu (x)
    quit_ev = threading.Event()  # kraj snimanja (q)

    def kb():
        while not quit_ev.is_set():
            line = sys.stdin.readline()
            s = line.strip().lower()
            if not line or s == "q":
                quit_ev.set()
                toggle.set()
                break
            if s == "x":
                abort.set()   # obelezi odbacivanje, pa prekini snimanje
            toggle.set()

    threading.Thread(target=kb, daemon=True).start()

    # cekaj prve poruke (joints i komanda su neophodni)
    print("Cekam prve ROS poruke (pokreni robota i teleop)...")
    while not quit_ev.is_set():
        js, _, _, _, cmd, _ = node.snapshot()
        if js is not None and cmd is not None:
            break
        time.sleep(0.1)
    if quit_ev.is_set():
        rclpy.shutdown()
        return
    js0, _, _, _, _, _ = node.snapshot()
    if len(js0) != 6:
        print(f"UPOZORENJE: /joint_states ima {len(js0)} vrednosti (ocekivano 6): {js0}")

    header = (
        ["frame", "t_img", "t_js", "t_cp", "t_cmd"]
        + [f"js_{i}" for i in range(6)]
        + [f"cart_{a}" for a in ["x", "y", "z", "rx", "ry", "rz"]]
        + [f"cmd_{i}" for i in range(6)]
        + ["image_path"]
    )

    ep_idx = next_episode_index(out_root)
    print("Spremno. ENTER = start/stop, x = odbaci tekucu, q = kraj.")

    while not quit_ev.is_set():
        print(f"\n[epizoda {ep_idx}] ENTER za start (q za kraj)...")
        toggle.wait()
        toggle.clear()
        if quit_ev.is_set():
            break

        abort.clear()  # svez start; ignorisi eventualni zaostali x
        ep_dir = out_root / f"episode_{ep_idx:06d}"
        (ep_dir / "images").mkdir(parents=True, exist_ok=True)
        (ep_dir / "task.txt").write_text(args.task + "\n")
        f = open(ep_dir / "data.csv", "w", newline="")
        writer = csv.writer(f)
        writer.writerow(header)

        print(f"  SNIMAM epizodu {ep_idx}... ENTER=stop/sacuvaj, x=odbaci.")
        frame = 0
        n_stale = 0
        t_start = time.monotonic()
        next_t = time.monotonic()

        while not toggle.is_set() and not quit_ev.is_set():
            # snepsot PRE capture-a: joints ~ vremenski poklopljeni sa scenom
            js, js_t, cp, cp_t, cmd, cmd_t = node.snapshot()
            t_ref = time.monotonic()

            # capture blokira ~55ms; ROS spin u drugoj niti drzi kesove svezim
            img = camera.capture_2d(settings_2d).image_rgba_srgb()

            if js is not None and cmd is not None:
                if (t_ref - js_t) * 1e3 > stale_ms:
                    n_stale += 1  # samo joints; stara komanda tokom drzanja je OK
                fname = f"frame_{frame:06d}.png"
                img.save(str(ep_dir / "images" / fname))
                cart = cp if cp is not None else [np.nan] * 6
                writer.writerow(
                    [frame, t_ref, js_t, cp_t, cmd_t]
                    + list(js) + list(cart) + list(cmd)
                    + [f"images/{fname}"]
                )
                frame += 1

            # ritam
            next_t += period
            s = next_t - time.monotonic()
            if s > 0:
                time.sleep(s)
            else:
                next_t = time.monotonic()  # zaostali smo -> resinhronizuj

        toggle.clear()
        f.close()

        if abort.is_set():
            abort.clear()
            shutil.rmtree(ep_dir, ignore_errors=True)
            print(f"  epizoda {ep_idx} ODBACENA (x). Nista nije sacuvano, broj se ponovo koristi.")
            # NE povecavamo ep_idx -> sledeca epizoda uzima isti broj
        else:
            dur = time.monotonic() - t_start
            eff = frame / dur if dur > 0 else 0.0
            print(
                f"  epizoda {ep_idx} SACUVANA: {frame} frejmova za {dur:.1f}s "
                f"(~{eff:.1f} Hz), stale={n_stale}. Resetuj scenu."
            )
            ep_idx += 1

    node.destroy_node()
    rclpy.shutdown()
    print("Kraj snimanja.")


if __name__ == "__main__":
    main()
