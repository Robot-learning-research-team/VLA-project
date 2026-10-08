#!/usr/bin/env python3
"""
faza2_convert.py — sirove epizode (Faza 1) -> LeRobotDataset za SmolVLA.

Radi:
  1. Cita svaku episode_XXXXXX/ (data.csv + task.txt + images/).
  2. action[t] = js[t+1]  (state-as-action, joint prostor) — na PUNOJ epizodi pre trima.
  3. Idle-trim: odseca mrtav POCETAK (pre prvog pokreta). REP (mirovanje na cilju)
     se ZADRZAVA da model nauci da STANE na cilju (osim uz --trim-tail). Sredinu ne dira.
  4. observation.state = [J1..J6, cart_x, cart_y, cart_rz] (9), action = [J1..J6] (6).
  5. Slika: RGBA->RGB (drop alfa), po zelji resize.
  6. Sklapa LeRobotDataset preko create/add_frame/save_episode/finalize (API 0.6.x).

VIDEO KODEK: default H.264 (torchcodec ga svuda dekodira). LeRobot-ov podrazumevani
AV1 (libsvtav1) cesto ne moze da se procita nazad. Ako i H.264 zeza pri citanju,
koristi --images (cuva PNG umesto videa, zaobilazi torchcodec u potpunosti).

Pokretanje (conda, lerobot 0.6.2):
  python faza2_convert.py --raw ~/vla_dataset_raw --out ~/vla_lerobot \
      --repo-id promaja/fanuc_push --overwrite
"""
import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from lerobot.datasets import LeRobotDataset
from lerobot.configs.video import RGBEncoderConfig

JOINTS = [f"js_{i}" for i in range(6)]
CART = ["cart_x", "cart_y", "cart_rz"]


def parse_resize(s):
    if not s:
        return None
    w, h = s.lower().split("x")
    return int(w), int(h)  # PIL trazi (W, H)


def load_image(path, resize_wh):
    img = Image.open(path).convert("RGB")  # drop alfa
    if resize_wh is not None:
        img = img.resize(resize_wh, Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)  # (H, W, 3)


def moving_mask(js, thr):
    d = np.linalg.norm(np.diff(js, axis=0), axis=1)  # N-1
    d = np.concatenate([[0.0], d])                   # poravnaj na N
    return d > thr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="~/vla_dataset_raw")
    ap.add_argument("--out", default="~/vla_lerobot")
    ap.add_argument("--repo-id", default="promaja/fanuc_push")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--idle-thr", type=float, default=0.002)
    ap.add_argument("--resize", default=None, help='npr. "320x256" (WxH)')
    ap.add_argument("--codec", default="h264", help="video kodek (h264 preporuceno)")
    ap.add_argument("--images", action="store_true",
                    help="Cuvaj PNG umesto videa (zaobilazi torchcodec)")
    ap.add_argument("--trim-tail", action="store_true",
                    help="Odseci i mirovanje na KRAJU epizode. Podrazumevano se REP ZADRZAVA "
                    "(uci model da stane na cilju).")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    raw = Path(args.raw).expanduser()
    out = Path(args.out).expanduser()
    resize_wh = parse_resize(args.resize)

    eps = sorted(p for p in raw.glob("episode_*") if p.is_dir())
    if not eps:
        print(f"Nema epizoda u {raw}")
        return
    print(f"Nadjeno {len(eps)} epizoda. Rep se {'ODSECA' if args.trim_tail else 'ZADRZAVA'}.")

    if out.exists():
        if args.overwrite:
            shutil.rmtree(out)
        else:
            print(f"{out} vec postoji. Dodaj --overwrite ili izaberi drugi --out.")
            return

    df0 = pd.read_csv(eps[0] / "data.csv")
    h, w, _ = load_image(eps[0] / df0["image_path"].iloc[0], resize_wh).shape
    img_dtype = "image" if args.images else "video"
    print(f"Rezolucija: {w}x{h} (WxH) | slike kao: {img_dtype}"
          + ("" if args.images else f" ({args.codec})"))

    features = {
        "observation.images.zivid": {
            "dtype": img_dtype,
            "shape": (h, w, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.state": {
            "dtype": "float32", "shape": (9,),
            "names": ["j1", "j2", "j3", "j4", "j5", "j6", "x", "y", "rz"],
        },
        "action": {
            "dtype": "float32", "shape": (6,),
            "names": ["j1", "j2", "j3", "j4", "j5", "j6"],
        },
    }

    create_kwargs = dict(
        repo_id=args.repo_id, fps=args.fps, features=features,
        root=out, robot_type="fanuc_crx10ia", use_videos=not args.images,
    )
    if not args.images:
        create_kwargs["rgb_encoder"] = RGBEncoderConfig(vcodec=args.codec)

    ds = LeRobotDataset.create(**create_kwargs)

    total_in, total_out = 0, 0
    for ep in eps:
        df = pd.read_csv(ep / "data.csv")
        task = (ep / "task.txt").read_text().strip()
        n = len(df)
        total_in += n
        if n < 3:
            print(f"  {ep.name}: samo {n} frejmova, preskacem.")
            continue

        js = df[JOINTS].to_numpy(dtype=np.float32)
        cart = df[CART].to_numpy(dtype=np.float32)
        action_all = js[1:]  # js[t+1]; poslednji frejm (n-1) nema akciju -> nikad se ne koristi

        mask = moving_mask(js, args.idle_thr)[: n - 1]
        idx = np.where(mask)[0]
        if len(idx) == 0:
            print(f"  {ep.name}: sve idle, preskacem.")
            continue
        a = int(idx[0])                              # odseci mrtav pocetak (pre prvog pokreta)
        # zadrzi rep do poslednjeg frejma koji IMA akciju (n-2); osim ako --trim-tail
        b = int(idx[-1]) if args.trim_tail else (n - 2)

        for t in range(a, b + 1):
            img = load_image(ep / df["image_path"].iloc[t], resize_wh)
            state = np.concatenate([js[t], cart[t]]).astype(np.float32)
            act = action_all[t].astype(np.float32)
            ds.add_frame({
                "observation.images.zivid": img,
                "observation.state": state,
                "action": act,
                "task": task,
            })
        ds.save_episode()
        kept = b - a + 1
        total_out += kept
        tail = "rep odsecen" if args.trim_tail else "rep zadrzan"
        print(f"  {ep.name}: {n} -> {kept} frejmova (odseceno {n - kept}, {tail}), task='{task}'")

    ds.finalize()
    print(f"\nGotovo. Ukupno {total_in} -> {total_out} frejmova. Dataset: {out}")


if __name__ == "__main__":
    main()
