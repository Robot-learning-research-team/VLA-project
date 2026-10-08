#!/usr/bin/env python3
"""
zivid_capture_test.py — provera da zivid-python hvata 2D i snima PNG.

Cilj: potvrditi da capture radi iz Pythona (ne samo iz Studija), izmeriti
stvarno capture vreme u Pythonu (Studio je pokazivao ~45 ms) i videti oblik
niza koji ide u dataset.

Pokretanje (u okruženju gde je instaliran zivid-python):
    python3 zivid_capture_test.py
"""

import time
import zivid
from pathlib import Path


def main():
    try:
        print("zivid-python:", zivid.__version__)
    except Exception:
        pass

    app = zivid.Application()
    camera = app.connect_camera()
    print("Kamera:", camera.info.model, "SN:", camera.info.serial_number)

    # 2D-only, brzo; iste presete kasnije učitavamo iz .yml (Parcels/ConsumerGoods Fast)
    preset_path = Path(__file__).parent / "preset_2d.yml"
    settings_2d = zivid.Settings2D.load(preset_path)

    image = None
    for i in range(5):  # prvih par su zagrevanje; gledaj ustaljeno vreme
        t0 = time.monotonic()
        frame_2d = camera.capture_2d(settings_2d)
        image = frame_2d.image_rgba_srgb()   # ako ne postoji u tvojoj SDK verziji:
                                             # probaj .image_srgb() ili pogledaj dir(frame_2d)
        rgba = image.copy_data()             # numpy (H, W, 4) uint8
        dt = (time.monotonic() - t0) * 1e3
        print(f"[{i}] capture+copy = {dt:6.1f} ms | shape={rgba.shape} dtype={rgba.dtype}")

    if image is not None:
        image.save("zivid_test.png")
        print("Snimljeno: zivid_test.png  (u dataset ide rgba[..., :3])")


if __name__ == "__main__":
    main()
