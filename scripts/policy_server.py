#!/usr/bin/env python3
"""
policy_server.py — MODEL server (pokretati u condi, Python 3.13, gde lerobot radi).

Ucita SmolVLA jednom i sluza akcije preko localhost TCP-a. Sav lerobot/torch rad
je ovde; ROS/Zivid klijent (faza3_rollout.py) nema te zavisnosti.

Protokol (simetrican, framed):
  [4B header_len][4B blob_len][header JSON][blob bajtovi]
  zahtev "reset": {"type":"reset"}                                  -> {"status":"ok"}
  zahtev "infer": {"type":"infer","shape":[H,W,3],"state":[...9...],
                   "task":"...","robot_type":"..."} + blob=slika    -> {"status":"ok","action":[...6...],"infer_ms":..}
  greska:                                                              {"status":"error","msg":"..."}

Pokretanje:
  conda activate <tvoj_env>
  python policy_server.py --ckpt outputs/train/fanuc_push/checkpoints/030000/pretrained_model
"""
import argparse
import json
import socket
import struct
import time
import traceback

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import prepare_observation_for_inference


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = torch.device(args.device)

    print(f"[server] Ucitavam model: {args.ckpt}")
    policy_cfg = PreTrainedConfig.from_pretrained(args.ckpt)
    policy = SmolVLAPolicy.from_pretrained(args.ckpt, config=policy_cfg)
    policy.to(device)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=args.ckpt,
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )
    print(f"[server] Model spreman. CUDA dostupna: {torch.cuda.is_available()}")

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.host, args.port))
    srv.listen(1)
    print(f"[server] Slusam na {args.host}:{args.port}. Cekam klijenta (Ctrl+C za kraj)...")

    try:
        while True:
            conn, addr = srv.accept()
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print(f"[server] Klijent povezan: {addr}")
            try:
                while True:
                    header, blob = recv_msg(conn)
                    mtype = header.get("type")

                    if mtype == "reset":
                        policy.reset()
                        for p in (preprocessor, postprocessor):
                            try:
                                p.reset()
                            except Exception:
                                pass
                        send_msg(conn, {"status": "ok"})
                        print("[server] reset (nova epizoda)")
                        continue

                    if mtype == "infer":
                        t0 = time.perf_counter()
                        try:
                            img = np.frombuffer(blob, dtype=np.uint8).reshape(tuple(header["shape"])).copy()
                            state = np.array(header["state"], dtype=np.float32)
                            obs = {
                                "observation.images.camera1": img,
                                "observation.state": state,
                            }
                            with torch.inference_mode():
                                obs = prepare_observation_for_inference(
                                    obs, device, task=header["task"],
                                    robot_type=header.get("robot_type"))
                                obs = preprocessor(obs)
                                action = policy.select_action(obs)
                                action = postprocessor(action)
                            act = action.squeeze(0).cpu().numpy().astype(np.float64).tolist()
                            dt = (time.perf_counter() - t0) * 1e3
                            send_msg(conn, {"status": "ok", "action": act, "infer_ms": round(dt, 1)})
                        except Exception as e:
                            print(f"[server] GRESKA u inferenci:\n{traceback.format_exc()}")
                            send_msg(conn, {"status": "error", "msg": f"{type(e).__name__}: {e}"})
                        continue

                    send_msg(conn, {"status": "error", "msg": f"nepoznat type: {mtype}"})

            except (ConnectionError, OSError) as e:
                print(f"[server] Klijent otkacen: {e}. Cekam novog...")
            finally:
                conn.close()
    except KeyboardInterrupt:
        print("\n[server] Kraj.")
    finally:
        srv.close()


if __name__ == "__main__":
    main()
