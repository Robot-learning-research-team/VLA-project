# FANUC SmolVLA — Instruction-Conditioned Cube Pushing

A Vision-Language-Action (VLA) system on a **FANUC CRX-10iA** collaborative arm that
**pushes a red cube to a target cell** named by a natural-language instruction
(e.g. `"push the red cube to the red cell"`), from a single Zivid RGB image.
The policy is **SmolVLA** (fine-tuned from `lerobot/smolvla_base`) trained through the
**LeRobot** ecosystem. Manipulation is **planar pushing** (no grasping): the tool is
held at constant height and orientation, so only X/Y vary.

This repository accompanies the accompanying report (see [`paper/`](paper/)) and documents
the full pipeline: teleoperated data collection, simulation data generation, training,
deployment and evaluation.

> Language note: this README is in English; the detailed walkthrough
> [`docs/Dokumentacija.md`](docs/Dokumentacija.md) is in Serbian.

## Contents
- [Key results](#key-results)
- [Repository structure](#repository-structure)
- [Setup (two Python environments)](#setup-two-python-environments)
- [Pipeline](#pipeline)
- [Simulation (Isaac Sim)](#simulation-isaac-sim)
- [Dataset & models](#dataset--models)
- [Hardware](#hardware)
- [Paper](#paper)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Key results

Success rate per model and instruction (success = cube fully inside the target cell;
special cases excluded). The red-cell task is the common basis across all models.

| Model | Training data | Red cell | Blue cell |
|---|---|---|---|
| 60ep  | 60 real              | 10.7% (3/28)  | — |
| 130ep | 130 real             | 35.7% (10/28) | — |
| 180ep | 180 real             | 50.0% (14/28) | 50.0% (14/28) |
| **simreal2** (co-train) | ~700 sim + ~250 real | **74.1% (43/58)** | **63.8% (37/58)** |

The main finding is that **co-training on real + simulated data forced the policy to
actively use the image instead of memorizing trajectories**, making it reactive to the
cube being moved mid-episode — behaviour the real-only models did not have. Figures and
the full analysis are in [`paper/`](paper/).

## Repository structure

```
.
├── README.md
├── LICENSE
├── .gitignore
├── requirements/
│   ├── conda-lerobot.txt       # Python 3.13 env (lerobot + torch)
│   └── venv-rollout.txt        # Python 3.10 env (rclpy + zivid)
├── paper/                      # report: LaTeX source, PDF, figures
├── docs/
│   └── Dokumentacija.md        # full pipeline guide (Serbian)
├── scripts/
│   ├── teleop_phone.py         # phone teleoperation (lock-z, lock-orientation)
│   ├── zivid_capture_test.py   # camera check / capture timing
│   ├── dry_run_logger.py       # ROS stream characterization
│   ├── faza1_recorder.py       # record raw episodes (PNG + CSV)
│   ├── faza2_convert.py        # raw -> LeRobotDataset
│   ├── loadback.py             # sanity-check a built dataset
│   ├── policy_server.py        # model server (localhost)
│   ├── faza3_rollout_ik.py     # rollout client with FK -> lock -> IK (main)
│   ├── faza3_rollout.py        # rollout client without lock (fallback)
│   └── preset_2d.yml           # Zivid 2D preset
├── sim/                        # Isaac Sim scene + demonstration generation (WIP)
├── hardware/
│   └── gripper_attachment.{stl,step}   # custom end-effector (fork-on-roller)
└── config/
    └── my_init.note.md         # note on the custom `my_init` MoveIt state (SRDF)
```

## Setup (two Python environments)

A hard version clash shapes the whole architecture and cannot be avoided with one
interpreter:
- `rclpy` (ROS2 Humble) runs **only** on system **Python 3.10**.
- `lerobot` requires **Python 3.12+**.

So there are two environments:

| Environment | Python | Contains | Used for |
|---|---|---|---|
| **conda** | 3.13 | `lerobot` (editable) + `torch` (CUDA) | conversion, training, policy server |
| **rollout venv** | 3.10 | `rclpy` (from ROS) + `zivid` + `numpy` | teleop, recording, rollout client |

```bash
# rollout venv (inherits ROS rclpy)
python3 -m venv --system-site-packages ~/venvs/rollout
source ~/venvs/rollout/bin/activate
pip install -r requirements/venv-rollout.txt   # e.g. zivid==2.17.2.*

# conda env
conda create -n lerobot python=3.13 && conda activate lerobot
pip install -r requirements/conda-lerobot.txt
```

Deployment is therefore **two processes** (model in conda, ROS/camera in venv) talking
over a localhost socket. See [`docs/Dokumentacija.md`](docs/Dokumentacija.md) for the
full environment, network and camera notes.

## Pipeline

```
[Record]            [Convert]           [Train]          [Deploy + Eval]
teleop_phone.py     faza2_convert.py    lerobot-train    policy_server.py (conda)
faza1_recorder.py   (raw -> LeRobot)    (SmolVLA FT)     faza3_rollout_ik.py (venv)
```

```bash
# 1) Record (venv) — robot stack + MoveIt and teleop_phone.py running separately
python scripts/faza1_recorder.py --task "push the red cube to the red cell"

# 2) Convert raw episodes to a LeRobotDataset (conda)
python scripts/faza2_convert.py --overwrite

# 3) Train SmolVLA (conda)
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=promaja/fanuc_push \
  --dataset.root=$HOME/vla_lerobot \
  --policy.device=cuda --batch_size=48 --steps=30000 --save_freq=5000 \
  --policy.push_to_hub=false \
  --output_dir=outputs/train/fanuc_push_XXX \
  --rename_map='{"observation.images.zivid": "observation.images.camera1"}'

# 4) Deploy / evaluate (two terminals)
# server (conda):
python scripts/policy_server.py --ckpt outputs/train/fanuc_push_XXX/checkpoints/030000/pretrained_model
# client (venv):
python scripts/faza3_rollout_ik.py --task "push the red cube to the red cell" \
  --group manipulator --ee-link tcp --base-frame base_link \
  --max-xy-step 0.025 --z-floor -112 --record-dir outputs/eval/model_XXX
```

Full details (recording tips, IK lock, evaluation protocol) are in
[`docs/Dokumentacija.md`](docs/Dokumentacija.md).

## Simulation (Isaac Sim)

To improve generalization and cheaply expand the dataset, the robot's environment was
recreated in **NVIDIA Isaac Sim** (CRX arm, cube, cells, camera) and demonstrations were
generated automatically. Roughly **700 simulated episodes** were produced and combined
with the real data (**co-training**) to train `simreal2`.

The value of the simulated data was not just quantity: it is far more **diverse** than the
real teleoperation set — it starts from many different initial robot poses and includes
**recovery** behaviour (reaching back for a missed cube). This diversity is what pushed the
policy to rely on the image rather than on memorized joint trajectories, so `simreal2`
became reactive to the cube's position.

> The Isaac Sim scene and the demonstration-generation scripts live in [`sim/`](sim/).
> *(This section and `sim/` are being finalized.)*

## Dataset & models

The dataset and trained checkpoints are **not** stored in git (large binaries). They live
on the Hugging Face Hub:
- Dataset: [`promaja/fanuc_push`](https://huggingface.co/datasets/promaja/fanuc_push)
- Model: `promaja/<model-repo>` *(TODO: fill in once pushed)*

## Hardware

- **Robot:** FANUC CRX-10iA (collaborative), controller R-30iB Mini Plus.
- **Camera:** Zivid 2+ (structured light); only 2D RGB capture is used.
- **End-effector:** a custom attachment (fork-on-roller) in [`hardware/`](hardware/).
- **Compute:** a single consumer GPU (NVIDIA RTX 4070, 12 GB).

The custom MoveIt initial state `my_init` is added to the robot's SRDF so every episode and
rollout starts from the same pose; see [`config/my_init.note.md`](config/my_init.note.md).

## Paper

The report (source and PDF) is in [`paper/`](paper/).

## Acknowledgements

- [SmolVLA](https://arxiv.org/abs/2506.01844) and the [LeRobot](https://github.com/huggingface/lerobot) ecosystem.
- Teleoperation uses our fork of [SpesRobotics/teleop]: (https://github.com/Robot-learning-research-team/teleop) (changes: planar-push locks + 10 Hz command gate)
- FANUC ROS2 control via the UofI-CDACS EtherNet/IP driver and MoveIt2.
- [NVIDIA Isaac Sim](https://developer.nvidia.com/isaac/sim).

## License

Released under the MIT License — see [`LICENSE`](LICENSE). Third-party components
(SpesRobotics/teleop, the UofI-CDACS driver, LeRobot) remain under their own licenses.
