# rollout_sim.py — rollout SmolVLA modela u Isaac Sim-u (izolacija model vs sim-to-real)
# Setup scene/robota/kamere/IK je 1:1 iz faza1_recorder_sim.py (dokazan).
# Razlika: umesto oracle-a, svaki tik salje (slika+stanje) policy serveru i primenjuje
# vraceni vektor od 6 zglobova direktno kao drive target. Isti protokol kao faza3_rollout_ik.py.
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import argparse, json, socket, struct, sys, time
import numpy as np

from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, VisualCuboid
from isaacsim.core.api.robots import Robot
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.numpy.rotations import rot_matrices_to_quats
import isaacsim.core.utils.numpy.rotations as rot_utils
from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver, ArticulationKinematicsSolver
from isaacsim.sensors.camera import Camera
from pxr import UsdGeom, Gf, Usd, UsdPhysics, PhysxSchema, UsdShade, Sdf, UsdLux
from pxr import UsdGeom as _UG

# ================= PARAMETRI =================
TASK_RED   = "push the red cube to the red cell"
TASK_BLUE  = "push the red cube to the blue cell"
FPS        = 10
STEPS_PER_TICK = 6           # 60Hz fizika / 10Hz kontrola (drzi akciju 6 koraka)
SUCCESS_TOL = 0.055          # radijalno < 5.5 cm od centra polja = uspeh
REAL_RZ    = -2.375018358230591   # zakucan rz iz realnog dataseta/treninga
CUBE = 0.06
ROBOT_USD    = "/home/robot/Desktop/Tina/fanuc_sim/crx10ia/crx10ia.usda"
NASTAVAK_USD = "/home/robot/Desktop/Tina/fanuc_sim/meshes/nastavak.usd"
DESC_YAML    = "/home/robot/Desktop/Tina/fanuc_sim/crx10ia_robot_description.yaml"
URDF_PATH    = "/home/robot/Desktop/Tina/fanuc_sim/crx10ia.urdf"
# ============================================

ap = argparse.ArgumentParser()
ap.add_argument("--host", default="127.0.0.1")
ap.add_argument("--port", type=int, default=5555)
ap.add_argument("--timeout", type=float, default=5.0)
ap.add_argument("--episodes", type=int, default=20)
ap.add_argument("--max-ticks", type=int, default=300)   # ~30s na 10Hz; timeout epizode
ap.add_argument("--hold-ticks", type=int, default=15,   # ~1.5s na 10Hz; koliko dugo mora da drzi
                help="uspeh = kockica ostane u polju ovoliko UZASTOPNIH tikova")
ap.add_argument("--seed", type=int, default=12345)      # dataset je bio 0
ap.add_argument("--robot-type", default="fanuc_crx10ia")
ap.add_argument("--max-joint-step", type=float, default=None,
                help="[raw] opc. sigurnosni clamp |dq| po zglobu/tik (rad). Prazno = bez.")
# --- PARNOST SA REALNIM (default): model->FK->lock Z+orijentacija->clamp XY->IK ---
ap.add_argument("--raw-joints", action="store_true",
                help="primeni 6 zglobova DIREKTNO (cista provera modela). "
                     "Default BEZ ovoga = parnost sa faza3_rollout_ik.py (FK-lock-IK).")
ap.add_argument("--max-xy-step", type=float, default=0.025,
                help="[mirror] max XY pomak po tiku (m), isto kao realni robot.")
ap.add_argument("--joint-safety-cap", type=float, default=0.15,
                help="[mirror] odbaci IK skok veci od ovoga (rad) -> drzi (anti-flip).")
ap.add_argument("--snap", action="store_true",
                help="primeni target odjednom i drzi ga STEPS_PER_TICK koraka (staro ponasanje). "
                     "Default = interpolacija do targeta preko tika, kao FANUC kontroler.")
ap.add_argument("--probe", action="store_true",
                help="dijagnostika: za zglobove iz TEST_QS ispisi SIM cart (Lula FK) i izadji. "
                     "Poredi sa probe_state_real.py za istom js.")
args = ap.parse_args()


# --------- protokol ka serveru (isti kao faza3_rollout_ik.py) ---------
def recv_all(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("veza zatvorena")
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
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.settimeout(timeout)
        self.sock.connect((host, port))
        print(f"[sim] povezan na server {host}:{port}")

    def reset(self):
        send_msg(self.sock, {"type": "reset"}); recv_msg(self.sock)

    def infer(self, img, state, task, robot_type):
        img = np.ascontiguousarray(img)
        send_msg(self.sock, {"type": "infer", "shape": list(img.shape),
                             "state": [float(x) for x in state],
                             "task": task, "robot_type": robot_type}, img.tobytes())
        h, _ = recv_msg(self.sock)
        if h.get("status") != "ok":
            raise RuntimeError(f"server greska: {h.get('msg')}")
        return np.array(h["action"], dtype=np.float64), h.get("infer_ms")


# ===================== SCENA (1:1 iz recordera) =====================
world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()

add_reference_to_stage(usd_path=ROBOT_USD, prim_path="/World/crx10ia")
robot_prim = world.stage.GetPrimAtPath("/World/crx10ia")
robot_prim.GetVariantSets().GetVariantSet("Physics").SetVariantSelection("physx")
robot_prim.Load()
for _ in range(3):
    simulation_app.update()

FLANGE = "/World/crx10ia/Geometry/base_link/J1_link/J2_link/J3_link/J4_link/J5_link/J6_link/flange"
container = FLANGE + "/nastavak"
add_reference_to_stage(usd_path=NASTAVAK_USD, prim_path=container)
xf = UsdGeom.Xformable(world.stage.GetPrimAtPath(container))
xf.ClearXformOpOrder()
xf.AddTranslateOp().Set(Gf.Vec3d(0.08977652033997023, -0.0016507908171414232, 0.004398669810314504))
xf.AddOrientOp().Set(Gf.Quatf(0.01898506, 0.7123969, 0.70145136, -0.009805846))
xf.AddScaleOp().Set(Gf.Vec3f(0.1, 0.1, 0.1))

cyan_mtl_path = "/World/Materials/cyan_nastavak"
cyan_mtl = UsdShade.Material.Define(world.stage, cyan_mtl_path)
shader = UsdShade.Shader.Define(world.stage, cyan_mtl_path + "/Shader")
shader.CreateIdAttr("UsdPreviewSurface")
shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.0, 0.75, 0.85))
cyan_mtl.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

for p in Usd.PrimRange(world.stage.GetPrimAtPath(container)):
    if p.IsA(_UG.Mesh):
        UsdPhysics.CollisionAPI.Apply(p)
        UsdPhysics.MeshCollisionAPI.Apply(p).CreateApproximationAttr().Set("convexDecomposition")
        cd = PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(p)
        cd.CreateMaxConvexHullsAttr().Set(32)
        cd.CreateHullVertexLimitAttr().Set(64)
        UsdShade.MaterialBindingAPI(p).Bind(cyan_mtl)

robot = world.scene.add(Robot(prim_path="/World/crx10ia/Geometry/base_link", name="crx10ia"))

for jname in ["J1", "J2", "J3", "J4", "J5", "J6"]:
    jp = world.stage.GetPrimAtPath(f"/World/crx10ia/Physics/{jname}")
    UsdPhysics.DriveAPI.Get(jp, "angular").GetMaxForceAttr().Set(1.0e7)

cube = world.scene.add(DynamicCuboid(
    prim_path="/World/cube", name="red_cube",
    position=np.array([0.45, 0.10, CUBE/2]), size=CUBE, color=np.array([0.8, 0.0, 0.0])))

FIELD = np.array([0.11, 0.11, 0.001])
FIELD_BLUE = np.array([0.83, -0.16])
FIELD_RED  = np.array([0.83,  0.05])
world.scene.add(VisualCuboid(prim_path="/World/field_blue", name="blue_field",
    position=np.array([FIELD_BLUE[0], FIELD_BLUE[1], 0.0005]), scale=FIELD, color=np.array([0.057, 0.263, 0.8])))
world.scene.add(VisualCuboid(prim_path="/World/field_red", name="red_field",
    position=np.array([FIELD_RED[0], FIELD_RED[1], 0.0005]), scale=FIELD, color=np.array([1.0, 0.243, 0.169])))

def _plate(name, x0, x1, y0, y1, z, color):
    world.scene.add(VisualCuboid(prim_path=f"/World/{name}", name=name,
        position=np.array([(x0+x1)/2, (y0+y1)/2, z]),
        scale=np.array([max(x1-x0, 1e-3), max(y1-y0, 1e-3), 0.0008]), color=color))

FLANGE_X, FLANGE_Y = 0.3024, 0.152
PAD1_X_NEAR = FLANGE_X - 0.024 - 0.03
PAD1_X_FAR  = PAD1_X_NEAR + 0.303
PAD1_Y_LEFT, PAD1_Y_RIGHT =  0.336, -0.298
PAD2_X_NEAR = PAD1_X_FAR
PAD2_X_FAR  = PAD2_X_NEAR + 0.36
PAD2_Y_LEFT, PAD2_Y_RIGHT =  0.336, -0.433
WHITE = np.array([0.9, 0.9, 0.88])
RUST  = np.array([0.03, 0.03, 0.035])
_plate("table_rust", PAD1_X_NEAR-0.15, PAD2_X_FAR+0.15, -0.7, 0.6, 0.0002, RUST)
_plate("pad1_white", PAD1_X_NEAR, PAD1_X_FAR, PAD1_Y_RIGHT, PAD1_Y_LEFT, 0.0004, WHITE)
_plate("pad2_white", PAD2_X_NEAR, PAD2_X_FAR, PAD2_Y_RIGHT, PAD2_Y_LEFT, 0.0004, WHITE)

CAM_POS = np.array([0.92, -0.05, 1.28])
cam = Camera(prim_path="/World/zivid_cam", position=CAM_POS, frequency=FPS,
             resolution=(1224, 1024),
             orientation=rot_utils.euler_angles_to_quats(np.array([0.0, 74.5, 180.0]), degrees=True))
cam.initialize()
cam.set_clipping_range(0.01, 1000000.0)
cam.set_focal_length(2.0)
cam.set_horizontal_aperture(2.0 * 2.0 * np.tan(np.deg2rad(33.9)/2))

for p in world.stage.Traverse():
    if p.HasAPI(UsdLux.LightAPI):
        UsdLux.LightAPI(p).GetIntensityAttr().Set(0.0)
dome = UsdLux.DomeLight.Define(world.stage, Sdf.Path("/World/dome"))
dome.CreateIntensityAttr(500)

world.reset()
n = robot.num_dof
controller = robot.get_articulation_controller()
controller.set_gains(kps=np.full(n, 1.0e8), kds=np.full(n, 1.0e6))

INIT_Q = np.array([0.9141335062791156, 0.11709020496210508, -0.9792838141852963,
                   -0.04329920432013505, -0.46853578953203856, -0.9171101214294972])

ik_solver = LulaKinematicsSolver(robot_description_path=DESC_YAML, urdf_path=URDF_PATH)
art_ik = ArticulationKinematicsSolver(robot, ik_solver, "flange")
base_t, base_r = robot.get_world_pose()
ik_solver.set_robot_base_pose(base_t, base_r)

robot.set_joint_positions(INIT_Q)
for _ in range(60):
    controller.apply_action(ArticulationAction(joint_positions=INIT_Q))
    world.step(render=True)
print(f"[sim] init poza greska: {np.max(np.abs(robot.get_joint_positions()-INIT_Q)):.4f}")

# zakljucana poza iz init poze (za mirror mod: lock Z + orijentacija)
cur_pos, cur_rot = art_ik.compute_end_effector_pose()
FIX_Z = float(cur_pos[2])
FIX_QUAT = rot_matrices_to_quats(cur_rot)

# ===================== PROBE (dijagnostika koordinatnog sistema) =====================
if args.probe:
    def dbg_state(tag, js, cx_mm, cy_mm, rz):
        print(f"[{tag}] js=[" + ", ".join(f"{v:+.4f}" for v in js) + "]  "
              f"cart_x={cx_mm:+.1f}mm  cart_y={cy_mm:+.1f}mm  rz={rz:+.5f}  "
              f"(rz-(-2.375018)={rz-(-2.375018):+.5f})", flush=True)

    # Ubaci ovde js ocitane sa realnog (probe_state_real.py). INIT_Q ostaje kao referenca.
    TEST_QS = {
        "INIT_Q": INIT_Q,
        # "poza1": np.array([ +0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000 ]),
        # "poza2": np.array([ +0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000 ]),
    }
    for name, q in TEST_QS.items():
        q = np.asarray(q, dtype=np.float64)
        robot.set_joint_positions(q)
        for _ in range(30):
            controller.apply_action(ArticulationAction(joint_positions=q))
            world.step(render=True)
        pos, _ = art_ik.compute_end_effector_pose()
        dbg_state(f"SIM {name}", robot.get_joint_positions()[:6],
                  float(pos[0]) * 1000.0, float(pos[1]) * 1000.0, -2.375018)  # rz = ono sto model DOBIJA u simu
    simulation_app.close()
    sys.exit(0)

# ===================== SAMPLING/RESET (iz recordera) =====================
MARGIN = 0.03
NOGO_X0, NOGO_X1 = FLANGE_X - 0.054, FLANGE_X + 0.114
NOGO_Y0, NOGO_Y1 = FLANGE_Y - 0.08,  FLANGE_Y + 0.08
CUBE_X_MIN = 0.42

def in_nogo(x, y):
    return (NOGO_X0 <= x <= NOGO_X1) and (NOGO_Y0 <= y <= NOGO_Y1)

def sample_cube_xy(rng):
    rects = [
        (PAD1_X_NEAR+MARGIN, PAD1_X_FAR,       PAD1_Y_RIGHT+MARGIN, PAD1_Y_LEFT-MARGIN),
        (PAD2_X_NEAR,        PAD2_X_FAR-MARGIN, PAD2_Y_RIGHT+MARGIN, PAD2_Y_LEFT-MARGIN),
    ]
    for _ in range(500):
        x0, x1, y0, y1 = rects[rng.integers(len(rects))]
        x = rng.uniform(x0, x1); y = rng.uniform(y0, y1)
        if x < CUBE_X_MIN:
            continue
        if not in_nogo(x, y):
            return np.array([x, y])
    return np.array([0.45, 0.10])

def reset_episode(rng):
    cxy = sample_cube_xy(rng)
    cube.set_world_pose(position=np.array([cxy[0], cxy[1], CUBE/2]),
                        orientation=np.array([1.0, 0.0, 0.0, 0.0]))
    cube.set_linear_velocity(np.zeros(3)); cube.set_angular_velocity(np.zeros(3))
    robot.set_joint_positions(INIT_Q)
    for _ in range(30):
        controller.apply_action(ArticulationAction(joint_positions=INIT_Q))
        world.step(render=True)
    return cxy

# ===================== ROLLOUT =====================
client = PolicyClient(args.host, args.port, args.timeout)
rng = np.random.default_rng(args.seed)

results = []
for ep in range(args.episodes):
    task = TASK_BLUE if rng.integers(2) == 0 else TASK_RED
    field_center = FIELD_BLUE if "blue" in task else FIELD_RED
    cxy = reset_episode(rng)
    client.reset()

    min_dist = float("inf")
    infer_ms_last = None
    dwell = 0              # uzastopni tikovi sa kockicom u polju
    held = False
    for tick in range(args.max_ticks):
        if not simulation_app.is_running():
            break
        rgb = cam.get_rgba()[:, :, :3].astype(np.uint8)
        js = robot.get_joint_positions()
        pos, _ = art_ik.compute_end_effector_pose()
        state = np.array([float(js[0]), float(js[1]), float(js[2]),
                          float(js[3]), float(js[4]), float(js[5]),
                          float(pos[0])*1000.0, float(pos[1])*1000.0, REAL_RZ], dtype=np.float32)

        act, infer_ms_last = client.infer(rgb, state, task, args.robot_type)

        if args.raw_joints:
            cmd = np.asarray(act, dtype=np.float64)
            if args.max_joint_step is not None:
                cmd = js + np.clip(cmd - js, -args.max_joint_step, args.max_joint_step)
        else:
            # PARNOST SA REALNIM: gde model hoce vrh (X,Y) -> FK modela
            # (pretpostavka: artikulacija je u istom redu kao Lula cspace = J1..J6;
            #  ako FK deluje pogresno, ovde je mesto da remapiras redosled zglobova)
            p_model, _ = ik_solver.compute_forward_kinematics("flange", np.asarray(act, dtype=np.float64))
            # trenutni XY iz compute_end_effector_pose (pos, u metrima) -> clamp XY koraka
            tx = float(pos[0]) + float(np.clip(p_model[0] - pos[0], -args.max_xy_step, args.max_xy_step))
            ty = float(pos[1]) + float(np.clip(p_model[1] - pos[1], -args.max_xy_step, args.max_xy_step))
            # zakljucan Z i orijentacija (iz init poze) -> IK
            sol, ok = art_ik.compute_inverse_kinematics(
                target_position=np.array([tx, ty, FIX_Z]), target_orientation=FIX_QUAT)
            if not ok:                                   # IK pao -> drzi, samo gazi vreme
                for _ in range(STEPS_PER_TICK):
                    world.step(render=True)
                continue
            cmd = np.asarray(sol.joint_positions, dtype=np.float64)   # IK vraca ArticulationAction
            if np.max(np.abs(cmd - js)) > args.joint_safety_cap:      # moguc IK flip -> drzi
                for _ in range(STEPS_PER_TICK):
                    world.step(render=True)
                continue

        # Izvrsavanje: realni FANUC kontroler interpolira do targeta preko celog tika
        # (JointTrajectory, time_from_start=period). Emuliramo to rampom js->cmd preko
        # pod-koraka, umesto snap-and-hold koji sa krutim drive-om pravi zvonjavu (jitter).
        q0 = np.asarray(js, dtype=np.float64)
        for k in range(STEPS_PER_TICK):
            if args.snap:
                q_sub = cmd
            else:
                q_sub = q0 + (cmd - q0) * ((k + 1) / STEPS_PER_TICK)
            controller.apply_action(ArticulationAction(joint_positions=q_sub))
            world.step(render=True)

        cube_xy = np.array(cube.get_world_pose()[0][:2])
        d = float(np.linalg.norm(cube_xy - field_center))
        min_dist = min(min_dist, d)
        if d < SUCCESS_TOL:
            dwell += 1
            if dwell >= args.hold_ticks:    # zadrzala se dovoljno dugo -> uspeh
                held = True
                break
        else:
            dwell = 0                       # ispala iz polja -> brojac od nule

    final_dist = float(np.linalg.norm(np.array(cube.get_world_pose()[0][:2]) - field_center))
    success = held     # uspeh = dovela I zadrzala u polju, ne samo dotakla
    results.append((success, final_dist, min_dist))
    print(f"[ep {ep:03d}] task='{task}' cube0={np.round(cxy,3)} "
          f"-> {'USPEH(drzano)' if success else 'pad          '} "
          f"final={final_dist*100:.1f}cm min={min_dist*100:.1f}cm "
          f"dwell={dwell} infer={infer_ms_last}ms")

n_ok = sum(1 for s, _, _ in results if s)
print(f"\n[sim] USPESNOST: {n_ok}/{len(results)} = {100.0*n_ok/max(len(results),1):.1f}%")
if results:
    print(f"[sim] final_dist: avg={np.mean([r[1] for r in results])*100:.1f}cm "
          f"min={np.min([r[1] for r in results])*100:.1f}cm")
    print(f"[sim] min_dist (najbliže tokom epizode): avg={np.mean([r[2] for r in results])*100:.1f}cm")

simulation_app.close()
