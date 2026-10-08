# faza1_recorder_sim.py — snimanje sim epizoda u format realnog recordera
# IZMENE: start poza varira 60% INIT / 40% nasumicno (X,Y; Z i orijentacija zakljucani),
#         polja plavo/crveno menjaju mesta 70/30, task 50/50,
#         kockica se spawnuje PRETEZNO u daljem padu (PAD2, gde su polja), uvek ISPRED.
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import csv, shutil
from pathlib import Path
import numpy as np
from PIL import Image

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

# ================= SNIMANJE =================
OUT_ROOT     = Path("/home/robot/vla_dataset_raw")
START_EP     = 212
TARGET_SAVED = 725             # test: 725 novih epizoda
FPS          = 10
STEPS_PER_RECORD = 6            # 60Hz fizika / 10Hz snimanje
TASK_RED     = "push the red cube to the red cell"
TASK_BLUE    = "push the red cube to the blue cell"
IN_FIELD_TOL = 0.055            # radijalno: kockica mora ovako blizu centra polja
STEP_MIN, STEP_MAX = 0.0007, 0.002   # nasumicna brzina po epizodi

# --- varijacija START poze ---
START_KEEP_PROB = 0.60          # 60% ostaje INIT_Q, 40% nasumicno po prostoru
START_DX, START_DY = 0.12, 0.15 # +/- opseg oko nominalnog starta (m) za onih 40%; STELUJ

# --- spawn kockice: pretezno dalji pad (PAD2) ---
FAR_PAD_PROB = 0.85             # 85% kockica u daljem padu (gde su polja), 15% blizi pad

# --- swap polja (70% podrazumevano, 30% obrnuto) ---
DEFAULT_LAYOUT_PROB = 0.50      # 50/50 levo/desno (swap plavo<->crveno)
CELL_A = np.array([0.83,  0.05])
CELL_B = np.array([0.83, -0.16])

# --- domain randomization (robusnost): nijansa kockice + osvetljenje IZA kamere ---
CUBE_R_MIN, CUBE_R_MAX = 0.60, 0.90    # crvena komponenta (nijansa)
CUBE_GB_MAX            = 0.10          # max zelena/plava (blagi ton, da ostane crveno)
DOME_I_MIN, DOME_I_MAX = 400.0, 650.0  # variranje ambijentalnog (dome) intenziteta
KEY_I_MIN,  KEY_I_MAX  = 15000.0, 40000.0  # key svetlo iza kamere; STELUJ po izgledu
# ============================================

def quat_to_euler(q):           # q=[w,x,y,z] -> (rx,ry,rz) rad
    w,x,y,z = q
    rx = np.arctan2(2*(w*x+y*z), 1-2*(x*x+y*y))
    ry = np.arcsin(np.clip(2*(w*y-z*x), -1, 1))
    rz = np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))
    return float(rx), float(ry), float(rz)

world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()

# --- Robot ---
ROBOT_USD = "/home/robot/Desktop/Tina/fanuc_sim/crx10ia/crx10ia.usda"
add_reference_to_stage(usd_path=ROBOT_USD, prim_path="/World/crx10ia")
robot_prim = world.stage.GetPrimAtPath("/World/crx10ia")
robot_prim.GetVariantSets().GetVariantSet("Physics").SetVariantSelection("physx")
robot_prim.Load()
for _ in range(3):
    simulation_app.update()

# --- Nastavak: transform iz GUI fajla ---
FLANGE = "/World/crx10ia/Geometry/base_link/J1_link/J2_link/J3_link/J4_link/J5_link/J6_link/flange"
NASTAVAK_USD = "/home/robot/Desktop/Tina/fanuc_sim/meshes/nastavak.usd"
container = FLANGE + "/nastavak"
add_reference_to_stage(usd_path=NASTAVAK_USD, prim_path=container)
xf = UsdGeom.Xformable(world.stage.GetPrimAtPath(container))
xf.ClearXformOpOrder()
xf.AddTranslateOp().Set(Gf.Vec3d(0.08977652033997023, -0.0016507908171414232, 0.004398669810314504))
xf.AddOrientOp().Set(Gf.Quatf(0.01898506, 0.7123969, 0.70145136, -0.009805846))
xf.AddScaleOp().Set(Gf.Vec3f(0.1, 0.1, 0.1))

# cyan materijal
cyan_mtl_path = "/World/Materials/cyan_nastavak"
cyan_mtl = UsdShade.Material.Define(world.stage, cyan_mtl_path)
shader = UsdShade.Shader.Define(world.stage, cyan_mtl_path + "/Shader")
shader.CreateIdAttr("UsdPreviewSurface")
shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.0, 0.75, 0.85))
cyan_mtl.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

# kolizija (convex decomposition) + cyan binding
for p in Usd.PrimRange(world.stage.GetPrimAtPath(container)):
    if p.IsA(_UG.Mesh):
        UsdPhysics.CollisionAPI.Apply(p)
        UsdPhysics.MeshCollisionAPI.Apply(p).CreateApproximationAttr().Set("convexDecomposition")
        cd = PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(p)
        cd.CreateMaxConvexHullsAttr().Set(32)
        cd.CreateHullVertexLimitAttr().Set(64)
        UsdShade.MaterialBindingAPI(p).Bind(cyan_mtl)

robot = world.scene.add(Robot(prim_path="/World/crx10ia/Geometry/base_link", name="crx10ia"))

# maxForce
for jname in ["J1","J2","J3","J4","J5","J6"]:
    jp = world.stage.GetPrimAtPath(f"/World/crx10ia/Physics/{jname}")
    UsdPhysics.DriveAPI.Get(jp, "angular").GetMaxForceAttr().Set(1.0e7)

# --- Kockica ---
CUBE = 0.06
cube = world.scene.add(DynamicCuboid(
    prim_path="/World/cube", name="red_cube",
    position=np.array([0.45, 0.10, CUBE/2]), size=CUBE, color=np.array([0.8, 0.0, 0.0]),
))

# --- Polja (0.83) --- (handle-ovi, da mozemo da im menjamo pozicije po epizodi)
FIELD = np.array([0.11, 0.11, 0.001])
field_blue_prim = world.scene.add(VisualCuboid(prim_path="/World/field_blue", name="blue_field",
    position=np.array([CELL_A[0], CELL_A[1], 0.0005]), scale=FIELD, color=np.array([0.057, 0.263, 0.8])))
field_red_prim = world.scene.add(VisualCuboid(prim_path="/World/field_red", name="red_field",
    position=np.array([CELL_B[0], CELL_B[1], 0.0005]), scale=FIELD, color=np.array([1.0, 0.243, 0.169])))

# --- Bele ploce + tamna podloga ---
def _plate(name, x0, x1, y0, y1, z, color):
    world.scene.add(VisualCuboid(prim_path=f"/World/{name}", name=name,
        position=np.array([(x0+x1)/2, (y0+y1)/2, z]),
        scale=np.array([max(x1-x0,1e-3), max(y1-y0,1e-3), 0.0008]), color=color))

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

# --- Kamera (tvoje naštelovane vrednosti) ---
CAM_POS = np.array([0.92, -0.05, 1.28])
cam = Camera(
    prim_path="/World/zivid_cam", position=CAM_POS, frequency=FPS,
    resolution=(1224, 1024),
    orientation=rot_utils.euler_angles_to_quats(np.array([0.0, 74.5, 180.0]), degrees=True),
)
cam.initialize()
cam.set_clipping_range(0.01, 1000000.0)   # near=1cm (bilo default vece), far veliko
cam.set_focal_length(2.0)
cam.set_horizontal_aperture(2.0 * 2.0 * np.tan(np.deg2rad(33.9)/2))

# --- Svetlo: ugasi default, dodaj dome ---
for p in world.stage.Traverse():
    if p.HasAPI(UsdLux.LightAPI):
        UsdLux.LightAPI(p).GetIntensityAttr().Set(0.0)
dome = UsdLux.DomeLight.Define(world.stage, Sdf.Path("/World/dome"))
dome.CreateIntensityAttr(500)

# --- Domain randomization setup ---
# materijal kockice (da menjamo nijansu po epizodi) -- bind direktno na /World/cube
cube_mtl = UsdShade.Material.Define(world.stage, "/World/Materials/cube_red")
cube_shader = UsdShade.Shader.Define(world.stage, "/World/Materials/cube_red/Shader")
cube_shader.CreateIdAttr("UsdPreviewSurface")
cube_diffuse = cube_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
cube_diffuse.Set(Gf.Vec3f(0.8, 0.0, 0.0))
cube_mtl.CreateSurfaceOutput().ConnectToSource(cube_shader.ConnectableAPI(), "surface")
UsdShade.MaterialBindingAPI(world.stage.GetPrimAtPath("/World/cube")).Bind(cube_mtl)

# key svetlo IZA KAMERE (x uvek veci od kamere), varira se po epizodi
key_light = UsdLux.SphereLight.Define(world.stage, Sdf.Path("/World/key_light"))
key_light.CreateRadiusAttr(0.15)
key_intensity = key_light.CreateIntensityAttr(25000.0)
key_translate = UsdGeom.Xformable(key_light.GetPrim()).AddTranslateOp()
key_translate.Set(Gf.Vec3d(CAM_POS[0] + 0.4, CAM_POS[1], 1.6))


# --- Reset + gainovi ---
world.reset()
n = robot.num_dof
controller = robot.get_articulation_controller()
controller.set_gains(kps=np.full(n, 1.0e8), kds=np.full(n, 1.0e6))

INIT_Q = np.array([0.9141335062791156, 0.11709020496210508, -0.9792838141852963,
                   -0.04329920432013505, -0.46853578953203856, -0.9171101214294972])

ik_solver = LulaKinematicsSolver(
    robot_description_path="/home/robot/Desktop/Tina/fanuc_sim/crx10ia_robot_description.yaml",
    urdf_path="/home/robot/Desktop/Tina/fanuc_sim/crx10ia.urdf",
)
art_ik = ArticulationKinematicsSolver(robot, ik_solver, "flange")
base_t, base_r = robot.get_world_pose()
ik_solver.set_robot_base_pose(base_t, base_r)

robot.set_joint_positions(INIT_Q)
for _ in range(60):
    controller.apply_action(ArticulationAction(joint_positions=INIT_Q))
    world.step(render=True)
print(f"[debug] init poza greska: {np.max(np.abs(robot.get_joint_positions()-INIT_Q)):.4f}")

cur_pos, cur_rot = art_ik.compute_end_effector_pose()
FIX_Z = float(cur_pos[2])
FIX_QUAT = rot_matrices_to_quats(cur_rot)
RX, RY, RZ = quat_to_euler(FIX_QUAT)
NOM_START_XY = cur_pos[:2].copy()   # nominalni start = FK(INIT_Q)

# --- Parametri oracle-a ---
MARGIN, APPROACH_DX, GRASP_DY = 0.03, 0.15, 0.003
FORK_GAP = 0.024                 # zazor viljuske do kockice
CATCH_DX = CUBE / 2 + FORK_GAP   # = 0.054: flansa staje na x_cube - 0.03 - 0.024 (viljuska nalegne na kockicu)
PUSH_X = 0.77
HOLD_STEPS = 120
CUBE_X_MIN = 0.42
# uspeh = kockica (skoro) CELA u polju (box kontejnment po OBE ose), ne samo centar blizu
CONTAIN_SLACK = 0.005                                   # koliko sme da viri iz polja (m); 0 = mora cela
CONTAIN_TOL   = FIELD[0] / 2 - CUBE / 2 + CONTAIN_SLACK  # ~0.03: dozvoljeno odstupanje centra po osi
NOGO_X0, NOGO_X1 = FLANGE_X - 0.054, FLANGE_X + 0.114
NOGO_Y0, NOGO_Y1 = FLANGE_Y - 0.08,  FLANGE_Y + 0.08
# ekstremni y: levo od gornjeg polja (y=0.05) i desno od donjeg (y=-0.16) -- najtezi slucajevi
EXTREME_Y_PROB = 0.5            # u daljem padu: pola slucajeva ekstremni y
EXTREME_GAP    = 0.09           # koliko iza ivice polja pocinje "ekstrem"
Y_FIELD_HI = max(CELL_A[1], CELL_B[1])   # 0.05 (gornje/levo polje)
Y_FIELD_LO = min(CELL_A[1], CELL_B[1])   # -0.16 (donje/desno polje)

HEADER = (["frame","t_img","t_js","t_cp","t_cmd"] + [f"js_{i}" for i in range(6)]
          + [f"cart_{a}" for a in ["x","y","z","rx","ry","rz"]]
          + [f"cmd_{i}" for i in range(6)] + ["image_path"])

def in_nogo(x, y):
    return (NOGO_X0 <= x <= NOGO_X1) and (NOGO_Y0 <= y <= NOGO_Y1)

def sample_cube_xy(rng):
    # PRETEZNO dalji pad (PAD2). U daljem padu pristrasnost ka EKSTREMNIM y:
    # levo od gornjeg polja i desno od donjeg -- najtezi slucajevi. Uvek ispred (x>=CUBE_X_MIN).
    for _ in range(500):
        if rng.random() < FAR_PAD_PROB:
            x = rng.uniform(PAD2_X_NEAR, PAD2_X_FAR - MARGIN)
            if rng.random() < EXTREME_Y_PROB:
                if rng.integers(2) == 0:
                    y = rng.uniform(Y_FIELD_HI + EXTREME_GAP, PAD2_Y_LEFT - MARGIN)   # levo od levog polja
                else:
                    y = rng.uniform(PAD2_Y_RIGHT + MARGIN, Y_FIELD_LO - EXTREME_GAP)  # desno od desnog polja
            else:
                y = rng.uniform(PAD2_Y_RIGHT + MARGIN, PAD2_Y_LEFT - MARGIN)
        else:
            x = rng.uniform(PAD1_X_NEAR + MARGIN, PAD1_X_FAR)
            y = rng.uniform(PAD1_Y_RIGHT + MARGIN, PAD1_Y_LEFT - MARGIN)
        if x < CUBE_X_MIN: continue
        if in_nogo(x, y): continue
        return np.array([x, y])
    return np.array([0.70, 0.0])   # fallback (dalji pad)

def sample_start_q(rng, cxy):
    # 60% nominalni INIT_Q; 40% nasumican X,Y (Z i orijentacija zakljucani) preko IK
    if rng.random() < START_KEEP_PROB:
        return INIT_Q.copy(), NOM_START_XY.copy()
    for _ in range(100):
        sx = NOM_START_XY[0] + rng.uniform(-START_DX, START_DX)
        sy = NOM_START_XY[1] + rng.uniform(-START_DY, START_DY)
        if in_nogo(sx, sy):
            continue
        if np.linalg.norm(np.array([sx, sy]) - cxy) < 0.12:   # ne startuj na kockici
            continue
        act, ok = art_ik.compute_inverse_kinematics(
            target_position=np.array([sx, sy, FIX_Z]), target_orientation=FIX_QUAT)
        if ok:
            return np.asarray(act.joint_positions, dtype=np.float64), np.array([sx, sy])
    return INIT_Q.copy(), NOM_START_XY.copy()   # fallback ako IK ne nadje resenje

def reset_episode(rng):
    cxy = sample_cube_xy(rng)
    cube.set_world_pose(position=np.array([cxy[0], cxy[1], CUBE/2]),
                        orientation=np.array([1.0,0.0,0.0,0.0]))
    cube.set_linear_velocity(np.zeros(3)); cube.set_angular_velocity(np.zeros(3))
    start_q, start_xy = sample_start_q(rng, cxy)
    robot.set_joint_positions(start_q)
    for _ in range(30):
        controller.apply_action(ArticulationAction(joint_positions=start_q))
        world.step(render=True)
    approach = np.array([cxy[0]-APPROACH_DX, cxy[1]+GRASP_DY])
    catch    = np.array([cxy[0]-CATCH_DX,    cxy[1]+GRASP_DY])
    return cxy, approach, catch, start_xy

def run_episode(ep_idx, task, step_size, approach_xy, catch_xy, goal_field, field_center, start_xy):
    ep_dir = OUT_ROOT / f"episode_{ep_idx:06d}"
    (ep_dir / "images").mkdir(parents=True, exist_ok=True)
    (ep_dir / "task.txt").write_text(task + "\n")
    f = open(ep_dir / "data.csv", "w", newline="")
    writer = csv.writer(f); writer.writerow(HEADER)

    lead = start_xy.copy()      # oracle krece od stvarnog starta ruke
    st = {"frame": 0, "last_action": None, "rc": 0}

    def record():
        i = st["frame"]; t = i * (1.0/FPS)
        js = robot.get_joint_positions()
        pos, _ = art_ik.compute_end_effector_pose()
        REAL_RZ = -2.375018358230591   # konstantan rz iz realnog dataseta
        cart = [float(pos[0])*1000.0, float(pos[1])*1000.0, float(pos[2])*1000.0, RX, RY, REAL_RZ]
        la = st["last_action"]
        cmd = list(la.joint_positions) if (la is not None and la.joint_positions is not None) else list(js)
        rgb = cam.get_rgba()[:, :, :3].astype(np.uint8)
        fname = f"frame_{i:06d}.png"
        Image.fromarray(rgb).save(str(ep_dir / "images" / fname))
        writer.writerow([i, t, t, t, t] + [float(v) for v in js] + cart
                        + [float(v) for v in cmd] + [f"images/{fname}"])
        st["frame"] = i + 1

    def one_step(txy):
        target = np.array([txy[0], txy[1], FIX_Z])
        action, ok = art_ik.compute_inverse_kinematics(target_position=target, target_orientation=FIX_QUAT)
        if ok:
            controller.apply_action(action); st["last_action"] = action
        world.step(render=True)
        st["rc"] += 1
        if st["rc"] % STEPS_PER_RECORD == 0:
            record()

    def move_to(goal):
        nonlocal lead
        guard = 0
        while guard < 200000:
            guard += 1
            delta = goal - lead; d = np.linalg.norm(delta)
            if d > step_size:
                lead = lead + delta/d*step_size
            else:
                lead = goal.copy(); one_step(lead); break
            one_step(lead)

    move_to(approach_xy)
    move_to(catch_xy)
    move_to(goal_field)
    for _ in range(HOLD_STEPS):
        one_step(goal_field)

    f.close()
    cube_xy = np.array(cube.get_world_pose()[0][:2])
    dx = abs(float(cube_xy[0]) - float(field_center[0]))
    dy = abs(float(cube_xy[1]) - float(field_center[1]))
    inside = (dx <= CONTAIN_TOL) and (dy <= CONTAIN_TOL)   # kockica (skoro) cela u polju
    return ep_dir, inside, dx, dy, st["frame"]

# --- auto-numeracija: nastavi od poslednje snimljene epizode + 1 (NE gazi postojece) ---
def next_episode_index(root, floor):
    # nadji MAKSIMALAN broj medju svim episode_* folderima pa kreni od max+1 (prazno -> floor)
    root = Path(root)
    mx = floor - 1
    if root.exists():
        for d in root.glob("episode_*"):
            try:
                mx = max(mx, int(d.name.split("_")[1]))
            except (IndexError, ValueError):
                continue
    return mx + 1

# --- Glavni loop: TARGET_SAVED novih, oba taska, swap polja 70/30 ---
OUT_ROOT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(43)
saved = 0
ep_idx = next_episode_index(OUT_ROOT, START_EP)
print(f"Snimam u {OUT_ROOT}, pocinjem od episode_{ep_idx:06d}, cilj {TARGET_SAVED} novih.")

while saved < TARGET_SAVED and simulation_app.is_running():
    task = TASK_BLUE if rng.integers(2) == 0 else TASK_RED

    # --- domain randomization po epizodi ---
    cube_diffuse.Set(Gf.Vec3f(float(rng.uniform(CUBE_R_MIN, CUBE_R_MAX)),
                              float(rng.uniform(0.0, CUBE_GB_MAX)),
                              float(rng.uniform(0.0, CUBE_GB_MAX))))
    dome.GetIntensityAttr().Set(float(rng.uniform(DOME_I_MIN, DOME_I_MAX)))
    key_intensity.Set(float(rng.uniform(KEY_I_MIN, KEY_I_MAX)))
    key_translate.Set(Gf.Vec3d(float(CAM_POS[0] + rng.uniform(0.2, 0.6)),   # x UVEK > kamera
                               float(CAM_POS[1] + rng.uniform(-0.3, 0.3)),
                               float(rng.uniform(1.4, 1.9))))

    # swap polja: 50/50 plavo/crveno levo/desno
    default_layout = rng.random() < DEFAULT_LAYOUT_PROB
    blue_pos = CELL_A if default_layout else CELL_B
    red_pos  = CELL_B if default_layout else CELL_A
    field_blue_prim.set_world_pose(position=np.array([blue_pos[0], blue_pos[1], 0.0005]))
    field_red_prim.set_world_pose(position=np.array([red_pos[0],  red_pos[1],  0.0005]))

    field_center = blue_pos if "blue" in task else red_pos
    goal_field = np.array([PUSH_X, field_center[1]])
    step_size = rng.uniform(STEP_MIN, STEP_MAX)
    while (OUT_ROOT / f"episode_{ep_idx:06d}").exists():   # NIKAD ne gazi postojecu epizodu
        ep_idx += 1
    cxy, approach_xy, catch_xy, start_xy = reset_episode(rng)
    ep_dir, inside, dx, dy, nframes = run_episode(ep_idx, task, step_size, approach_xy, catch_xy,
                                                  goal_field, field_center, start_xy)
    if inside and nframes >= 3:
        saved += 1
        layout = "A" if default_layout else "B(obrnuto)"
        print(f"[SACUVANO {saved}/{TARGET_SAVED}] episode_{ep_idx:06d} task='{task}' "
              f"layout={layout} start={np.round(start_xy,3)} kockica={np.round(cxy,3)} "
              f"dx={dx*100:.1f}cm dy={dy*100:.1f}cm frejmova={nframes}")
        ep_idx += 1
    else:
        shutil.rmtree(ep_dir, ignore_errors=True)
        print(f"[ODBACENO] dx={dx*100:.1f}cm dy={dy*100:.1f}cm (kockica nije cela u polju)")

print(f"\nGotovo. Sacuvano {saved} epizoda u {OUT_ROOT}.")
simulation_app.close()
