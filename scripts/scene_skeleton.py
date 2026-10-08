# scene_skeleton.py  — epizodna petlja sa resetom i oracle-om (DEBUG, bez snimanja)
# IZMENE: start 60/40 (bilo gde na padu, Z/orijentacija zakljucani), swap polja 70/30,
#         task 50/50, spawn kockice van NOGO footprinta nastavka (prati tool),
#         prilaz kocki kroz plan_route (obilazak, bez bocnog zakacinjanja viljuskom).
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import numpy as np
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, VisualCuboid
from isaacsim.core.api.robots import Robot
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.numpy.rotations import rot_matrices_to_quats
from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver, ArticulationKinematicsSolver
from pxr import UsdGeom, Gf
from pxr import Usd, UsdPhysics, PhysxSchema, UsdShade, Sdf
from pxr import UsdGeom as _UG
from isaacsim.sensors.camera import Camera
import isaacsim.core.utils.numpy.rotations as rot_utils

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

# --- Nastavak: tacan transform iz GUI fajla ---
FLANGE = "/World/crx10ia/Geometry/base_link/J1_link/J2_link/J3_link/J4_link/J5_link/J6_link/flange"
NASTAVAK_USD = "/home/robot/Desktop/Tina/fanuc_sim/meshes/nastavak.usd"
container = FLANGE + "/nastavak"
add_reference_to_stage(usd_path=NASTAVAK_USD, prim_path=container)
np_prim = world.stage.GetPrimAtPath(container)

xf = UsdGeom.Xformable(np_prim)
xf.ClearXformOpOrder()
xf.AddTranslateOp().Set(Gf.Vec3d(0.08977652033997023, -0.0016507908171414232, 0.004398669810314504))
xf.AddOrientOp().Set(Gf.Quatf(0.01898506, 0.7123969, 0.70145136, -0.009805846))  # w, x, y, z
xf.AddScaleOp().Set(Gf.Vec3f(0.1, 0.1, 0.1))

# kolizija nastavka (convex decomposition - prorez ostaje otvoren)
for p in Usd.PrimRange(world.stage.GetPrimAtPath(container)):
    if p.IsA(_UG.Mesh):
        UsdPhysics.CollisionAPI.Apply(p)
        UsdPhysics.MeshCollisionAPI.Apply(p).CreateApproximationAttr().Set("convexDecomposition")
        cd = PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(p)
        cd.CreateMaxConvexHullsAttr().Set(32)
        cd.CreateHullVertexLimitAttr().Set(64)

robot = world.scene.add(Robot(prim_path="/World/crx10ia/Geometry/base_link", name="crx10ia"))

# --- Boja nastavka: cyan (nametni materijal na mesh) ---
cyan_mtl_path = "/World/Materials/cyan_nastavak"
cyan_mtl = UsdShade.Material.Define(world.stage, cyan_mtl_path)
shader = UsdShade.Shader.Define(world.stage, cyan_mtl_path + "/Shader")
shader.CreateIdAttr("UsdPreviewSurface")
shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.0, 0.75, 0.85))  # cyan
cyan_mtl.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
for p in Usd.PrimRange(world.stage.GetPrimAtPath(container)):
    if p.IsA(_UG.Mesh):
        UsdShade.MaterialBindingAPI(p).Bind(cyan_mtl)

# digni maxForce da jak drive moze da deluje
for jname in ["J1", "J2", "J3", "J4", "J5", "J6"]:
    jp = world.stage.GetPrimAtPath(f"/World/crx10ia/Physics/{jname}")
    UsdPhysics.DriveAPI.Get(jp, "angular").GetMaxForceAttr().Set(1.0e7)

# --- Kockica ---
CUBE = 0.06
cube = world.scene.add(DynamicCuboid(
    prim_path="/World/cube", name="red_cube",
    position=np.array([0.45, 0.10, CUBE/2]), size=CUBE, color=np.array([0.8, 0, 0]),
))

# --- Polja (handle-ovi, da mozemo da im menjamo pozicije po epizodi) ---
# swap 70/30: 70% plavo=CELL_A([0.83,0.05]), crveno=CELL_B([0.83,-0.16]); 30% obrnuto
DEFAULT_LAYOUT_PROB = 0.70
CELL_A = np.array([0.83,  0.05])
CELL_B = np.array([0.83, -0.16])
FIELD = np.array([0.11, 0.11, 0.001])
field_blue_prim = world.scene.add(VisualCuboid(prim_path="/World/field_blue", name="blue_field",
    position=np.array([CELL_A[0], CELL_A[1], 0.0005]), scale=FIELD, color=np.array([0.057, 0.263, 0.8])))
field_red_prim = world.scene.add(VisualCuboid(prim_path="/World/field_red", name="red_field",
    position=np.array([CELL_B[0], CELL_B[1], 0.0005]), scale=FIELD, color=np.array([1, 0.243, 0.169])))

# --- Bele ploce (podloga) + zardjala podloga ispod ---
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
RUST = np.array([0.03, 0.03, 0.035])   # skoro crno, blago hladan ton (metal)
_plate("table_rust", PAD1_X_NEAR-0.15, PAD2_X_FAR+0.15, -0.7, 0.6, 0.0002, RUST)
_plate("pad1_white", PAD1_X_NEAR, PAD1_X_FAR, PAD1_Y_RIGHT, PAD1_Y_LEFT, 0.0004, WHITE)
_plate("pad2_white", PAD2_X_NEAR, PAD2_X_FAR, PAD2_Y_RIGHT, PAD2_Y_LEFT, 0.0004, WHITE)

# --- Kamera (fiksna, eye-to-hand) ---
CAM_POS = np.array([0.92, -0.05, 1.28])
cam = Camera(
    prim_path="/World/zivid_cam",
    position=CAM_POS,
    frequency=10,
    resolution=(1224, 1024),
    orientation=rot_utils.euler_angles_to_quats(np.array([0.0, 74.5, 180.0]), degrees=True),
)
cam.initialize()
cam.set_focal_length(2.0)
cam.set_horizontal_aperture(2.0 * 2.0 * np.tan(np.deg2rad(33.9)/2))

from pxr import UsdLux
for p in world.stage.Traverse():
    if p.HasAPI(UsdLux.LightAPI):
        UsdLux.LightAPI(p).GetIntensityAttr().Set(0.0)
dome = UsdLux.DomeLight.Define(world.stage, Sdf.Path("/World/dome"))
dome.CreateIntensityAttr(500)

# --- Reset + gainovi + IK ---
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

# --- Init poza: teleport + stabilizacija ---
robot.set_joint_positions(INIT_Q)
for _ in range(60):
    controller.apply_action(ArticulationAction(joint_positions=INIT_Q))
    world.step(render=True)
q_now = robot.get_joint_positions()
print(f"[debug] init poza greska (max): {np.max(np.abs(q_now - INIT_Q)):.4f}")

cur_pos, cur_rot = art_ik.compute_end_effector_pose()
FIX_Z = float(cur_pos[2])
FIX_QUAT = rot_matrices_to_quats(cur_rot)
NOM_START_XY = cur_pos[:2].copy()   # nominalni start = FK(INIT_Q)

# --- Parametri oracle-a i uzorkovanja ---
MARGIN      = 0.03
APPROACH_DX = 0.15
CATCH_DX    = 0.08
GRASP_DY    = 0.003
STEP        = 0.004
HOLD_STEPS  = 120
TASK_RED    = "push the red cube to the red cell"
TASK_BLUE   = "push the red cube to the blue cell"
PUSH_X      = 0.77

# --- varijacija START poze (recovery) ---
START_KEEP_PROB = 0.30          # 60% INIT_Q, 40% bilo gde na padu
SPAWN_MARGIN    = 0.01

# no-go pravougaonik oko nastavka (izmereno)
NOGO_X0, NOGO_X1 = FLANGE_X - 0.054, FLANGE_X + 0.114
NOGO_Y0, NOGO_Y1 = FLANGE_Y - 0.08,  FLANGE_Y + 0.08
CUBE_X_MIN = 0.35

# footprint nastavka iz NOGO (puna velicina viljuske, rel. flanse; orijentacija zakljucana)
FORK_X0 = NOGO_X0 - FLANGE_X     # -0.054
FORK_X1 = NOGO_X1 - FLANGE_X     # +0.114 (domet napred)
FORK_Y0 = NOGO_Y0 - FLANGE_Y     # -0.08
FORK_Y1 = NOGO_Y1 - FLANGE_Y     # +0.08
CUBE_HALF = CUBE / 2.0
X_SAFE_OFF = CUBE_HALF + FORK_X1                   # iza ovoga je viljuska cista po X
Y_SAFE_OFF = CUBE_HALF + max(FORK_Y1, -FORK_Y0)    # dalje od ovoga po Y je viljuska cista
assert APPROACH_DX >= X_SAFE_OFF, (
    f"APPROACH_DX ({APPROACH_DX}) mora biti >= X_SAFE_OFF ({X_SAFE_OFF:.3f}); "
    f"digni APPROACH_DX ili smanji NOGO domet.")

# --- spawn kockice: NEPOSREDNA BLIZINA nastavka (najkriticniji slucajevi) ---
CUBE_DX_MIN, CUBE_DX_MAX = -0.064, 0.11   # x offset od centra nastavka (tool XY)
CUBE_DY_MIN, CUBE_DY_MAX =  0.09,  0.13   # |y| offset od centra; strana (+/-) bira se 50/50
PAD_RECTS = [
    (PAD1_X_NEAR+MARGIN, PAD1_X_FAR,       PAD1_Y_RIGHT+MARGIN, PAD1_Y_LEFT-MARGIN),
    (PAD2_X_NEAR,        PAD2_X_FAR-MARGIN, PAD2_Y_RIGHT+MARGIN, PAD2_Y_LEFT-MARGIN),
]

def in_nogo(x, y):
    return (NOGO_X0 <= x <= NOGO_X1) and (NOGO_Y0 <= y <= NOGO_Y1)

def _sample_pad_xy(rng):
    rects = [
        (PAD1_X_NEAR+MARGIN, PAD1_X_FAR,        PAD1_Y_RIGHT+MARGIN, PAD1_Y_LEFT-MARGIN),
        (PAD2_X_NEAR,        PAD2_X_FAR-MARGIN,  PAD2_Y_RIGHT+MARGIN, PAD2_Y_LEFT-MARGIN),
    ]
    x0, x1, y0, y1 = rects[rng.integers(len(rects))]
    return rng.uniform(x0, x1), rng.uniform(y0, y1)

def on_pad(x, y):
    for (x0, x1, y0, y1) in PAD_RECTS:
        if x0 <= x <= x1 and y0 <= y <= y1:
            return True
    return False

def sample_cube_xy(rng, tool_xy):
    # kockica u NEPOSREDNOJ BLIZINI nastavka: x offset [-0.064,+0.11], |y| offset [0.09,0.13] (strana 50/50)
    tx, ty = float(tool_xy[0]), float(tool_xy[1])
    for _ in range(500):
        dx = rng.uniform(CUBE_DX_MIN, CUBE_DX_MAX)
        side = 1.0 if rng.integers(2) == 0 else -1.0
        dy = side * rng.uniform(CUBE_DY_MIN, CUBE_DY_MAX)
        x, y = tx + dx, ty + dy
        if x < CUBE_X_MIN:           # ne preblizu baze / ostaje dohvatljivo
            continue
        if not on_pad(x, y):         # ostaje na ploci
            continue
        return np.array([x, y])
    return np.array([max(tx + CUBE_DX_MAX, CUBE_X_MIN), ty + CUBE_DY_MAX])  # fallback

def sample_start_q(rng):
    if rng.random() < START_KEEP_PROB:
        return INIT_Q.copy(), NOM_START_XY.copy()
    for _ in range(200):
        sx, sy = _sample_pad_xy(rng)
        if in_nogo(sx, sy):
            continue
        act, ok = art_ik.compute_inverse_kinematics(
            target_position=np.array([sx, sy, FIX_Z]), target_orientation=FIX_QUAT)
        if ok:
            return np.asarray(act.joint_positions, dtype=np.float64), np.array([sx, sy])
    return INIT_Q.copy(), NOM_START_XY.copy()

def plan_route(start_xy, cxy):
    # grid ruta do approach tacke iza kocke (za celu viljusku, iz NOGO)
    cx, cy = float(cxy[0]), float(cxy[1])
    sx, sy = float(start_xy[0]), float(start_xy[1])
    x_safe = cx - X_SAFE_OFF
    xa = cx - APPROACH_DX
    y_lane = cy + GRASP_DY
    if sx <= x_safe:
        return [np.array([sx, y_lane]), np.array([xa, y_lane])]   # iza po X -> poravnaj Y pa na approach
    return [np.array([xa, sy]), np.array([xa, y_lane])]           # ispred/strana -> iza u X pa poravnaj Y

def setup_episode(rng):
    # task 50/50
    task = TASK_BLUE if rng.integers(2) == 0 else TASK_RED
    # swap polja 70/30
    default_layout = rng.random() < DEFAULT_LAYOUT_PROB
    blue_pos = CELL_A if default_layout else CELL_B
    red_pos  = CELL_B if default_layout else CELL_A
    field_blue_prim.set_world_pose(position=np.array([blue_pos[0], blue_pos[1], 0.0005]))
    field_red_prim.set_world_pose(position=np.array([red_pos[0],  red_pos[1],  0.0005]))
    field_center = blue_pos if "blue" in task else red_pos
    goal_field = np.array([PUSH_X, field_center[1]])
    # start (bilo gde) -> tool XY -> kockica van footprinta -> ruta
    start_q, start_xy = sample_start_q(rng)
    robot.set_joint_positions(start_q)
    for _ in range(30):
        controller.apply_action(ArticulationAction(joint_positions=start_q))
        world.step(render=True)
    tool_xy = art_ik.compute_end_effector_pose()[0][:2]
    cxy = sample_cube_xy(rng, tool_xy)
    cube.set_world_pose(position=np.array([cxy[0], cxy[1], CUBE/2]),
                        orientation=np.array([1.0, 0.0, 0.0, 0.0]))
    cube.set_linear_velocity(np.zeros(3)); cube.set_angular_velocity(np.zeros(3))
    route = plan_route(start_xy, cxy)
    catch = np.array([cxy[0]-CATCH_DX, cxy[1]+GRASP_DY])
    goals = list(route) + [catch, goal_field]   # ruta -> catch -> guranje
    return task, cxy, start_xy, goals, default_layout

# --- Epizodna petlja (grid: waypoint-i rute -> catch -> push) ---
rng = np.random.default_rng(0)
task, cxy, start_xy, goals, layout = setup_episode(rng)
lead_xy = start_xy.copy()
gi = 0; goal_xy = goals[0].copy()
ep = 0
print(f"[epizoda {ep}] task='{task}' layout={'A' if layout else 'B(obrnuto)'} "
      f"start={np.round(start_xy,3)} kockica={np.round(cxy,3)}")

while simulation_app.is_running():
    delta = goal_xy - lead_xy
    dist = np.linalg.norm(delta)
    if dist > STEP:
        lead_xy = lead_xy + delta/dist*STEP
    else:
        lead_xy = goal_xy.copy()
        if gi < len(goals) - 1:
            gi += 1; goal_xy = goals[gi].copy()
        else:
            for _ in range(HOLD_STEPS):
                action, ok = art_ik.compute_inverse_kinematics(
                    target_position=np.array([goal_xy[0], goal_xy[1], FIX_Z]),
                    target_orientation=FIX_QUAT)
                if ok:
                    controller.apply_action(action)
                world.step(render=True)
            ep += 1
            task, cxy, start_xy, goals, layout = setup_episode(rng)
            lead_xy = start_xy.copy()
            gi = 0; goal_xy = goals[0].copy()
            print(f"[epizoda {ep}] task='{task}' layout={'A' if layout else 'B(obrnuto)'} "
                  f"start={np.round(start_xy,3)} kockica={np.round(cxy,3)}")

    target_pos = np.array([lead_xy[0], lead_xy[1], FIX_Z])
    action, ok = art_ik.compute_inverse_kinematics(target_position=target_pos, target_orientation=FIX_QUAT)
    if ok:
        controller.apply_action(action)
    world.step(render=True)

simulation_app.close()