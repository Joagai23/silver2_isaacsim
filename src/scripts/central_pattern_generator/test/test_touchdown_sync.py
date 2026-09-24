"""
Independent Diagnostic Test: Touchdown Event vs. Limit Cycle State Verification (Test 4)
Quantifies phase synchronization between physical ground contact and CPG limit cycle states.
"""

RENDER_SIMULATION = True

import isaacsim
from isaacsim.simulation_app import SimulationApp

config = {
    "headless": not RENDER_SIMULATION,
    "physics_engine": "Newton",
}
simulation_app = SimulationApp(config)

from isaacsim.core.utils.extensions import enable_extension
enable_extension("omni.usd.schema.newton")
enable_extension("isaacsim.physics.newton")
enable_extension("isaacsim.physics.newton.tensors")

import torch
import numpy as np
import omni
from pxr import Usd, UsdPhysics, Gf, Sdf

DEVICE = "cuda:0"
torch.set_default_device(DEVICE)

# Cross-Device Safe Tensor Assign Interceptor
import isaacsim.core.utils.torch.tensor as torch_tensor_utils
import isaacsim.core.utils.torch as torch_utils

_orig_assign = torch_tensor_utils.assign

def _safe_assign(dst, src, indices):
    if isinstance(dst, torch.Tensor):
        dev = dst.device
        if isinstance(src, torch.Tensor):
            if src.device != dev:
                src = src.to(dev)
        elif isinstance(src, (np.ndarray, list, tuple, float, int)):
            src = torch.as_tensor(src, device=dev)

        if indices is not None:
            if isinstance(indices, (list, tuple)):
                new_indices = []
                for idx in indices:
                    if isinstance(idx, torch.Tensor):
                        new_indices.append(idx.to(dev) if idx.device != dev else idx)
                    elif isinstance(idx, np.ndarray):
                        new_indices.append(torch.as_tensor(idx, device=dev))
                    else:
                        new_indices.append(idx)
                indices = tuple(new_indices)
            elif isinstance(indices, torch.Tensor) and indices.device != dev:
                indices = indices.to(dev)

    return _orig_assign(dst, src, indices)

torch_tensor_utils.assign = _safe_assign
if hasattr(torch_utils, "assign"):
    torch_utils.assign = _safe_assign

from isaacsim.core.api.world import World
from isaacsim.core.prims import Articulation
from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.simulation_manager import SimulationManager
import isaacsim.physics.newton as newton_ext

from ..silver2_isaac_constants import *
from ..numpy_cpg_controller import NumpyHexapodCPGController

USD_PATH = "/home/jorge/Documents/Code/silver2_isaacsim/src/scenes/newton/silver2_isaac_sim_locomotion.usd"
STAGE_NAME = "silver2_isaac_sim_locomotion.usd"
ROBOT_PRIM_PATH = "/World/SILVER2"

SIM_DT = 1.0 / 420.0
RENDER_FPS = 60.0
RENDER_INTERVAL = int(round((1.0 / RENDER_FPS) / SIM_DT))
DECIMATION = 4
CPG_DT = SIM_DT * DECIMATION


def audit_and_repair_joints(stage, robot_root_path=ROBOT_PRIM_PATH):
    robot_prim = stage.GetPrimAtPath(robot_root_path)
    if not robot_prim.IsValid():
        raise RuntimeError(f"Robot root prim not found at {robot_root_path}")
    
    for prim in Usd.PrimRange(robot_prim):
        if not prim.IsA(UsdPhysics.Joint):
            continue

        joint_name = prim.GetName()
        drive_type = "tibia" if "tibia" in joint_name else "femur" if "femur" in joint_name else "coxa"
        cfg = DEFAULT_DRIVES[drive_type]

        drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
        drive.CreateTypeAttr().Set("force")
        drive.CreateStiffnessAttr().Set(cfg["stiffness"])
        drive.CreateDampingAttr().Set(cfg["damping"])
        drive.CreateMaxForceAttr().Set(cfg["max_force"])

        rev = UsdPhysics.RevoluteJoint(prim)
        rev.CreateLowerLimitAttr().Set(-180.0)
        rev.CreateUpperLimitAttr().Set(180.0)


def setup_newton_scene(stage, physics_scene_path="/physicsScene"):
    scene_prim = stage.GetPrimAtPath(physics_scene_path)
    if not scene_prim.IsValid():
        scene = UsdPhysics.Scene.Define(stage, physics_scene_path)
        scene_prim = scene.GetPrim()

    scene = UsdPhysics.Scene(scene_prim)
    scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr().Set(9.81)

    if scene_prim.HasAttribute("physics:engine"):
        scene_prim.GetAttribute("physics:engine").Set("Newton")
    else:
        scene_prim.CreateAttribute("physics:engine", Sdf.ValueTypeNames.Token).Set("Newton")

def forward_kinematics_foot(q_coxa, q_femur, q_tibia, link_lengths):
    """Computes operational-space foot position relative to leg base."""
    L1, L2, L3 = link_lengths
    c1, s1 = np.cos(q_coxa), np.sin(q_coxa)
    
    # Analytical FK matching SILVER2 coordinate frame
    r = L1 + L2 * np.cos(q_femur) + L3 * np.cos(q_femur + q_tibia - np.pi)
    z = L2 * np.sin(q_femur) + L3 * np.sin(q_femur + q_tibia - np.pi)
    x = r * c1
    y = r * s1
    return np.array([x, y, z])

def main():
    usd_context = omni.usd.get_context()
    current_stage_url = usd_context.get_stage_url() or ""

    if STAGE_NAME not in current_stage_url:
        print(f"[INFO] Loading stage: {USD_PATH}")
        usd_context.open_stage(USD_PATH)

    stage = get_current_stage()
    audit_and_repair_joints(stage, robot_root_path=ROBOT_PRIM_PATH)
    setup_newton_scene(stage, "/physicsScene")
    SimulationManager.switch_physics_engine("newton")

    ns = newton_ext.acquire_stage()
    if ns is not None:
        ns.cfg.solver_cfg.nconmax = 1000

    world = World.instance()
    if world is None:
        world = World(
            physics_dt=SIM_DT,
            rendering_dt=1.0 / RENDER_FPS,
            stage_units_in_meters=1.0,
            physics_prim_path="/physicsScene",
            backend="torch",
            device=DEVICE
        )
    world.initialize_physics()

    robot_name = "SILVER2_robot"
    if world.scene.object_exists(robot_name):
        robot_view = world.scene.get_object(robot_name)
    else:
        robot_view = Articulation(
            prim_paths_expr=ROBOT_PRIM_PATH,
            name=robot_name,
            reset_xform_properties=False
        )
        world.scene.add(robot_view)

    if not world.is_playing():
        world.play()
    world.reset()

    # Index mapping
    canonical_joint_names = []
    for leg_idx in range(6):
        canonical_joint_names.append(f"coxa_joint_{leg_idx}")
        canonical_joint_names.append(f"femur_joint_{leg_idx}")
        canonical_joint_names.append(f"tibia_joint_{leg_idx}")

    dof_names = list(robot_view.dof_names)
    canonical_to_newton = torch.tensor(
        [canonical_joint_names.index(name) for name in dof_names],
        dtype=torch.long,
        device=DEVICE
    )
    newton_to_canonical = torch.tensor(
        [dof_names.index(name) for name in canonical_joint_names],
        dtype=torch.long,
        device=DEVICE
    )

    # Controller with verified clearance (l2 = 0.010)
    cpg_controller = NumpyHexapodCPGController(
        leg_mounts=SILVER2_MOUNTS,
        link_lengths=SILVER2_LINKS,
        dt=CPG_DT,
        total_period=2.0,
        gait="tripod"
    )
    cpg_controller.l2 = 0.010

    # 1. Standing posture settle
    standing_targets_deg = np.array(SILVER2_STANDING_ANGLES_DEG, dtype=np.float32).flatten()
    standing_targets_rad = np.deg2rad(standing_targets_deg).astype(np.float32)
    standing_targets_gpu = torch.as_tensor(standing_targets_rad, dtype=torch.float32, device=DEVICE)
    torch_standing_targets = standing_targets_gpu[canonical_to_newton].unsqueeze(0)

    robot_view.set_joint_positions(torch_standing_targets)
    robot_view.set_joint_velocities(torch.zeros_like(torch_standing_targets))

    for step in range(1000):
        robot_view.set_joint_position_targets(torch_standing_targets)
        world.step(render=RENDER_SIMULATION and ((step + 1) % RENDER_INTERVAL == 0))

    # Extract true loaded joint angles (in degrees, shaped (6, 3))
    settled_q_rad = robot_view.get_joint_positions().squeeze()[newton_to_canonical].cpu().numpy()
    settled_q_deg = np.rad2deg(settled_q_rad).reshape((6, 3))

    # Compute empirical foot baseline directly from settled physics
    calibrated_feet_body = cpg_controller.compute_default_feet_body(settled_q_deg, SILVER2_MOUNTS, SILVER2_LINKS)

    print("\n[INFO] Auto-Calibrated Feet Z-Coordinates (incorporating gravity sag):")
    for i in range(6):
        print(f"  Leg {i}: Z = {calibrated_feet_body[i, 2]:.4f} m (Nominal: {SILVER2_DEFAULT_FEET_BODY[i, 2]:.4f} m)")

    # 2. Run Test 4: 3 Strides (630 CPG ticks)
    TOTAL_PERIOD = 2.0
    NUM_CYCLES = 3
    TOTAL_STEPS = int(round((NUM_CYCLES * TOTAL_PERIOD) / SIM_DT))
    NUM_CPG_TICKS = TOTAL_STEPS // DECIMATION

    print("=" * 80)
    print(f"RUNNING TEST 4: EVALUATING TOUCHDOWN TIMING ACROSS {NUM_CYCLES} CYCLES")
    print(f"Parameters: l2 = {cpg_controller.l2:.3f}, Stride Forward = 0.005 m")
    print("=" * 80)

    time_log = np.zeros(NUM_CPG_TICKS)
    y_state_log = np.zeros((NUM_CPG_TICKS, 6))
    foot_world_z_log = np.zeros((NUM_CPG_TICKS, 6))

    current_targets = torch_standing_targets
    cpg_tick = 0

    for step in range(TOTAL_STEPS):
        if step % DECIMATION == 0 and cpg_tick < NUM_CPG_TICKS:
            time_log[cpg_tick] = cpg_tick * CPG_DT

            canonical_targets = cpg_controller.compute_joint_targets(
                calibrated_feet_body,
                dir_angle_rad=0.0,
                stride_forward=0.005,
                stride_lateral=0.000,
                yaw_rate=0.0,
                ramp=1.0
            )

            # Store oscillator state y
            y_state_log[cpg_tick] = cpg_controller.y.copy()

            # Stream targets
            cpg_targets_gpu = torch.as_tensor(canonical_targets.flatten(), dtype=torch.float32, device=DEVICE)
            current_targets = cpg_targets_gpu[canonical_to_newton].unsqueeze(0)

            # Read actual joints and compute foot world Z
            curr_poses, _ = robot_view.get_world_poses()
            body_z = curr_poses[0, 2].item()
            measured_q = robot_view.get_joint_positions().squeeze()[newton_to_canonical].cpu().numpy()

            for leg_i in range(6):
                q_leg = measured_q[leg_i * 3 : leg_i * 3 + 3]
                p_foot_local = forward_kinematics_foot(q_leg[0], q_leg[1], q_leg[2], SILVER2_LINKS)
                # World Z = Body Z + Mount Z + Foot Local Z
                mount_z = SILVER2_MOUNTS[cpg_controller.leg_names[leg_i]]['pos'][2]
                foot_world_z_log[cpg_tick, leg_i] = body_z + mount_z + p_foot_local[2]

            cpg_tick += 1

        robot_view.set_joint_position_targets(current_targets)
        world.step(render=RENDER_SIMULATION and ((step + 1) % RENDER_INTERVAL == 0))

    # 3. Post-Process Touchdown Timing
    print("\n" + "=" * 90)
    print(f"{'Leg':<8} | {'Cycle #':<10} | {'Touchdown t (s)':<18} | {'y at Touchdown':<18} | {'Phase Error (deg)'} | {'Status'}")
    print("=" * 90)

    for leg_i in [0, 1]:  # Inspect representative front and middle legs
        y_series = y_state_log[:, leg_i]
        z_series = foot_world_z_log[:, leg_i]

        # Stance contact threshold (lowest 15% of vertical stroke)
        z_ground_contact = np.min(z_series) + 0.008

        # Detect swing-to-stance touchdown events (descent phase crossing ground threshold)
        for k in range(1, NUM_CPG_TICKS - 1):
            is_descending = z_series[k] < z_series[k - 1]
            crossed_to_ground = z_series[k - 1] >= z_ground_contact and z_series[k] < z_ground_contact
            
            if crossed_to_ground and is_descending:
                t_td = time_log[k]
                y_td = y_series[k]
                
                # Phase angle on limit cycle: phi = arctan2(y, x)
                # At ideal touchdown (y = 0), phase = 0 deg
                phase_error_deg = np.rad2deg(np.arcsin(np.clip(y_td / 10.0, -1.0, 1.0)))

                if abs(y_td) <= 1.0:
                    status = "EXCELLENT"
                elif y_td < -1.0:
                    status = "EARLY (Stubbing risk)"
                else:
                    status = "LATE (Lost traction)"

                print(f"Leg {leg_i:<4} | t = {t_td:6.3f} s | {t_td:14.3f} s   | y = {y_td:11.3f}   | {phase_error_deg:14.1f}°   | {status}")

    print("=" * 90 + "\n")

    world.stop()
    simulation_app.close()

if __name__ == "__main__":
    main()