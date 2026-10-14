RENDER_SIMULATION = True

import isaacsim
from isaacsim.simulation_app import SimulationApp

config = {
    "headless": not RENDER_SIMULATION,
    "physics_engine": "Newton",
}
simulation_app = SimulationApp(config)

# Enable verified Newton extensions
from isaacsim.core.utils.extensions import enable_extension
enable_extension("omni.usd.schema.newton")
enable_extension("isaacsim.physics.newton")
enable_extension("isaacsim.physics.newton.tensors")

import torch
import numpy as np
import omni
import warp as wp
from pxr import Usd, UsdPhysics, Gf, Sdf

DEVICE = "cuda:0"
wp.init()
torch.set_default_device(DEVICE)

# Cross-Device Safe Tensor Assign Interceptor
import isaacsim.core.utils.torch.tensor as torch_tensor_utils
import isaacsim.core.utils.torch as torch_utils

_orig_assign = torch_tensor_utils.assign

def _safe_assign(dst, src, indices):
    """Ensures src and indices match dst device before in-place slice assignment."""
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
from isaacsim.core.prims import Articulation, RigidPrim
from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.simulation_manager import SimulationManager
import isaacsim.physics.newton as newton_ext

import sys
from pathlib import Path

# Resolve path to 'src/scripts' (2 levels up from 'newton/')
SCRIPTS_DIR = Path(__file__).resolve().parents[2]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from silver2_constants.isaac_constants import *
from warp_cpg_controller import WarpHexapodCPGController

# Path Resolution & Simulation Parameters
SCRIPT_DIR = Path(__file__).resolve()
PROJECT_ROOT = next((p for p in SCRIPT_DIR.parents if (p / "src").is_dir()), SCRIPT_DIR.parents[2])

STAGE_NAME = "silver2_isaac_sim_locomotion.usd"
USD_REL_PATH = Path("src/scenes/newton") / STAGE_NAME
USD_PATH = str((PROJECT_ROOT / USD_REL_PATH).resolve())
ROBOT_PRIM_PATH = "/World/SILVER2"

SIM_DT = 1 / 420.0
RENDER_FPS = 60.0
RENDER_INTERVAL = int(round((1.0 / RENDER_FPS) / SIM_DT))
DECIMATION_RATIO = 4
CPG_DT = SIM_DT * DECIMATION_RATIO

# Benchmark Reference Speed (CPG Champion Nominal: Front = 0.238 m/s Right = 0.201 m/s )
TARGET_SPEED = 0.201
STRIDE_FWD_CMD = 0.005
STRIDE_LAT_CMD = 0.010
WALKING_DIRECTION = SILVER2_DIRECTION_MAP["right"]

TEST_DURATION_S = 6.0
NUM_LOCOMOTION_STEPS = int(round(TEST_DURATION_S / SIM_DT))
RAMP_STEPS = 200
ROBOT_WEIGHT = SILVER2_TOTAL_MASS * 9.81

def audit_and_repair_joints(stage, robot_root_path=ROBOT_PRIM_PATH):
    """Explicit UsdPhysics.DriveAPI properties for all joints in the Hexapod Robot."""
    robot_prim = stage.GetPrimAtPath(robot_root_path)
    if not robot_prim.IsValid():
        raise RuntimeError(f"Robot root prim not found at {robot_root_path}")
    
    for prim in Usd.PrimRange(robot_prim):
        # Configure Joint Drives
        if prim.IsA(UsdPhysics.Joint):
            drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
            drive.CreateTypeAttr().Set("force")
            drive.CreateStiffnessAttr().Set(K_P)
            drive.CreateDampingAttr().Set(K_D)
            drive.CreateMaxForceAttr().Set(MAX_TORQUE)

            rev = UsdPhysics.RevoluteJoint(prim)
            rev.CreateLowerLimitAttr().Set(-180.0)
            rev.CreateUpperLimitAttr().Set(180.0)

        # Enable contact reporting on all Tibia links
        if prim.HasAPI(UsdPhysics.CollisionAPI) or prim.HasAPI(UsdPhysics.RigidBodyAPI):
            prim_name = prim.GetName().lower()
            if "tibia" in prim_name or "foot" in prim_name:
                if not prim.HasAttribute("physics:contactReport"):
                    prim.CreateAttribute("physics:contactReport", Sdf.ValueTypeNames.Bool).Set(True)
                else:
                    prim.GetAttribute("physics:contactReport").Set(True)

def setup_newton_scene(stage, physics_scene_path="/physicsScene"):
    """Configures USD PhysicsScene prim attributes for Newton GPU execution."""
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
        
def main():
    usd_context = omni.usd.get_context()
    current_stage_url = usd_context.get_stage_url() or ""
    if STAGE_NAME not in current_stage_url:
        print(f"[INFO] Loading stage into Newton environment: {USD_REL_PATH}")
        usd_context.open_stage(USD_PATH)

    stage = get_current_stage()
    audit_and_repair_joints(stage, robot_root_path=ROBOT_PRIM_PATH)
    setup_newton_scene(stage, "/physicsScene")
    SimulationManager.switch_physics_engine("newton")

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

    ns = newton_ext.acquire_stage()
    if ns is not None:
        ns.cfg.solver_cfg.nconmax = 1000
        ns.cfg.solver_cfg.njmax = 1200
        if hasattr(ns.cfg.solver_cfg, "num_iterations"):
            ns.cfg.solver_cfg.num_iterations = 32
        if hasattr(ns.cfg.solver_cfg, "num_ls_iterations"):
            ns.cfg.solver_cfg.num_ls_iterations = 6

    # Identify Tibia prim paths
    tibia_prim_paths = [f"{ROBOT_PRIM_PATH}/Tibia_{i}" for i in range(6)]
    feet_view = RigidPrim(
        prim_paths_expr=tibia_prim_paths,
        name="silver2_feet_contact_view",
        track_contact_forces=True,
        prepare_contact_sensors=True
    )
    world.scene.add(feet_view)

    # Bind Articualtion
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

    world.reset()
    if not world.is_playing():
        world.play()

    # Enforce verified drive gains across all DOFs in memory
    joint_kps = torch.full((1, 18), K_P, device=DEVICE)
    joint_kds = torch.full((1, 18), K_D, device=DEVICE)
    try:
        robot_view.set_gains(kps=joint_kps, kds=joint_kds)
    except Exception:
        pass

    # Hardware Abstraction Mapping Setup
    canonical_joint_names = []
    view_paths = feet_view.prim_paths
    canonical_foot_indices = []

    for leg_idx in range(6):
        canonical_joint_names.append(f"coxa_joint_{leg_idx}")
        canonical_joint_names.append(f"femur_joint_{leg_idx}")
        canonical_joint_names.append(f"tibia_joint_{leg_idx}")

        for view_idx, path in enumerate(view_paths):
            path_lower = path.lower()
            if f"_{leg_idx}" in path_lower or f"tibia{leg_idx}" in path_lower:
                canonical_foot_indices.append(view_idx)
                break
            
    assert len(canonical_foot_indices) == 6, (
        f"Failed to match all 6 feet! Found {len(canonical_foot_indices)} in {view_paths}"
    )
    feet_to_canonical = torch.tensor(canonical_foot_indices, dtype=torch.long, device=DEVICE)

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

    # Instantiate Warp CPG Controller
    cpg_controller = WarpHexapodCPGController(
        leg_mounts=SILVER2_MOUNTS,
        link_lengths=SILVER2_LINKS,
        dt=CPG_DT,
        total_period=1.200,
        gait="tripod",
        device=DEVICE
    )

    # Optimized Gains for { STRIDE_FWD_CMD = 0.010, STRIDE_LAT_CMD = 0.005 } 
    """
    cpg_controller.l2 = 0.00459
    cpg_controller.k1 = 0.925
    cpg_controller.k2 = 0.0134
    cpg_controller.k3 = 1.0111
    cpg_controller.epsilon = 0.5005
    cpg_controller.total_period = 1.348
    cpg_controller.coupling_strength = 0.1522
    cpg_controller.b = 3.146
    cpg_controller.f_touch = 43.25
    cpg_controller.f_release = 13.81
    """

    # Optimized Gains for { STRIDE_FWD_CMD = 0.005, STRIDE_LAT_CMD = 0.010 }
    cpg_controller.l2 = 0.00224
    cpg_controller.k1 = 0.6437
    cpg_controller.k2 = 0.0611
    cpg_controller.k3 = 0.9433
    cpg_controller.epsilon = 0.500
    cpg_controller.total_period = 1.200
    cpg_controller.coupling_strength = 0.4048
    cpg_controller.b = 3.359
    cpg_controller.f_touch = 45.68
    cpg_controller.f_release = 6.87
    
    # Dynamically-computed standing pose
    standing_targets_deg = np.array(SILVER2_STANDING_ANGLES_DEG, dtype=np.float32).flatten()
    standing_targets_rad = np.deg2rad(standing_targets_deg).astype(np.float32)
    standing_targets_gpu = torch.as_tensor(standing_targets_rad, dtype=torch.float32, device=DEVICE)
    torch_standing_targets = standing_targets_gpu[canonical_to_newton].unsqueeze(0)

    robot_view.set_joint_positions(torch_standing_targets)
    robot_view.set_joint_velocities(torch.zeros_like(torch_standing_targets))

    print("[INFO] Running 500 stance settling steps...")
    for step in range(500):
        robot_view.set_joint_position_targets(torch_standing_targets)
        world.step(render=RENDER_SIMULATION and ((step + 1) % RENDER_INTERVAL == 0))

    # Compute empirical foot baseline directly from settled physics
    calibrated_feet_body = cpg_controller.compute_default_feet_body(SILVER2_STANDING_ANGLES_DEG)

    cmd_heading_deg = float(np.rad2deg(WALKING_DIRECTION))

    root_poses, _ = robot_view.get_world_poses()
    start_pos = root_poses[0].clone()
    peak_impact_n = 0.0
    total_energy = 0.0
    tilt_samples = []

    # Filter buffers (6 feet, 3 axes)
    accumulated_forces = torch.zeros((6, 3), device=DEVICE)
    filtered_forces = torch.zeros((6, 3), device=DEVICE)

    print(f"[INFO] Running {NUM_LOCOMOTION_STEPS} active locomotion steps with sensory phase resetting...")
    for step in range(NUM_LOCOMOTION_STEPS):
        # A. CPG Control Step (Decimated)
        if step % DECIMATION_RATIO == 0:
            ramp = min(1.0, (step + 1) / float(RAMP_STEPS))

            # Apply moving average filter across sub-steps
            if step > 0:
                mean_forces = accumulated_forces / float(DECIMATION_RATIO)
                accumulated_forces.zero_()
                filtered_forces = 0.7 * filtered_forces + 0.3 * mean_forces[feet_to_canonical]

            contact_forces_wp = wp.from_torch(filtered_forces.contiguous(), dtype=wp.vec3)
            warp_targets = cpg_controller.compute_joint_targets(
                default_feet_pos_body=calibrated_feet_body,
                dir_angle_rad=WALKING_DIRECTION,
                stride_forward=STRIDE_FWD_CMD,
                stride_lateral=STRIDE_LAT_CMD,
                yaw_rate=0.0,
                yaw_gain=0.0,
                ramp=ramp,
                contact_forces=contact_forces_wp
            )

            torch_targets = wp.to_torch(warp_targets).reshape(-1)
            newton_targets = torch_targets[canonical_to_newton].unsqueeze(0)
            robot_view.set_joint_position_targets(newton_targets)

            # Stability Telemetry
            _, curr_rot = robot_view.get_world_poses(clone=False)
            tilt = float(2.0 * (curr_rot[0, 1]**2 + curr_rot[0, 2]**2))
            tilt_samples.append(tilt)

        # Physics Integration Step
        should_render = RENDER_SIMULATION and ((step + 1) % RENDER_INTERVAL == 0)
        world.step(render=should_render)

        # Single Unified State & Force Query per Sub-Step
        q_curr = robot_view.get_joint_positions(clone=False)
        q_vels = robot_view.get_joint_velocities(clone=False)

        # Single Contact Query per Sub-step
        raw_forces = feet_view.get_net_contact_forces(clone=False)
        if raw_forces.dim() == 3:
            raw_forces = raw_forces.squeeze(0)
        continuous_forces = raw_forces / SIM_DT

        # Accumulate forces for CPG filtering
        accumulated_forces += continuous_forces

        # Continuous Work / Energy Integration
        tau_efforts = torch.clamp(
            joint_kps * (newton_targets - q_curr) - joint_kds * q_vels,
            -MAX_TORQUE,
            MAX_TORQUE
        )
        step_power = torch.sum(torch.abs(tau_efforts * q_vels)).item()
        total_energy += step_power * SIM_DT

        # Peak Impact Capture (post transient)
        if step >= 40:
            cf = continuous_forces[feet_to_canonical]
            step_peak = float(torch.norm(cf, dim=-1).max().item())
            if step_peak > peak_impact_n:
                peak_impact_n = step_peak

    # 3. Trial Results Evaluation
    curr_pos, _ = robot_view.get_world_poses()
    final_pos = curr_pos[0]
    delta_xy = final_pos[:2] - start_pos[:2]
    dx = float(delta_xy[0].item())
    dy = float(delta_xy[1].item())
    total_planar_dist = float(torch.linalg.norm(delta_xy).item())

    mean_planar_velocity = total_planar_dist / TEST_DURATION_S
    trajectory_heading_deg = float(np.rad2deg(np.arctan2(dy, dx)))
    mean_tilt = float(np.mean(tilt_samples)) if tilt_samples else 0.0
    vel_tracking_ratio = (mean_planar_velocity / TARGET_SPEED * 100.0) if TARGET_SPEED > 1e-4 else 0.0

    # Energetics & Cost of Transport
    safe_disp = max(0.02, total_planar_dist)
    cost_of_transport = total_energy / (ROBOT_WEIGHT * safe_disp)
    
    print("\n" + "=" * 85, flush=True)
    print("CPG GAIT BENCHMARK SUMMARY", flush=True)
    print("=" * 85, flush=True)
    print(f"Stride Command (Fwd / Lat)    : ({STRIDE_FWD_CMD * 100:.1f} cm, {STRIDE_LAT_CMD * 100:.1f} cm) | Dir={cmd_heading_deg:.1f}°", flush=True)
    print(f"Gait Parameters (T, Step H)   : Period={cpg_controller.total_period:.3f} s | Step H={cpg_controller.l2 * 100:.2f} cm", flush=True)
    print("-" * 85, flush=True)
    print(f"Total Planar Distance (XY)    : {total_planar_dist:.3f} m", flush=True)
    print(f"  --> Longitudinal Disp (X)   : {dx:.3f} m", flush=True)
    print(f"  --> Lateral Disp      (Y)   : {dy:.3f} m", flush=True)
    print(f"Mean Planar Velocity          : {mean_planar_velocity:.3f} m/s (Target: {TARGET_SPEED:.3f} m/s -> {vel_tracking_ratio:.1f}%)", flush=True)
    print(f"Trajectory Heading Angle      : {trajectory_heading_deg:.2f}° (Commanded: {cmd_heading_deg:.2f}°)", flush=True)
    print(f"Cost of Transport (CoT)       : {cost_of_transport:.2f}", flush=True)
    print(f"Total Mechanical Energy       : {total_energy:.2f} J", flush=True)
    print(f"Peak Impact Force             : {peak_impact_n:.1f} N", flush=True)
    print(f"Mean Chassis Tilt             : {mean_tilt:.4f}", flush=True)
    print("=" * 85 + "\n", flush=True)

    world.stop()
    simulation_app.close()

if __name__ == "__main__":
    main()