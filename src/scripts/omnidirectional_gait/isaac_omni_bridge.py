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

import sys
import math
from pathlib import Path

# Append project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

DEVICE = "cuda:0"
wp.init()
torch.set_default_device(DEVICE)

# Safe Tensor Assign Interceptor
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
                indices = tuple(
                    idx.to(dev) if isinstance(idx, torch.Tensor) and idx.device != dev
                    else torch.as_tensor(idx, device=dev) if isinstance(idx, np.ndarray)
                    else idx for idx in indices
                )
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

from silver2_constants.isaac_constants import *
from omnidirectional_gait_controller import OmnidirectionalGaitController

# --------------------------------------------------------------------------
# Simulation & Benchmark Parameters
# --------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve()
PROJECT_ROOT = next((p for p in SCRIPT_DIR.parents if (p / "src").is_dir()), SCRIPT_DIR.parents[2])

STAGE_NAME = "silver2_isaac_sim_locomotion.usd"
USD_REL_PATH = Path("src/scenes/newton") / STAGE_NAME
USD_PATH = str((PROJECT_ROOT / USD_REL_PATH).resolve())
ROBOT_PRIM_PATH = "/World/SILVER2"

SIM_DT = 1.0 / 420.0
RENDER_FPS = 60.0
RENDER_INTERVAL = int(round((1.0 / RENDER_FPS) / SIM_DT))
DECIMATION_RATIO = 4
CTRL_DT = SIM_DT * DECIMATION_RATIO

TEST_DURATION_S = 6.0
NUM_LOCOMOTION_STEPS = int(round(TEST_DURATION_S / SIM_DT))
RAMP_STEPS = 200
ROBOT_WEIGHT = SILVER2_TOTAL_MASS * 9.81

CPG_STEP_HEIGHT = 0.025
CPG_GAIT_PERIOD = 1.350

# m/s
CMD_VX = 0.000
CMD_VY = 0.208
# rad/s
CMD_WZ = 0.000

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

    # Hardware Drive Polarities
    joint_signs = torch.tensor([-1.0, -1.0, 1.0] * 6, device=DEVICE, dtype=torch.float32)

    # Instantiate Omnidirectional Gait Controller
    omni_gait_controller = OmnidirectionalGaitController(
        gait_type="tripod",
        step_height=CPG_STEP_HEIGHT,
        gait_period=CPG_GAIT_PERIOD,
        device=DEVICE
    )
    
    # Native Isaac Sim Standing Pose
    standing_targets_deg = np.array(SILVER2_STANDING_ANGLES_DEG, dtype=np.float32).flatten()
    standing_targets_rad = np.deg2rad(standing_targets_deg).astype(np.float32)
    standing_targets_gpu = torch.as_tensor(standing_targets_rad, dtype=torch.float32, device=DEVICE)
    torch_standing_targets = standing_targets_gpu[canonical_to_newton].unsqueeze(0)

    # Kinematic Controller Standing Pose
    q_standing_kine = (standing_targets_gpu * joint_signs).reshape(6, 3)
    # Compute Kinematic Foot Baseline
    local_standing_feet = omni_gait_controller.batch_leg_for_kine(q_standing_kine)
    default_feet_body = omni_gait_controller.local_to_body_frame(local_standing_feet)
    omni_gait_controller.default_feet_body = default_feet_body

    # Settle the Physical Robot in Isaac Sim
    robot_view.set_joint_positions(torch_standing_targets)
    robot_view.set_joint_velocities(torch.zeros_like(torch_standing_targets))

    print("[INFO] Running 500 stance settling steps...")
    for step in range(500):
        robot_view.set_joint_position_targets(torch_standing_targets)
        world.step(render=RENDER_SIMULATION and ((step + 1) % RENDER_INTERVAL == 0))

    root_poses, _ = robot_view.get_world_poses()
    start_pos = root_poses[0].clone()
    peak_impact_n = 0.0
    phase_normalized = 0.0
    total_energy = 0.0
    tilt_samples = []

    print(f"[INFO] Running {NUM_LOCOMOTION_STEPS} active locomotion steps with Omnidirectional Kinematic Control...")
    print(f"  --> Commanded Velocity: Vx={CMD_VX:.3f} m/s | Vy={CMD_VY:.3f} m/s | Wz={CMD_WZ:.3f} rad/s")

    for step in range(NUM_LOCOMOTION_STEPS):
        # A. Kinematic Control Decimation
        if step % DECIMATION_RATIO == 0:
            ramp = min(1.0, (step + 1) / float(RAMP_STEPS))
            vx = CMD_VX * ramp
            vy = CMD_VY * ramp
            wz = CMD_WZ * ramp

            # Advance continuous normalized gait phase: d_tau = dt / T
            phase_normalized = (phase_normalized + CTRL_DT / omni_gait_controller.T) % 1.0
            equiv_t = phase_normalized * omni_gait_controller.T

            # Compute 3D Cartesian foot targets in Body Frame
            # Flip velocities to adjust for Isaac Sim coordinate system
            feet_targets_body = omni_gait_controller.compute_foot_trajectories(
                t=equiv_t, vx=vy, vy=vx, wz=wz,
                default_feet_body=default_feet_body
            )

            # Transform Body Frame -> Local Coxa Mount Frames
            feet_targets_local = omni_gait_controller.body_to_local_frame(feet_targets_body)

            # Closed-Form Inverse Kinematics
            q_kine, is_admissible = omni_gait_controller.batch_leg_inv_kine(feet_targets_local)

            if not is_admissible.all():
                if step % 200 == 0:
                    failing_legs = torch.where(~is_admissible)[0].tolist()
                    print(f"[WARN] IK workspace limit touched at step {step} on leg(s): {failing_legs}", flush=True)

            # Convert Kinematic Angles -> Newton Joint Order & Polarity
            q_kine_flat = q_kine.reshape(-1)
            q_newton_flat = q_kine_flat * joint_signs
            newton_targets = q_newton_flat[canonical_to_newton].unsqueeze(0)
            robot_view.set_joint_position_targets(newton_targets)

            # Stability Telemetry
            _, curr_rot = robot_view.get_world_poses(clone=False)
            tilt = float(2.0 * (curr_rot[0, 1]**2 + curr_rot[0, 2]**2))
            tilt_samples.append(tilt)

        # Physics Integration Step
        should_render = RENDER_SIMULATION and ((step + 1) % RENDER_INTERVAL == 0)
        world.step(render=should_render)

        # Contact Force & Mechanical Power Integration (Every Physics Sub-Step)
        q_curr = robot_view.get_joint_positions(clone=False)
        q_vels = robot_view.get_joint_velocities(clone=False)

        raw_forces = feet_view.get_net_contact_forces(clone=False)
        if raw_forces.dim() == 3:
            raw_forces = raw_forces.squeeze(0)
        continuous_forces = raw_forces / SIM_DT

        # Mechanical Power = sum(|tau * q_dot|)
        tau_efforts = torch.clamp(
            K_P * (newton_targets - q_curr) - K_D * q_vels,
            -MAX_TORQUE, MAX_TORQUE
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

    cmd_speed = math.hypot(CMD_VX, CMD_VY)
    cmd_heading_deg = float(np.rad2deg(np.arctan2(CMD_VY, CMD_VX))) if cmd_speed > 1e-4 else 0.0
    vel_tracking_ratio = (mean_planar_velocity / cmd_speed * 100.0) if cmd_speed > 1e-4 else 0.0

    # Energetics & Cost of Transport
    safe_disp = max(0.02, total_planar_dist)
    cost_of_transport = total_energy / (ROBOT_WEIGHT * safe_disp)

    print("\n" + "=" * 85, flush=True)
    print("OMNIDIRECTIONAL MODEL-BASED GAIT BENCHMARK SUMMARY", flush=True)
    print("=" * 85, flush=True)
    print(f"Velocity Command (Vx, Vy, Wz) : ({CMD_VX:.3f} m/s, {CMD_VY:.3f} m/s, {CMD_WZ:.3f} rad/s)", flush=True)
    print(f"Gait Parameters (T, Step H)   : Period={CPG_GAIT_PERIOD:.3f} s | Step H={CPG_STEP_HEIGHT * 100:.2f} cm", flush=True)
    print("-" * 85, flush=True)
    print(f"Total Planar Distance (XY)    : {total_planar_dist:.3f} m", flush=True)
    print(f"  --> Longitudinal Disp (X)   : {dx:.3f} m", flush=True)
    print(f"  --> Lateral Disp      (Y)   : {dy:.3f} m", flush=True)
    print(f"Mean Planar Velocity          : {mean_planar_velocity:.3f} m/s (Target: {cmd_speed:.3f} m/s -> {vel_tracking_ratio:.1f}%)", flush=True)
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