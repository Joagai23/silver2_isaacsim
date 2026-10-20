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

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from silver2_constants.isaac_constants import *
from warp_hydrodynamics_wrapper import WarpHydrodynamicsWrapper

# Path Resolution & Simulation Parameters
SCRIPT_DIR = Path(__file__).resolve()
PROJECT_ROOT = next((p for p in SCRIPT_DIR.parents if (p / "src").is_dir()), SCRIPT_DIR.parents[2])

STAGE_NAME = "silver2_scaffolding.usd"
USD_REL_PATH = Path("src/scenes/newton") / STAGE_NAME
USD_PATH = str((PROJECT_ROOT / USD_REL_PATH).resolve())
ROBOT_PRIM_PATH = "/World/SILVER2"
JSON_CONFIG_PATH = PROJECT_ROOT / "src/scripts/hydrodynamics/silver_hydro_config.json"

SIM_DT = 1 / 420.0
RENDER_FPS = 60.0
RENDER_INTERVAL = int(round((1.0 / RENDER_FPS) / SIM_DT))

TEST_DURATION_S = 20.0
NUM_SIM_STEPS = int(round(TEST_DURATION_S / SIM_DT))
MAX_PHYS_ACCEL = 20.0

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

def get_silver2_link_paths(stage, robot_root_path=ROBOT_PRIM_PATH):
    """Discovers all 19 rigid-body link prim paths under the SILVER2 hierarchy."""
    root_prim = stage.GetPrimAtPath(robot_root_path)
    if not root_prim.IsValid():
        raise RuntimeError(f"Robot prim not found at {robot_root_path}")
    link_paths = []
    for prim in Usd.PrimRange(root_prim):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            link_paths.append(prim.GetPath().pathString)
    return link_paths

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
            rendering_dt=SIM_DT,
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

    # Discover all 19 SILVER2 links and create a RigidPrim view
    link_paths = get_silver2_link_paths(stage, ROBOT_PRIM_PATH)
    print(f"[INFO] Discovered {len(link_paths)} rigid body links for SILVER2 hydrodynamics.")

    robot_links_view = RigidPrim(
        prim_paths_expr=link_paths,
        name="silver2_hydro_links",
        reset_xform_properties=False
    )
    world.scene.add(robot_links_view)

    # Fetch per-link masses for accurate safety clamping
    link_masses = robot_links_view.get_masses().to(dtype=torch.float32).view(-1, 1)

    # Instantiate Warp Hydrodynamics Wrapper from JSON
    hydro_calculator = WarpHydrodynamicsWrapper.from_json(
        json_path=JSON_CONFIG_PATH,
        link_names_or_paths=link_paths,
        device=DEVICE
    )
    print(f"[INFO] HydrodynamicsBehavior (Warp GPU) initialized for {len(link_paths)} SILVER2 links.")

    # Identify Tibia prim paths for contact reporting
    tibia_prim_paths = [f"{ROBOT_PRIM_PATH}/Tibia_{i}" for i in range(6)]
    feet_view = RigidPrim(
        prim_paths_expr=tibia_prim_paths,
        name="silver2_feet_contact_view",
        track_contact_forces=True,
        prepare_contact_sensors=True
    )
    world.scene.add(feet_view)

    # Bind Articulation
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
        canonical_joint_names.append(f"Joint_Body_Coxa_{leg_idx}")
        canonical_joint_names.append(f"Joint_Coxa_Femur_{leg_idx}")
        canonical_joint_names.append(f"Joint_Femur_Tibia_{leg_idx}")

        for view_idx, path in enumerate(view_paths):
            path_lower = path.lower()
            if f"_{leg_idx}" in path_lower or f"tibia{leg_idx}" in path_lower:
                canonical_foot_indices.append(view_idx)
                break
            
    assert len(canonical_foot_indices) == 6, (
        f"Failed to match all 6 feet! Found {len(canonical_foot_indices)} in {view_paths}"
    )

    dof_names = list(robot_view.dof_names)
    canonical_to_newton = torch.tensor(
        [canonical_joint_names.index(name) for name in dof_names],
        dtype=torch.long,
        device=DEVICE
    )
    
    # Dynamically-computed standing pose
    standing_targets_deg = np.array(SILVER2_STANDING_ANGLES_DEG, dtype=np.float32).flatten()
    standing_targets_rad = np.deg2rad(standing_targets_deg).astype(np.float32)
    standing_targets_gpu = torch.as_tensor(standing_targets_rad, dtype=torch.float32, device=DEVICE)
    torch_standing_targets = standing_targets_gpu[canonical_to_newton].unsqueeze(0)

    robot_view.set_joint_positions(torch_standing_targets)
    robot_view.set_joint_velocities(torch.zeros_like(torch_standing_targets))

    print(f"[INFO] Running {NUM_SIM_STEPS} hydrodynamic steps...")
    last_linear_velocity = None
    last_angular_velocity = None
    z_force_history = []

    for step in range(NUM_SIM_STEPS):
        # Fetch world poses and velocities for all 19 links in parallel
        positions, orientations = robot_links_view.get_world_poses()
        velocities = robot_links_view.get_velocities()
        linear_velocity = velocities[:, :3]
        angular_velocity = velocities[:, 3:]

        if linear_velocity is None or len(linear_velocity) == 0:
            break

        # Ensure contiguous PyTorch tensors on CUDA
        positions = positions.to(dtype=torch.float32).contiguous()
        orientations = orientations.to(dtype=torch.float32).contiguous()
        linear_velocity = linear_velocity.to(dtype=torch.float32).contiguous()
        angular_velocity = angular_velocity.to(dtype=torch.float32).contiguous() 

        # Convert quaternion from Isaac Sim (w, x, y, z) to Warp (x, y, z, w)
        orientation = orientations[:, [1, 2, 3, 0]].contiguous()
        
        # Initialize velocities on first step
        if last_linear_velocity is None:
            last_linear_velocity = linear_velocity.clone()
            last_angular_velocity = angular_velocity.clone()
            linear_acceleration = torch.zeros_like(linear_velocity)
            angular_acceleration = torch.zeros_like(angular_velocity)
        else:
            linear_acceleration = torch.clamp(
                (linear_velocity - last_linear_velocity) / SIM_DT, -MAX_PHYS_ACCEL, MAX_PHYS_ACCEL
            )
            angular_acceleration = torch.clamp(
                (angular_velocity - last_angular_velocity) / SIM_DT, -MAX_PHYS_ACCEL, MAX_PHYS_ACCEL
            )

        # Run Parallel GPU Kernel across all 19 links
        (
            t_buoyancy_force,
            t_drag_force,
            t_lift_force,
            t_drag_torque,
            t_added_mass_force,
            t_added_mass_torque,
            t_center_of_buoyancy,
            t_center_of_pressure,
        ) = hydro_calculator.calculate_hydrodynamic_forces(
            positions,
            orientation,
            linear_velocity,
            angular_velocity,
            linear_acceleration,
            angular_acceleration,
        )

        # Calculate torques generated by off-center forces for each link
        t_torque_buoyancy = torch.cross(t_center_of_buoyancy - positions, t_buoyancy_force, dim=-1)
        t_torque_drag = torch.cross(t_center_of_pressure - positions, t_drag_force, dim=-1)
        t_torque_lift = torch.cross(t_center_of_pressure - positions, t_lift_force, dim=-1)

        # Safely Clamp Dynamic Forces
        dynamic_force = t_drag_force + t_lift_force + t_added_mass_force
        dynamic_torque = t_torque_drag + t_torque_lift + t_drag_torque + t_added_mass_torque
        
        max_force = link_masses * MAX_PHYS_ACCEL
        force_mag = torch.norm(dynamic_force, dim=-1, keepdim=True)
        scale = torch.clamp(max_force / (force_mag + 1e-6), max=1.0)

        net_force = t_buoyancy_force + (dynamic_force * scale)
        net_torque = t_torque_buoyancy + (dynamic_torque * scale)
        
        # Apply forces and torques to all 19 links in the simulation
        robot_links_view.apply_forces_and_torques_at_pos(
            forces=net_force.contiguous(),
            torques=net_torque.contiguous(),
            is_global=True
        )

        # Update stored velocities for the next frame
        last_linear_velocity = linear_velocity.clone()
        last_angular_velocity = angular_velocity.clone()

        # Apply locomotion standing targets to articulation joints
        robot_view.set_joint_position_targets(torch_standing_targets)

        should_render = RENDER_SIMULATION and ((step + 1) % RENDER_INTERVAL == 0)
        world.step(render=should_render)

        raw_foot_forces = feet_view.get_net_contact_forces(dt=SIM_DT)
        if raw_foot_forces is not None:
            ordered_foot_forces = raw_foot_forces[canonical_foot_indices]
            foot_fz = ordered_foot_forces[:, 2]
            z_force_history.append(foot_fz.detach().cpu().numpy())

            if (step + 1) % 420 == 0:
                total_fz = foot_fz.sum().item()
                total_weight = (link_masses.sum() * 9.81).item()
                total_buoyancy = t_buoyancy_force[:, 2].sum().item()
                expected_submerged_weight = total_weight - total_buoyancy
                fz_str = ", ".join([f"L{i}: {val:5.2f}N" for i, val in enumerate(foot_fz.tolist())])
                print(
                    f"[t={(step+1)*SIM_DT:4.1f}s] Tip Fz: [{fz_str}] | "
                    f"Sum Fz: {total_fz:6.2f}N (Expected W_sub: {expected_submerged_weight:6.2f}N)"
                )
        
    world.stop()
    simulation_app.close()

if __name__ == "__main__":
    main()