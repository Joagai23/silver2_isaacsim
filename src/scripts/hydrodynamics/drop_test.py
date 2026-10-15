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
import warp as wp
from pxr import UsdPhysics, Gf, Sdf
from isaacsim.core.experimental.prims import RigidPrim

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
from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.simulation_manager import SimulationManager
import isaacsim.physics.newton as newton_ext

import sys
from pathlib import Path

# Resolve path to 'src/scripts'
SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from warp_hydrodynamics_wrapper import WarpHydrodynamicsWrapper

# Path Resolution & Simulation Parameters
SCRIPT_DIR = Path(__file__).resolve()
PROJECT_ROOT = next((p for p in SCRIPT_DIR.parents if (p / "src").is_dir()), SCRIPT_DIR.parents[2])

STAGE_NAME = "hydro_cube.usd"
USD_REL_PATH = Path("src/scenes/newton") / STAGE_NAME
USD_PATH = str((PROJECT_ROOT / USD_REL_PATH).resolve())
CUBE_PRIM_PATH = "/World/Cube"

SIM_DT = 1 / 420.0
RENDER_FPS = 60.0
RENDER_INTERVAL = int(round((1.0 / RENDER_FPS) / SIM_DT))

TEST_DURATION_S = 10.0
NUM_SIM_STEPS = int(round(TEST_DURATION_S / SIM_DT))
MAX_PHYS_ACCEL = 20.0 

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

    # Create and set Prim before initializing the Physics
    rigid_prim = RigidPrim(
            paths=str(CUBE_PRIM_PATH)
        )
    
    rigid_prim.set_masses(wp.array([500.0], dtype=wp.float32, device=DEVICE))
    raw_values = np.random.normal(0, 1, 4)
    normalized = raw_values / np.linalg.norm(raw_values)
    random_quat = np.array([normalized])
    rigid_prim.set_local_poses(orientations=random_quat)

    world.initialize_physics()

    ns = newton_ext.acquire_stage()
    if ns is not None:
        ns.cfg.solver_cfg.nconmax = 1000
        ns.cfg.solver_cfg.njmax = 1200
        if hasattr(ns.cfg.solver_cfg, "num_iterations"):
            ns.cfg.solver_cfg.num_iterations = 32
        if hasattr(ns.cfg.solver_cfg, "num_ls_iterations"):
            ns.cfg.solver_cfg.num_ls_iterations = 6

    world.reset()
    if not world.is_playing():
        world.play()

    # Instantiate Warp Hydrodynamics Wrapper
    hydro_calculator = WarpHydrodynamicsWrapper(
        width=1.00, 
        depth=1.00, 
        height=1.00,
        linear_drag_coefficient=1.2,
        angular_drag_coefficient=0.8,
        linear_damping=300.00,
        angular_damping=150.00,
        water_density=1025.00, 
        gravity=9.81,
        linear_mass_coeff = 0.2,
        angular_mass_coeff = 0.1,
        lift_coefficient=0.5,
        device=DEVICE
    )

    # Fetch mass to avoid explosions
    masses = rigid_prim.get_masses()
    mass = wp.to_torch(masses)[0]
    last_linear_velocity = None
    last_angular_velocity = None

    print(f"HydrodynamicsBehavior (Warp GPU) initialized for {CUBE_PRIM_PATH} with mass {mass} kg")

    for step in range(NUM_SIM_STEPS):
        positions, orientations = rigid_prim.get_world_poses()
        linear_velocity, angular_velocity = rigid_prim.get_velocities()

        if linear_velocity is None or len(linear_velocity) == 0:
            break

        # Ensure contiguous PyTorch tensors on CUDA for Warp kernel compatibility
        positions = wp.to_torch(positions).to(dtype=torch.float32).clone()
        orientations = wp.to_torch(orientations).to(dtype=torch.float32).clone()
        linear_velocity = wp.to_torch(linear_velocity).to(dtype=torch.float32).clone()
        angular_velocity = wp.to_torch(angular_velocity).to(dtype=torch.float32).clone() 

        # Convert quaternion from Isaac Sim (w, x, y, z) to Warp (x, y, z, w)
        orientation = orientations[:, [1, 2, 3, 0]].contiguous()
        
        # Initialize previous velocities on the first step
        if last_linear_velocity is None:
            last_linear_velocity = linear_velocity.clone()
            last_angular_velocity = angular_velocity.clone()
            linear_acceleration = torch.zeros_like(linear_velocity)
            angular_acceleration = torch.zeros_like(angular_velocity)
        else:
            # Estimate and clamp acceleration to prevent added-mass explosions
            linear_acceleration = torch.clamp((linear_velocity - last_linear_velocity) / SIM_DT, -MAX_PHYS_ACCEL, MAX_PHYS_ACCEL)
            angular_acceleration = torch.clamp((angular_velocity - last_angular_velocity) / SIM_DT, -MAX_PHYS_ACCEL, MAX_PHYS_ACCEL)

        # Run GPU Kernel
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

        # Calculate torques generated by off-center forces
        t_torque_buoyancy = torch.cross(t_center_of_buoyancy - positions, t_buoyancy_force, dim=-1)
        t_torque_drag = torch.cross(t_center_of_pressure - positions, t_drag_force, dim=-1)
        t_torque_lift = torch.cross(t_center_of_pressure - positions, t_lift_force, dim=-1)

        # Calculate final net force and torque
        net_force = t_buoyancy_force + t_drag_force + t_lift_force + t_added_mass_force
        net_torque = (
            t_torque_buoyancy
            + t_torque_drag
            + t_torque_lift
            + t_drag_torque
            + t_added_mass_torque
        )
        
        # Safety Clamp (Prevent Explosions)
        max_force = mass * MAX_PHYS_ACCEL
        force_mag = torch.norm(net_force, dim=-1, keepdim=True)
        scale = torch.clamp(max_force / (force_mag + 1e-6), max=1.0)
        net_force = net_force * scale
        net_torque = net_torque * scale
        
        # Apply to Simulation
        rigid_prim.apply_forces_and_torques_at_pos(
            forces=wp.from_torch(net_force.contiguous(), dtype=wp.float32),
            torques=wp.from_torch(net_torque.contiguous(), dtype=wp.float32),
        )

        # Update stored velocities for the next frame
        last_linear_velocity = linear_velocity.clone()
        last_angular_velocity = angular_velocity.clone()

        if step % 30 == 0:
            print(
                f"[Step {step:04d}] Z_pos: {positions[0, 2].item():+.3f}m | "
                f"Z_vel: {linear_velocity[0, 2].item():+.3f}m/s | "
                f"Z_acc: {linear_acceleration[0, 2].item():+.1f}m/s² | "
                f"F_buoy: {t_buoyancy_force[0, 2].item():+.1f}N | "
                f"F_drag: {t_drag_force[0, 2].item():+.1f}N | "
                f"F_net: {net_force[0, 2].item():+.1f}N",
                flush=True,
            )
        
        should_render = RENDER_SIMULATION and ((step + 1) % RENDER_INTERVAL == 0)
        world.step(render=should_render)

    world.stop()
    simulation_app.close()

if __name__ == "__main__":
    main()