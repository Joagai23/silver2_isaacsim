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

from silver2_constants.isaac_constants import *
from warp_cpg_controller import WarpHexapodCPGController

# Path Resolution & Simulation Parameters
USD_PATH = "/home/jorge/Documents/Code/silver2_isaacsim/src/scenes/newton/silver2_isaac_sim_locomotion.usd"
STAGE_NAME = "silver2_isaac_sim_locomotion.usd"
ROBOT_PRIM_PATH = "/World/SILVER2"

SIM_DT = 1 / 420.0
RENDER_FPS = 60.0
RENDER_INTERVAL = int(round((1.0 / RENDER_FPS) / SIM_DT))
DECIMATION_RATIO = 4

def audit_and_repair_joints(stage, robot_root_path=ROBOT_PRIM_PATH):
    """Explicit UsdPhysics.DriveAPI properties for all joints in the Hexapod Robot."""
    robot_prim = stage.GetPrimAtPath(robot_root_path)
    if not robot_prim.IsValid():
        raise RuntimeError(f"Robot root prim not found at {robot_root_path}")
    
    for prim in Usd.PrimRange(robot_prim):
        # Drive configuration
        if prim.IsA(UsdPhysics.Joint):
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

        # Enable contact reporting on all Tibia links
        if prim.HasAPI(UsdPhysics.CollisionAPI) or prim.HasAPI(UsdPhysics.RigidBodyAPI):
            if "tibia" in prim.GetName().lower():
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
    # Stage Init.
    usd_context = omni.usd.get_context()
    current_stage_url = usd_context.get_stage_url() or ""

    if STAGE_NAME not in current_stage_url:
        print(f"[INFO] Loading stage into Newton environment: {USD_PATH}")
        usd_context.open_stage(USD_PATH)

    stage = get_current_stage()
    audit_and_repair_joints(stage, robot_root_path=ROBOT_PRIM_PATH)
    setup_newton_scene(stage, "/physicsScene")
    SimulationManager.switch_physics_engine("newton")

    ns = newton_ext.acquire_stage()
    if ns is not None:
        ns.cfg.solver_cfg.nconmax = 1000

    # World Init.
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

    if not world.is_playing():
        world.play()
    world.reset()

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
    cpg_dt = SIM_DT * DECIMATION_RATIO
    cpg_controller = WarpHexapodCPGController(
        leg_mounts=SILVER2_MOUNTS,
        link_lengths=SILVER2_LINKS,
        dt=cpg_dt,
        total_period=2.0,
        gait="tripod",
        device=DEVICE
    )

    # Dynamically-computed standing pose
    standing_targets_deg = np.array(SILVER2_STANDING_ANGLES_DEG, dtype=np.float32).flatten()
    standing_targets_rad = np.deg2rad(standing_targets_deg).astype(np.float32)
    standing_targets_gpu = torch.as_tensor(standing_targets_rad, dtype=torch.float32, device=DEVICE)
    torch_standing_targets = standing_targets_gpu[canonical_to_newton].unsqueeze(0)
    newton_targets = torch_standing_targets

    robot_view.set_joint_positions(torch_standing_targets)
    robot_view.set_joint_velocities(torch.zeros_like(torch_standing_targets))

    print("[INFO] Running 1000 stance settling steps...")
    for step in range(1000):
        robot_view.set_joint_position_targets(torch_standing_targets)
        world.step(render=RENDER_SIMULATION and ((step + 1) % RENDER_INTERVAL == 0))

    # Extract true loaded joint angles
    settled_q_rad = robot_view.get_joint_positions().squeeze()[newton_to_canonical].cpu().numpy()
    settled_q_deg = np.rad2deg(settled_q_rad).reshape((6, 3))

    # Compute empirical foot baseline directly from settled physics
    calibrated_feet_body = cpg_controller.compute_default_feet_body(settled_q_deg)

    # Locomotion Parameters
    num_locomotion_steps = 2000
    ramp_steps = 200 
    walking_direction = SILVER2_DIRECTION_MAP["right"]

    print(f"[INFO] Running {num_locomotion_steps} active locomotion steps with sensory phase resetting...")
    for step in range(num_locomotion_steps):
        if step % DECIMATION_RATIO == 0:
            ramp = min(1.0, (step + 1) / ramp_steps)

            # Query raw contact forces from Newton GPU memory
            raw_contact_forces = feet_view.get_net_contact_forces(clone=False)
            if raw_contact_forces.dim() == 3:
                raw_contact_forces = raw_contact_forces.squeeze(0)

            # Convert Newton contact impulses (N*s) to continuous force (N)
            continuous_contact_forces = raw_contact_forces / SIM_DT

            # Re-order into canonical leg order and enforce contiguity
            canonical_contact_forces = continuous_contact_forces[feet_to_canonical].contiguous()
            contact_forces_wp = wp.from_torch(canonical_contact_forces, dtype=wp.vec3)

            # Compute joint targets with closed-loop phase resetting
            warp_targets = cpg_controller.compute_joint_targets(
                default_feet_pos_body=calibrated_feet_body,
                dir_angle_rad=walking_direction,
                stride_forward=0.005,
                stride_lateral=0.005,
                yaw_rate=0.0,
                yaw_gain=0.0,
                ramp=ramp,
                contact_forces=contact_forces_wp
            )

            # Zero-copy view into PyTorch
            torch_targets = wp.to_torch(warp_targets).reshape(-1)
            newton_targets = torch_targets[canonical_to_newton].unsqueeze(0)

        # Hold targets across 420 Hz physics steps; render viewport at 60 Hz
        robot_view.set_joint_position_targets(newton_targets)
        should_render = RENDER_SIMULATION and ((step + 1) % RENDER_INTERVAL == 0)
        world.step(render=should_render)

    poses, _ = robot_view.get_world_poses()
    final_pos = poses[0].cpu().numpy()
    print(f"[INFO] Final Position: X={final_pos[0]:.3f}, Y={final_pos[1]:.3f}, Z={final_pos[2]:.3f} m")

    world.stop()
    simulation_app.close()

if __name__ == "__main__":
    main()