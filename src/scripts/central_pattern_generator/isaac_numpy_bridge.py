RENDER_SIMULATION = True

import isaacsim
from isaacsim.simulation_app import SimulationApp
simulation_app = SimulationApp({"headless": not RENDER_SIMULATION})

# perform any Isaac Sim / Omniverse imports after instantiating the class
import numpy as np
import omni
import os
from isaacsim.core.api.world import World
from isaacsim.core.prims import Articulation
from pxr import Gf, UsdGeom, PhysxSchema, Usd, UsdPhysics
from numpy_cpg_controller import HexapodCPGController
from isaacsim.core.utils.stage import get_current_stage
from silver2_isaac_constants import SILVER2_LINKS, SILVER2_MOUNTS, SILVER2_DEFAULT_FEET_BODY, SILVER2_STANDING_ANGLES_DEG, SILVER2_DIRECTION_MAP

# 0. Path Resolution
USD_PATH = "/home/jorge/Documents/Code/silver2_isaacsim/src/scenes/silver2_isaac_sim_locomotion.usd"
if not os.path.exists(USD_PATH):
    raise FileNotFoundError(f"[ERROR] Could not locate USD at: {USD_PATH}")

# ---------------------------------------------------------
# Robot & Simulation Configuration
# ---------------------------------------------------------
ROBOT_PRIM_PATH = "/World/SILVER2"
SIM_DT = 0.01  # 100 Hz physics step, matching controller dt

def main():

    # ==============================================================================
    # 1. Clean Stage & Physics Initialization
    # ==============================================================================
    usd_context = omni.usd.get_context()
    current_stage_url = usd_context.get_stage_url() or ""

    # Only open the stage if it is not already loaded in the viewport
    if "silver2_isaac_sim_locomotion.usd" not in current_stage_url:
        print(f"[INFO] Opening stage directly: {USD_PATH}")
        usd_context.open_stage(USD_PATH)
    else:
        print("[INFO] Scene is already open in viewport. Skipping reload.")

    # Retain singleton; do not call clear_instance()
    world = World.instance()
    if world is None:
        world = World(
            physics_dt=SIM_DT,
            rendering_dt=1.0 / 30.0,
            stage_units_in_meters=1.0,
            physics_prim_path="/physicsScene"
        )

    # Force the SimulationContext to bind to /physicsScene
    world.initialize_physics()

    stage = get_current_stage()
    physics_prim = stage.GetPrimAtPath("/physicsScene")
    if not physics_prim.IsValid():
        raise RuntimeError("[ERROR] /physicsScene not found! Ensure the stage is fully loaded in the viewport.")

    # 2. Locate & Wrap Articulation Root
    robot_prim_path = ROBOT_PRIM_PATH
    robot_prim = stage.GetPrimAtPath(robot_prim_path)

    if not robot_prim.IsValid():
        for prim in stage.Traverse():
            if prim.GetName() == "SILVER2":
                robot_prim_path = str(prim.GetPath())
                robot_prim = prim
                break

    if not robot_prim.IsValid():
        raise RuntimeError(f"[FATAL] Could not find SILVER2 prim on stage!")

    # ==============================================================================
    # 2. Bind Articulation
    # ==============================================================================
    robot_name = "SILVER2_robot"
    if world.scene.object_exists(robot_name):
        robot = world.scene.get_object(robot_name)
    else:
        robot = Articulation(
            prim_paths_expr=robot_prim_path,
            name=robot_name,
            reset_xform_properties=False
        )
        world.scene.add(robot)

    # Start timeline and reset to bind physics handles
    if not world.is_playing():
        world.play()
    world.reset()

    # Canonical order: 6 legs (0 to 5), each having [coxa, femur, tibia]
    canonical_joint_names = []
    for leg_idx in range(6):
        canonical_joint_names.append(f"coxa_joint_{leg_idx}")
        canonical_joint_names.append(f"femur_joint_{leg_idx}")
        canonical_joint_names.append(f"tibia_joint_{leg_idx}")

    dof_names = list(robot.dof_names)

    # 1. Maps incoming PhysX telemetry -> Canonical [6, 3] layout:
    physx_to_canonical = np.array([dof_names.index(name) for name in canonical_joint_names], dtype=np.int64)

    # 2. Maps outgoing Canonical [6, 3] commands -> PhysX drive target layout:
    canonical_to_physx = np.array([canonical_joint_names.index(name) for name in dof_names], dtype=np.int64)

    print("\n" + "=" * 60)
    print("HARDWARE ABSTRACTION LAYER INITIALIZED")
    print("=" * 60)
    print(f"Canonical DOFs: {len(canonical_joint_names)} | Registered PhysX DOFs: {len(dof_names)}")
    print("Joint mapping successfully verified.")

    # ==============================================================================
    # 4. CPG LOCOMOTION EXECUTION (500 Steps / 5.0 Seconds)
    # ==============================================================================
    print("\n" + "=" * 60)
    print("INITIALIZING CPG CONTROLLER & STANCE ANCHORS")
    print("=" * 60)

    # Link lengths, mount, and nominal stance coordinate parameters
    silver2_links = SILVER2_LINKS
    silver2_mounts = SILVER2_MOUNTS
    default_feet_body = SILVER2_DEFAULT_FEET_BODY
    silver2_standing_deg = SILVER2_STANDING_ANGLES_DEG

    # Instantiate CPG Controller
    cpg_controller = HexapodCPGController(leg_mounts=silver2_mounts, link_lengths=silver2_links, dt=SIM_DT)

    # 3. Settle on ground in standing posture (60 steps = 0.6s)
    print("[INFO] Holding standing posture for settling...")
    standing_targets_canonical = np.radians(silver2_standing_deg)
    standing_targets_physx = standing_targets_canonical.flatten()[canonical_to_physx]

    for _ in range(60):
        robot.set_joint_position_targets(standing_targets_physx)
        world.step(render=RENDER_SIMULATION)

    # Record initial root position to track progress
    initial_pos, initial_rot = robot.get_world_poses()
    initial_pos = initial_pos[0]
    initial_rot = initial_rot[0]

    print(f"[INFO] Initial Chassis Position: X={initial_pos[0]:.3f}, Y={initial_pos[1]:.3f}, Z={initial_pos[2]:.3f}")

    print("\n" + "=" * 60)
    print("EXECUTING CPG LOCOMOTION LOOP")
    print("=" * 60)

    # 4. Stream dynamic CPG joint targets
    num_locomotion_steps = 500  # 5 seconds of walking (2.5 full strides)
    ramp_steps = 100  # 1.0 second smooth ramp
    walking_direction = SILVER2_DIRECTION_MAP["left"]
    for step in range(num_locomotion_steps):
        # Linear ramp scale: 0.0 -> 1.0
        ramp = min(1.0, (step + 1) / ramp_steps)
        # Compute joint targets in canonical [6, 3] layout (rad)
        canonical_targets = cpg_controller.compute_joint_targets(default_feet_body,dir_angle_rad=walking_direction, ramp=ramp)

        # Route through HAL: Canonical -> PhysX DOF order
        physx_targets = canonical_targets.flatten()[canonical_to_physx]

        # Send targets to PhysX articulation drives
        robot.set_joint_position_targets(physx_targets)
        world.step(render=RENDER_SIMULATION)

        if (step + 1) % 100 == 0:
            current_pos, _ = robot.get_world_poses()
            dy = current_pos[0][1] - initial_pos[1]
            print(f"Step {step + 1:3d}/{num_locomotion_steps} | Y-Advance: {dy:+.4f} m | Height Z: {current_pos[0][2]:.4f} m")

    # Final Telemetry Check
    final_pos, _ = robot.get_world_poses()
    displacement = final_pos[0] - initial_pos

    print("\n" + "=" * 60)
    print("LOCOMOTION RESULTS")
    print("=" * 60)
    print(f"Total Displacement (m):")
    print(f"  ΔX (Lateral Drift)  : {displacement[0]:+.4f} m")
    print(f"  ΔY (Forward Advance): {displacement[1]:+.4f} m")
    print(f"  ΔZ (Chassis Drop)   : {displacement[2]:+.4f} m")
    print("=" * 60 + "\n")

    world.stop()
    simulation_app.close()

if __name__ == "__main__":
    main()