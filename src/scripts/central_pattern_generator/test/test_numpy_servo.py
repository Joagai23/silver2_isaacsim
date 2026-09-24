import os
import sys
from pathlib import Path
import numpy as np
from pxr import Usd, UsdPhysics
import omni.usd
from isaacsim.core.api.world import World
from isaacsim.core.prims import Articulation
from isaacsim.core.utils.stage import get_current_stage

# 0. Path Resolution
USD_PATH = "/home/jorge/Documents/Code/silver2_isaacsim/src/scenes/silver2_isaac_sim_locomotion.usd"
if not os.path.exists(USD_PATH):
    raise FileNotFoundError(f"[ERROR] Could not locate USD at: {USD_PATH}")

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
        physics_dt=1.0 / 120.0,
        rendering_dt=1.0 / 60.0,
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
robot_prim_path = "/World/SILVER2"
robot_prim = stage.GetPrimAtPath(robot_prim_path)

if not robot_prim.IsValid():
    for prim in stage.Traverse():
        if prim.GetName() == "SILVER2":
            robot_prim_path = str(prim.GetPath())
            robot_prim = prim
            break

if not robot_prim.IsValid():
    raise RuntimeError(f"[FATAL] Could not find SILVER2 prim on stage!")

if not robot_prim.HasAPI(UsdPhysics.ArticulationRootAPI):
    UsdPhysics.ArticulationRootAPI.Apply(robot_prim)

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

# ==============================================================================
# 3. HARDWARE ABSTRACTION LAYER: CANONICAL JOINT MAPPER
# ==============================================================================
# Canonical order: 6 legs (0 to 5), each having [coxa, femur, tibia]
canonical_joint_names = []
for leg_idx in range(6):
    canonical_joint_names.append(f"coxa_joint_{leg_idx}")
    canonical_joint_names.append(f"femur_joint_{leg_idx}")
    canonical_joint_names.append(f"tibia_joint_{leg_idx}")

dof_names = list(robot.dof_names)

# 1. Maps incoming PhysX telemetry -> Canonical [6, 3] layout:
#    q_canonical = q_physx[physx_to_canonical].reshape(6, 3)
physx_to_canonical = np.array([dof_names.index(name) for name in canonical_joint_names], dtype=np.int64)

# 2. Maps outgoing Canonical [6, 3] commands -> PhysX drive target layout:
#    q_physx = q_canonical.flatten()[canonical_to_physx]
canonical_to_physx = np.array([canonical_joint_names.index(name) for name in dof_names], dtype=np.int64)

print("\n" + "=" * 60)
print("HARDWARE ABSTRACTION LAYER INITIALIZED")
print("=" * 60)
print(f"Canonical DOFs: {len(canonical_joint_names)} | Registered PhysX DOFs: {len(dof_names)}")
print("Joint mapping successfully verified.")

# ==============================================================================
# 4. RUN PHYSICAL STABILITY & SERVO HOLD TEST
# ==============================================================================
print("\n" + "=" * 60)
print("RUNNING 120-STEP PHYSICAL STABILITY TEST")
print("=" * 60)

initial_physx_positions = np.asarray(robot.get_joint_positions()).flatten()
target_positions = np.copy(initial_physx_positions)
robot.set_joint_position_targets(target_positions)

sim_failed = False
for step in range(120):
    world.step(render=True)
    current_q = np.asarray(robot.get_joint_positions()).flatten()
    
    if np.isnan(current_q).any() or np.isinf(current_q).any():
        print(f"[FATAL] Simulation exploded at step {step}! Joint positions contain NaN/Inf.")
        sim_failed = True
        break

if not sim_failed:
    settled_q = np.asarray(robot.get_joint_positions()).flatten()
    settled_torques = np.asarray(robot.get_applied_joint_efforts()).flatten()
    
    max_error = float(np.max(np.abs(settled_q - target_positions)))
    max_torque = float(np.max(np.abs(settled_torques)))
    
    print(f"[OK] Simulation completed 120 steps successfully.")
    print(f"  Max Joint Position Tracking Error: {max_error:.6f} rad ({max_error * 180.0 / np.pi:.3f} deg)")
    print(f"  Max Applied Joint Torque: {max_torque:.3f} N*m")
    
    # Read back canonical pose using our HAL mapper
    canonical_settled_q = settled_q[physx_to_canonical].reshape(6, 3)
    print("\nSettled Joint Angles per Leg (Canonical [coxa, femur, tibia] in deg):")
    for leg_idx in range(6):
        angles_deg = np.rad2deg(canonical_settled_q[leg_idx])
        print(f"  Leg {leg_idx}: coxa={angles_deg[0]:+6.1f}°, femur={angles_deg[1]:+6.1f}°, tibia={angles_deg[2]:+6.1f}°")
print("=" * 60 + "\n")