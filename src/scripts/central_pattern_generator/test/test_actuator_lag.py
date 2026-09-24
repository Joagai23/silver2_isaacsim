"""
Independent Diagnostic Test: Actuator Mechanical Phase Lag & Dynamic Tracking (Test 3)
Instruments SILVER2 in Newton GPU physics to quantify dynamic tracking lag.
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

# ------------------------------------------------------------------------------
# Simulation Parameters
# ------------------------------------------------------------------------------
USD_PATH = "/home/jorge/Documents/Code/silver2_isaacsim/src/scenes/newton/silver2_isaac_sim_locomotion.usd"
STAGE_NAME = "silver2_isaac_sim_locomotion.usd"
ROBOT_PRIM_PATH = "/World/SILVER2"

SIM_DT = 1.0 / 420.0
RENDER_FPS = 60.0
RENDER_INTERVAL = int(round((1.0 / RENDER_FPS) / SIM_DT))
DECIMATION = 4
CPG_DT = SIM_DT * DECIMATION  # ~0.009524 s (105 Hz)


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


def main():
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

    # Hardware Abstraction Mapping Setup
    canonical_joint_names = []
    for leg_idx in range(6):
        canonical_joint_names.append(f"coxa_joint_{leg_idx}")
        canonical_joint_names.append(f"femur_joint_{leg_idx}")
        canonical_joint_names.append(f"tibia_joint_{leg_idx}")

    dof_names = list(robot_view.dof_names)
    num_dofs = len(dof_names)

    canonical_to_newton = torch.tensor(
        [canonical_joint_names.index(name) for name in dof_names],
        dtype=torch.long,
        device=DEVICE
    )

    # Invert mapping so we can extract measured Newton states back into canonical order
    # newton_to_canonical maps: canonical_index = newton_to_canonical[dof_index]
    newton_to_canonical = torch.tensor(
        [dof_names.index(name) for name in canonical_joint_names],
        dtype=torch.long,
        device=DEVICE
    )

    # Instantiate Controller
    cpg_controller = NumpyHexapodCPGController(
        leg_mounts=SILVER2_MOUNTS,
        link_lengths=SILVER2_LINKS,
        dt=CPG_DT,
        total_period=2.0,
        gait="tripod"
    )

    # 1. Settle in Standing Posture for 100 Steps
    standing_targets_deg = np.array(SILVER2_STANDING_ANGLES_DEG, dtype=np.float32).flatten()
    standing_targets_rad = np.deg2rad(standing_targets_deg).astype(np.float32)
    standing_targets_gpu = torch.as_tensor(standing_targets_rad, dtype=torch.float32, device=DEVICE)
    torch_standing_targets = standing_targets_gpu[canonical_to_newton].unsqueeze(0)

    robot_view.set_joint_positions(torch_standing_targets)
    robot_view.set_joint_velocities(torch.zeros_like(torch_standing_targets))

    print("\n[INFO] Settling in standing posture (100 steps)...")
    for step in range(100):
        robot_view.set_joint_position_targets(torch_standing_targets)
        world.step(render=RENDER_SIMULATION and ((step + 1) % RENDER_INTERVAL == 0))

    # 2. Test Execution: Log 3 Complete Strides (6.0 seconds = 630 CPG ticks = 2520 physics steps)
    TOTAL_PERIOD = 2.0
    NUM_CYCLES = 3
    TOTAL_TIME = NUM_CYCLES * TOTAL_PERIOD
    NUM_PHYSICS_STEPS = int(round(TOTAL_TIME / SIM_DT))
    NUM_CPG_TICKS = NUM_PHYSICS_STEPS // DECIMATION

    print("=" * 80)
    print(f"RUNNING TEST 3: LOGGING {NUM_CYCLES} CYCLES ({TOTAL_TIME:.1f} s) OF ACTIVE LOCOMOTION")
    print(f"Sampling Rate: {1.0 / CPG_DT:.1f} Hz ({NUM_CPG_TICKS} control samples)")
    print("=" * 80)

    # Preallocate telemetry buffers (shape: [NUM_CPG_TICKS, 18])
    time_log = np.zeros(NUM_CPG_TICKS)
    target_q_log = np.zeros((NUM_CPG_TICKS, 18))
    actual_q_log = np.zeros((NUM_CPG_TICKS, 18))

    current_targets = torch_standing_targets
    cpg_tick = 0

    for step in range(NUM_PHYSICS_STEPS):
        if step % DECIMATION == 0 and cpg_tick < NUM_CPG_TICKS:
            time_log[cpg_tick] = cpg_tick * CPG_DT

            # Compute CPG targets
            canonical_targets = cpg_controller.compute_joint_targets(
                SILVER2_DEFAULT_FEET_BODY,
                dir_angle_rad=0.0,
                stride_forward=0.005,
                stride_lateral=0.005,
                yaw_rate=0.0,
                yaw_gain=0.01,
                ramp=1.0
            )
            canonical_flat = canonical_targets.flatten()
            target_q_log[cpg_tick] = canonical_flat

            # Convert to Newton ordering
            cpg_targets_gpu = torch.as_tensor(canonical_flat, dtype=torch.float32, device=DEVICE)
            current_targets = cpg_targets_gpu[canonical_to_newton].unsqueeze(0)

            # Sample actual joint positions (mapped back to canonical ordering)
            measured_newton_q = robot_view.get_joint_positions().squeeze()
            measured_canonical_q = measured_newton_q[newton_to_canonical].cpu().numpy()
            actual_q_log[cpg_tick] = measured_canonical_q

            cpg_tick += 1

        robot_view.set_joint_position_targets(current_targets)
        should_render = RENDER_SIMULATION and ((step + 1) % RENDER_INTERVAL == 0)
        world.step(render=should_render)

    # 3. Post-Processing & Telemetry Analysis
    print("\n" + "=" * 90)
    print(f"{'Probe Joint':<16} | {'Phase Lag (ms)':<15} | {'Amp Ratio (%)':<15} | {'Max Error (deg)':<16} | {'RMS Error (deg)'}")
    print("=" * 90)

    # Probe representative joints: Leg 0 (Front-Left), Leg 1 (Middle-Left), Leg 4 (Middle-Right)
    probe_indices = [
        (0, "Leg 0 Coxa"),
        (1, "Leg 0 Femur"),
        (2, "Leg 0 Tibia"),
        (3, "Leg 1 Coxa"),
        (4, "Leg 1 Femur"),
        (5, "Leg 1 Tibia"),
        (12, "Leg 4 Coxa"),
        (13, "Leg 4 Femur"),
        (14, "Leg 4 Tibia"),
    ]

    for j_idx, j_name in probe_indices:
        cmd_series = target_q_log[:, j_idx]
        act_series = actual_q_log[:, j_idx]

        # Ignore initial settling tick if any
        cmd = cmd_series - np.mean(cmd_series)
        act = act_series - np.mean(act_series)

        # Cross-Correlation to find mechanical phase delay
        corr = np.correlate(act, cmd, mode='full')
        lags = np.arange(-len(cmd) + 1, len(cmd))
        best_lag_idx = lags[np.argmax(corr)]
        time_lag_ms = best_lag_idx * CPG_DT * 1000.0

        # Amplitude range (attenuation)
        cmd_range = np.ptp(cmd_series)
        act_range = np.ptp(act_series)
        amp_ratio_pct = (act_range / cmd_range * 100.0) if cmd_range > 1e-4 else 100.0

        # Tracking error statistics in degrees
        err_deg = np.rad2deg(act_series - cmd_series)
        max_err = np.max(np.abs(err_deg))
        rms_err = np.sqrt(np.mean(err_deg**2))

        print(f"{j_name:<16} | {time_lag_ms:10.1f} ms    | {amp_ratio_pct:10.1f} %    | {max_err:11.2f}°       | {rms_err:10.2f}°")

    print("=" * 90 + "\n")

    world.stop()
    simulation_app.close()


if __name__ == "__main__":
    main()