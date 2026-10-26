RENDER_SIMULATION = False

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
from pxr import Usd, UsdPhysics, Gf, Sdf
import re
import csv
import json
import os
from datetime import datetime
from pathlib import Path
import sys

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

# Resolve path to 'src/scripts'
SCRIPT_DIR = Path(__file__).resolve()
PROJECT_ROOT = next((p for p in SCRIPT_DIR.parents if (p / "src").is_dir()), SCRIPT_DIR.parents[3])
SCRIPTS_DIR = PROJECT_ROOT / "src" / "scripts"

for p in (PROJECT_ROOT, SCRIPTS_DIR, SCRIPT_DIR.parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src.scripts.silver2_constants.isaac_constants import *
from src.scripts.central_pattern_generator.evolutionary_algorithm.evo_warp_cpg_controller import VectorizedWarpHexapodCPGController
from src.scripts.hydrodynamics.warp_hydrodynamics_wrapper import WarpHydrodynamicsWrapper

# --------------------------------------------------------------------------
# Simulation & Drive Parameters (Hydrodynamic Newton Calibrated)
# --------------------------------------------------------------------------
STAGE_NAME = "silver2_training_grid_hydro.usd"
USD_REL_PATH = Path("src/scenes/newton") / STAGE_NAME
USD_PATH = str((PROJECT_ROOT / USD_REL_PATH).resolve())
JSON_CONFIG_PATH = PROJECT_ROOT / "src/scripts/hydrodynamics/silver_hydro_config.json"

SIM_HZ = 420.0
SIM_DT = 1 / SIM_HZ
DECIMATION_RATIO = 4
CTRL_DT = SIM_DT * DECIMATION_RATIO

STEPS_PER_EVAL = 2520
NUM_CPG_STEPS = STEPS_PER_EVAL // DECIMATION_RATIO
MAX_PHYS_ACCEL = 20.0
RAMP_STEPS = 210
ROBOT_WEIGHT = SILVER2_TOTAL_MASS * 9.81

STRIDE_FWD_CMD = 0.005
STRIDE_LAT_CMD = 0.010
DIRECTION = SILVER2_DIRECTION_MAP["right"]

# --------------------------------------------------------------------------
# Checkpoint & Telemetry Paths
# --------------------------------------------------------------------------
RUN_ID = datetime.now().strftime("cpg_ga_%Y%m%d_%H%M%S")
RUN_DIR = os.path.join(str(PROJECT_ROOT), "src/scripts/central_pattern_generator/runs", RUN_ID)
os.makedirs(RUN_DIR, exist_ok=True)

CSV_LOG_PATH = os.path.join(RUN_DIR, "history.csv")
CHECKPOINT_PATH = os.path.join(RUN_DIR, "checkpoint_latest.pt")
FINAL_JSON_PATH = os.path.join(RUN_DIR, "best_cpg_parameters.json")
FINAL_PT_PATH = os.path.join(RUN_DIR, "best_cpg_genome.pt")

GENOME_PARAM_NAMES = [
    "l2_clearance",
    "k1_stride_gain",
    "k2_stance_gain",
    "k3_vertical_gain",
    "epsilon_duty_factor",
    "total_period_T",
    "coupling_strength_w",
    "sigmoid_slope_b",
    "f_touch_td",
    "f_release_rel"
]

def natural_sort_key(s):
    """Sorts strings containing numbers naturally: SILVER2_0, SILVER2_01, ..., SILVER2_10."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def save_checkpoint(gen, population, fitness, best_genome_overall, best_fitness_overall):
    """Saves rolling binary checkpoint after each generation."""
    torch.save({
        "generation": gen,
        "population": population.cpu(),
        "fitness": fitness.cpu(),
        "best_genome_overall": best_genome_overall.cpu(),
        "best_fitness_overall": float(best_fitness_overall)
    }, CHECKPOINT_PATH)

def export_best_parameters(best_genome, best_fitness, total_generations, population_size):
    """Exports structured JSON and PyTorch weights upon optimization completion."""
    g = best_genome.tolist()
    config_dict = {
        "metadata": {
            "run_id": RUN_ID,
            "export_time": datetime.now().isoformat(),
            "best_fitness_score": float(best_fitness),
            "num_generations": total_generations,
            "population_size": population_size
        },
        "raw_genome_vector": g,
        "cpg_parameters": {
            "clearance_l2_m": round(g[0], 5),
            "clearance_l2_mm": round(g[0] * 1000.0, 2),
            "stride_gain_k1": round(g[1], 4),
            "stance_gain_k2": round(g[2], 4),
            "vertical_gain_k3": round(g[3], 4),
            "duty_factor_epsilon": round(g[4], 4),
            "gait_period_T_s": round(g[5], 3),
            "coupling_strength_w": round(g[6], 4),
            "sigmoid_slope_b": round(g[7], 3),
            "touchdown_force_F_td_N": round(g[8], 2),
            "release_force_F_rel_N": round(g[9], 2)
        },
        "python_snippet_constants": {
            "CPG_OPTIMIZED_GAINS": {
                "l2": round(g[0], 5),
                "k1": round(g[1], 4),
                "k2": round(g[2], 4),
                "k3": round(g[3], 4),
                "epsilon": round(g[4], 4),
                "period": round(g[5], 3),
                "coupling": round(g[6], 4),
                "b": round(g[7], 3),
                "f_touch": round(g[8], 2),
                "f_release": round(g[9], 2)
            }
        }
    }

    with open(FINAL_JSON_PATH, "w") as f:
        json.dump(config_dict, f, indent=4)

    torch.save(best_genome, FINAL_PT_PATH)

    print("\n" + "=" * 85)
    print("[SUCCESS] Optimization complete! Exported champion genome:")
    print(f"  --> JSON Configuration: {FINAL_JSON_PATH}")
    print(f"  --> PyTorch Tensor:     {FINAL_PT_PATH}")
    print("=" * 85)

def audit_and_repair_all_drives(stage, envs_scope_path="/World/env"):
    """Enforces verified stiffness, damping, torque ceilings, and contact reporting across all robots."""
    scope_prim = stage.GetPrimAtPath(envs_scope_path)
    if not scope_prim.IsValid():
        return
    
    for prim in Usd.PrimRange(scope_prim):
        # Configure Joint Drives
        if prim.IsA(UsdPhysics.Joint):
            drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
            drive.CreateTypeAttr().Set("force")
            drive.CreateStiffnessAttr().Set(float(np.deg2rad(K_P)))
            drive.CreateDampingAttr().Set(float(np.deg2rad(K_D)))
            drive.CreateMaxForceAttr().Set(float(MAX_TORQUE))

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

def get_population_link_paths(stage, target_robot_paths="/World/env"):
    """Discovers all rigid-body link prim paths across all parallel robot instances."""
    link_paths = []
    for r_path in target_robot_paths:
        robot_prim = stage.GetPrimAtPath(r_path)
        if not robot_prim.IsValid():
            continue
        for prim in Usd.PrimRange(robot_prim):
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
    audit_and_repair_all_drives(stage, envs_scope_path="/World/env")
    setup_newton_scene(stage, "/physicsScene")
    SimulationManager.switch_physics_engine("newton")
    
    # Discover Robots under /World/env
    envs_scope = stage.GetPrimAtPath("/World/env")
    if not envs_scope.IsValid():
        raise RuntimeError("Could not find '/World/env' scope on stage!")

    target_robot_paths = [
        child.GetPath().pathString
        for child in envs_scope.GetChildren()
        if "silver2" in child.GetName().lower()
    ]
    target_robot_paths = sorted(target_robot_paths, key=natural_sort_key)
    num_envs = len(target_robot_paths)

    print(f"\n[INFO] Initialized experiment run: {RUN_ID}")
    print(f"[INFO] Streaming CSV metrics to: {CSV_LOG_PATH}")
    print(f"[INFO] Discovered {num_envs} robot instances under '{envs_scope.GetPath()}':")
    for r_idx, r_path in enumerate(target_robot_paths[:4]):
        print(f"  [{r_idx}] -> {r_path}")
    if num_envs > 4:
        print(f"  ... and {num_envs - 4} more instances.")

    assert num_envs > 0, "No SILVER2 robots found under /env!"

    # Initialize CSV header row
    csv_headers = [
        "generation",
        "best_fitness",
        "mean_fitness",
        "std_fitness",
        "best_target_dist_m",
        "best_cross_track_drift_m",
        "best_cost_of_transport",
        "best_peak_impact_N",
        "best_chassis_var",
    ] + GENOME_PARAM_NAMES

    with open(CSV_LOG_PATH, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(csv_headers)

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
        ns.cfg.solver_cfg.nconmax = 10560
        ns.cfg.solver_cfg.njmax = 12000
        
        if hasattr(ns.cfg.solver_cfg, "num_iterations"):
            ns.cfg.solver_cfg.num_iterations = 32
        if hasattr(ns.cfg.solver_cfg, "num_ls_iterations"):
            ns.cfg.solver_cfg.num_ls_iterations = 6

    # Discover all 19 SILVER2 links per Robot and create a RigidPrim view
    all_link_paths = get_population_link_paths(stage, target_robot_paths)
    print(f"[INFO] Discovered {len(all_link_paths)} total rigid body links across {num_envs} environments.")
    
    robot_links_view = RigidPrim(
        prim_paths_expr=all_link_paths,
        name="silver2_population_hydro_links",
        reset_xform_properties=False
    )
    world.scene.add(robot_links_view)

    # Instantiate Warp Hydrodynamics Wrapper from JSON
    hydro_calculator = WarpHydrodynamicsWrapper.from_json(
        json_path=JSON_CONFIG_PATH,
        link_names_or_paths=robot_links_view.prim_paths,
        device=DEVICE
    )
    print(f"[INFO] HydrodynamicsBehavior (Warp GPU) initialized for {len(all_link_paths)} SILVER2 links.")

    # Discover the Exact Tibia Link Names from the First Robot
    first_robot_prim = stage.GetPrimAtPath(target_robot_paths[0])
    discovered_tibia_names = []
    for prim in Usd.PrimRange(first_robot_prim):
        name = prim.GetName()
        if "tibia_" in name.lower() and prim.HasAPI(UsdPhysics.RigidBodyAPI):
            discovered_tibia_names.append(name)

    tibia_names = sorted(list(set(discovered_tibia_names)), key=natural_sort_key)
    print(f"[INFO] Discovered canonical tibia links in USD: {tibia_names}")
    assert len(tibia_names) == 6, (
        f"Expected exactly 6 tibia rigid bodies, but found {len(tibia_names)}: {tibia_names}"
    )

    all_tibia_paths = [f"{r_path}/{t_name}" for r_path in target_robot_paths for t_name in tibia_names]
    feet_view = RigidPrim(
        prim_paths_expr=all_tibia_paths,
        name="silver2_population_feet_view",
        track_contact_forces=True,
        prepare_contact_sensors=True
    )
    world.scene.add(feet_view)

    # Vectorized Articulation View
    robot_view = Articulation(
        prim_paths_expr=target_robot_paths,
        name="silver2_population_view",
        reset_xform_properties=False
    )
    world.scene.add(robot_view)

    world.reset()
    if not world.is_playing():
        world.play()

    # Fetch per-link masses for accurate safety clamping
    link_masses = robot_links_view.get_masses().to(dtype=torch.float32).view(-1, 1)

    # Enforce verified drive gains across all DOFs in memory
    joint_kps = torch.full((num_envs, 18), K_P, device=DEVICE)
    joint_kds = torch.full((num_envs, 18), K_D, device=DEVICE)
    max_efforts = torch.full((num_envs, 18), MAX_TORQUE, device=DEVICE)
    robot_view.set_gains(kps=joint_kps, kds=joint_kds)
    robot_view.set_max_efforts(max_efforts)

    # 5. Joint & Foot Hardware Mappings
    canonical_joint_names = []
    for leg_idx in range(6):
        canonical_joint_names.append(f"Joint_Body_Coxa_{leg_idx}")
        canonical_joint_names.append(f"Joint_Coxa_Femur_{leg_idx}")
        canonical_joint_names.append(f"Joint_Femur_Tibia_{leg_idx}")

    dof_names = list(robot_view.dof_names)
    canonical_to_newton = torch.tensor(
        [canonical_joint_names.index(name) for name in dof_names],
        dtype=torch.long,
        device=DEVICE
    )

    view_paths = [p.lower() for p in feet_view.prim_paths]
    canonical_foot_indices = []
    for r_path in target_robot_paths:
        for leg_idx in range(6):
            expected = f"{r_path}/{tibia_names[leg_idx]}".lower()
            matched_idx = -1
            for v_idx, v_path in enumerate(view_paths):
                if v_path == expected:
                    matched_idx = v_idx
                    break
            if matched_idx == -1:
                for v_idx, v_path in enumerate(view_paths):
                    if r_path.lower() in v_path and f"tibia_{leg_idx}" in v_path:
                        matched_idx = v_idx
                        break
            assert matched_idx != -1, f"Could not map foot {expected} in RigidPrim view!"
            canonical_foot_indices.append(matched_idx)

    assert len(canonical_foot_indices) == num_envs * 6, (
        f"Foot indexing mismatch! Expected {num_envs * 6}, matched {len(canonical_foot_indices)}"
    )
    feet_to_canonical = torch.tensor(canonical_foot_indices, dtype=torch.long, device=DEVICE)

    # 6. Instantiate Vectorized Controller
    cpg_controller = VectorizedWarpHexapodCPGController(
        num_envs=num_envs,
        leg_mounts=SILVER2_MOUNTS,
        link_lengths=SILVER2_LINKS,
        dt=CTRL_DT,
        gait="tripod",
        device=DEVICE
    )

    standing_rad = np.deg2rad(np.array(SILVER2_STANDING_ANGLES_DEG, dtype=np.float32).flatten())
    standing_tensor = torch.as_tensor(standing_rad, dtype=torch.float32, device=DEVICE)
    standing_targets_pop = standing_tensor[canonical_to_newton].unsqueeze(0).repeat(num_envs, 1)

    robot_view.set_joint_positions(standing_targets_pop)
    robot_view.set_joint_velocities(torch.zeros_like(standing_targets_pop))

    # Helper function to apply one step of Warp hydrodynamics
    last_linear_velocity = None
    last_angular_velocity = None

    def apply_hydrodynamics_step(last_lin_vel, last_ang_vel):
        positions, orientations = robot_links_view.get_world_poses()
        velocities = robot_links_view.get_velocities()
        linear_velocity = velocities[:, :3].to(dtype=torch.float32).contiguous()
        angular_velocity = velocities[:, 3:].to(dtype=torch.float32).contiguous()
        positions = positions.to(dtype=torch.float32).contiguous()
        orientations = orientations.to(dtype=torch.float32).contiguous()
        orientation = orientations[:, [1, 2, 3, 0]].contiguous()

        if last_lin_vel is None:
            lin_acc = torch.zeros_like(linear_velocity)
            ang_acc = torch.zeros_like(angular_velocity)
        else:
            lin_acc = torch.clamp((linear_velocity - last_lin_vel) / SIM_DT, -MAX_PHYS_ACCEL, MAX_PHYS_ACCEL)
            ang_acc = torch.clamp((angular_velocity - last_ang_vel) / SIM_DT, -MAX_PHYS_ACCEL, MAX_PHYS_ACCEL)

        (
            t_buoyancy_force, t_drag_force, t_lift_force, t_drag_torque,
            t_added_mass_force, t_added_mass_torque, t_center_of_buoyancy, t_center_of_pressure
        ) = hydro_calculator.calculate_hydrodynamic_forces(
            positions, orientation, linear_velocity, angular_velocity, lin_acc, ang_acc
        )

        t_torque_buoyancy = torch.cross(t_center_of_buoyancy - positions, t_buoyancy_force, dim=-1)
        t_torque_drag = torch.cross(t_center_of_pressure - positions, t_drag_force, dim=-1)
        t_torque_lift = torch.cross(t_center_of_pressure - positions, t_lift_force, dim=-1)

        dynamic_force = t_drag_force + t_lift_force + t_added_mass_force
        dynamic_torque = t_torque_drag + t_torque_lift + t_drag_torque + t_added_mass_torque

        max_force = link_masses * MAX_PHYS_ACCEL
        force_mag = torch.norm(dynamic_force, dim=-1, keepdim=True)
        scale = torch.clamp(max_force / (force_mag + 1e-6), max=1.0)

        net_force = t_buoyancy_force + (dynamic_force * scale)
        net_torque = t_torque_buoyancy + (dynamic_torque * scale)

        robot_links_view.apply_forces_and_torques_at_pos(
            forces=net_force.contiguous(),
            torques=net_torque.contiguous(),
            is_global=True
        )
        return linear_velocity.clone(), angular_velocity.clone()

    settling_steps = int(2 * SIM_HZ)
    print(f"[INFO] Settling robot underwater for {settling_steps} steps...")
    for _ in range(settling_steps):
        last_linear_velocity, last_angular_velocity = apply_hydrodynamics_step(
            last_linear_velocity, last_angular_velocity
        )
        robot_view.set_joint_position_targets(standing_targets_pop)
        world.step(render=False)

    # Capture the verified settled base poses and joint states
    settled_root_pos, settled_root_rot = robot_view.get_world_poses(clone=True)
    settled_q = robot_view.get_joint_positions(clone=True)
    print(f"[INFO] Settling complete. Base Z: {settled_root_pos[0, 2]:.4f} m. State buffers established.")

    angles_rad = np.deg2rad(np.array(SILVER2_STANDING_ANGLES_DEG, dtype=np.float32))
    default_feet_body_host = np.zeros((6, 3), dtype=np.float32)
    for i in range(6):
        q1, q2, q3 = angles_rad[i]
        l1, l2, l3 = SILVER2_LINKS
        r_dist = l1 + l2 * np.cos(q2) + l3 * np.cos(q2 + q3)
        z_pos = -l2 * np.sin(q2) - l3 * np.sin(q2 + q3)
        x_pos = r_dist * np.cos(q1)
        y_pos = r_dist * np.sin(q1)
        yaw = float(SILVER2_MOUNTS[list(SILVER2_MOUNTS.keys())[i]]['yaw'])
        m_pos = SILVER2_MOUNTS[list(SILVER2_MOUNTS.keys())[i]]['pos']
        c_y, s_y = np.cos(yaw), np.sin(yaw)
        default_feet_body_host[i] = [
            m_pos[0] + x_pos * c_y - y_pos * s_y,
            m_pos[1] + x_pos * s_y + y_pos * c_y,
            m_pos[2] + z_pos
        ]
    default_feet_body_wp = wp.array(default_feet_body_host, dtype=wp.vec3, device=DEVICE)
    num_feet_total = num_envs * 6

    # --------------------------------------------------------------------------
    # 7. Evolutionary Optimization (10 Parameters)
    # Genome: [l2, k1, k2, k3, epsilon, total_period, coupling_strength, b, f_touch, f_release]
    # --------------------------------------------------------------------------
    GENOME_MIN_WIDE = torch.tensor([0.001, 0.45, 0.00, 0.72, 0.44, 0.90, 0.15, 1.5, 1.80, 0.20], device=DEVICE)
    GENOME_MAX_WIDE = torch.tensor([0.020, 1.25, 0.12, 1.15, 0.72, 2.50, 0.90, 5.0, 6.50, 1.50], device=DEVICE)

    CURRICULUM_MASKS = {
        "stage1_spatial": torch.tensor([True, True, True, True, False, False, False, False, False, False], dtype=torch.bool, device=DEVICE),
        "stage2_temporal": torch.tensor([False, False, False, False, True, True, True, True, False, False], dtype=torch.bool, device=DEVICE),
        "stage3_sensory": torch.tensor([False, False, False, False, False, False, False, False, True, True], dtype=torch.bool, device=DEVICE),
        "stage4_fine_tuning": torch.tensor([True] * 10, dtype=torch.bool, device=DEVICE),
    }

    CURRICULUM = [
        ("stage1_spatial",     120),
        ("stage2_temporal",    120),
        ("stage3_sensory",     60),
        ("stage4_fine_tuning", 120),
    ]

    TOTAL_GENERATIONS = sum(gens for _, gens in CURRICULUM)

    # Seed with the verified Underwater Champion Genome (0.144 m/s, 21.6 N peak, CoT 0.16)
    current_champion_genome = torch.tensor([
        0.00200,  # l2_clearance
        0.55475,   # k1_stride_gain
        0.11106,   # k2_stance_gain
        0.80386,   # k3_vertical_gain
        0.49081,   # epsilon_duty_factor
        1.17118,   # total_period_T 
        0.65987,   # coupling_strength_w
        3.16319,   # sigmoid_slope_b
        5.00882,   # f_touch_td
        0.21571    # f_release_rel
    ], device=DEVICE)

    best_overall_fitness = -float("inf")
    best_overall_genome = current_champion_genome.clone()

    global_gen = 0
    print(f"\n[INFO] Launching Full Curriculum: {TOTAL_GENERATIONS} Total Generations across {num_envs} Parallel Envs")
    print("=" * 115)

    # --------------------------------------------------------------------------
    # 8. Hierarchical Curriculum Execution
    # --------------------------------------------------------------------------
    for stage_name, stage_generations in CURRICULUM:
        print(f"\n{'#' * 85}")
        print(f"  >>> ENTERING CURRICULUM: {stage_name.upper()} ({stage_generations} Generations)")
        print(f"{'#' * 85}\n")

        active_mask = CURRICULUM_MASKS[stage_name]
        frozen_mask = ~active_mask
        stage_best_fitness = -float("inf")

        # Select dynamic bounds based on active curriculum stage
        if "fine_tuning" in stage_name:
            span = (GENOME_MAX_WIDE - GENOME_MIN_WIDE) * 0.15
            bounds_min = torch.max(GENOME_MIN_WIDE, current_champion_genome - span)
            bounds_max = torch.min(GENOME_MAX_WIDE, current_champion_genome + span)
        else:
            bounds_min = GENOME_MIN_WIDE
            bounds_max = GENOME_MAX_WIDE

        # Initialize stage population seeded around the latest champion
        population = current_champion_genome.repeat(num_envs, 1)
        if active_mask.any():
            random_noise = bounds_min + torch.rand((num_envs, 10), device=DEVICE) * (bounds_max - bounds_min)
            population = torch.where(active_mask.unsqueeze(0), random_noise, population)

        population[0] = current_champion_genome.clone()
        cpg_controller.set_population_genomes(population)

        for stage_gen in range(stage_generations):
            global_gen += 1

            # 1. Deterministic Generation Reset
            robot_view.set_world_poses(positions=settled_root_pos, orientations=settled_root_rot)
            robot_view.set_velocities(torch.zeros((num_envs, 6), device=DEVICE))
            robot_view.set_joint_positions(settled_q)
            robot_view.set_joint_velocities(torch.zeros_like(settled_q))
            cpg_controller.reset_all_states()

            # Reset hydrodynamic finite-difference velocity history for the new episode
            last_linear_velocity = None
            last_angular_velocity = None
    
            newton_targets = settled_q.clone()
            robot_view.set_joint_position_targets(newton_targets)

            # 2. State & Telemetry Buffers
            start_pos = settled_root_pos.clone()
            peak_impact_forces = torch.zeros(num_envs, device=DEVICE)
            total_energy = torch.zeros(num_envs, device=DEVICE)
            torque_sq_accum = torch.zeros(num_envs, device=DEVICE)
            sum_fz_accum = torch.zeros(num_envs, device=DEVICE)
            post_ramp_steps = 0

            cmd_theta = torch.as_tensor(DIRECTION, device=DEVICE, dtype=torch.float32)
            cos_theta = torch.cos(cmd_theta)
            sin_theta = torch.sin(cmd_theta)
    
            root_z_buf = torch.empty((NUM_CPG_STEPS, num_envs), device=DEVICE)
            root_tilt_buf = torch.empty((NUM_CPG_STEPS, num_envs), device=DEVICE)
            root_yaw_buf = torch.empty((NUM_CPG_STEPS, num_envs), device=DEVICE)
            path_drift_buf = torch.empty((NUM_CPG_STEPS, num_envs), device=DEVICE)

            accumulated_forces = torch.zeros((num_feet_total, 3), device=DEVICE)
            filtered_forces = torch.zeros((num_feet_total, 3), device=DEVICE)

            # Parallel Episode Stepping
            for step in range(STEPS_PER_EVAL):
                # A. CPG Control Evaluation (Every DECIMATION steps)
                if step % DECIMATION_RATIO == 0:
                    cpg_idx = step // DECIMATION_RATIO
                    ramp = min(1.0, (step + 1) / RAMP_STEPS)
    
                    if step > 0:
                        mean_forces = accumulated_forces / float(DECIMATION_RATIO)
                        accumulated_forces.zero_()
                        filtered_forces = 0.8 * filtered_forces + 0.2 * mean_forces[feet_to_canonical]
                    
                    contact_forces_wp = wp.from_torch(filtered_forces.contiguous(), dtype=wp.vec3)
                    warp_targets = cpg_controller.compute_joint_targets_vec(
                        default_feet_pos_body=default_feet_body_wp,
                        dir_angle_rad=DIRECTION,
                        stride_forward=STRIDE_FWD_CMD,
                        stride_lateral=STRIDE_LAT_CMD,
                        ramp=ramp,
                        contact_forces=contact_forces_wp
                    )

                    torch_targets = wp.to_torch(warp_targets).reshape(num_envs, 18)
                    newton_targets = torch_targets[:, canonical_to_newton]

                    # Tracking telemetry
                    curr_root_pos, curr_root_rot = robot_view.get_world_poses(clone=False)
                    root_z_buf[cpg_idx] = curr_root_pos[:, 2]
                    root_tilt_buf[cpg_idx] = 2.0 * (curr_root_rot[:, 1]**2 + curr_root_rot[:, 2]**2)

                    # Relative quaternion z-component (q_init^-1 * q_curr)_z for pure yaw deviation
                    q_rel_z = (
                        settled_root_rot[:, 0] * curr_root_rot[:, 3]
                        - settled_root_rot[:, 1] * curr_root_rot[:, 2]
                        + settled_root_rot[:, 2] * curr_root_rot[:, 1]
                        - settled_root_rot[:, 3] * curr_root_rot[:, 0]
                    )
                    root_yaw_buf[cpg_idx] = 2.0 * (q_rel_z ** 2)

                    step_dx = curr_root_pos[:, 0] - start_pos[:, 0]
                    step_dy = curr_root_pos[:, 1] - start_pos[:, 1]
                    path_drift_buf[cpg_idx] = torch.abs(step_dx * sin_theta - step_dy * cos_theta)

                # B. Apply Hydrodynamic Forces and CPG Joint Targets
                last_linear_velocity, last_angular_velocity = apply_hydrodynamics_step(
                    last_linear_velocity, last_angular_velocity
                )
                robot_view.set_joint_position_targets(newton_targets)

                # C. Physics Simulation Step
                world.step(render=False)

                # D. Single Unified State & Force Query per Sub-Step
                q_curr = robot_view.get_joint_positions(clone=False)
                q_vels = robot_view.get_joint_velocities(clone=False)

                continuous_forces = feet_view.get_net_contact_forces(dt=SIM_DT, clone=False)
                if continuous_forces.dim() == 3:
                    continuous_forces = continuous_forces.squeeze(0)

                # Accumulate forces for CPG filtering
                accumulated_forces += continuous_forces

                # E. Continuous Work / Energy Integration
                tau_efforts = torch.clamp(
                    joint_kps * (newton_targets - q_curr) - joint_kds * q_vels,
                    -MAX_TORQUE,
                    MAX_TORQUE
                )
                step_power = torch.sum(torch.abs(tau_efforts * q_vels), dim=-1)
                total_energy += step_power * SIM_DT
                torque_sq_accum += torch.mean(tau_efforts ** 2, dim=-1) * SIM_DT

                # F. Filtered Peak Impact & Benthic Normal Load Capture (post-ramp)
                if step >= RAMP_STEPS:
                    post_ramp_steps += 1
                    cf_3d = filtered_forces.reshape(num_envs, 6, 3)
                    fn_mags = torch.norm(cf_3d, dim=-1)
                    peak_impact_forces = torch.maximum(peak_impact_forces, fn_mags.max(dim=1).values)

                    # Track mean vertical seabed contact force (Sum Fz per robot)
                    raw_cf_3d = continuous_forces[feet_to_canonical].reshape(num_envs, 6, 3)
                    sum_fz_accum += raw_cf_3d[:, :, 2].sum(dim=1)

            # ----------------------------------------------------------------------
            # Multi-Objective Fitness Evaluation (Underwater Calibrated)
            # ----------------------------------------------------------------------
            end_poses, _ = robot_view.get_world_poses()
            delta_x = end_poses[:, 0] - start_pos[:, 0]
            delta_y = end_poses[:, 1] - start_pos[:, 1]

            eval_time_s = STEPS_PER_EVAL * SIM_DT

            target_displacement = delta_x * cos_theta + delta_y * sin_theta
            cross_track_drift = torch.abs(delta_x * sin_theta - delta_y * cos_theta)
            mean_path_drift = torch.mean(path_drift_buf, dim=0)
            v_target = target_displacement / eval_time_s

            safe_disp = torch.clamp(target_displacement, min=0.02)
            drift_ratio = torch.clamp(cross_track_drift / safe_disp, max=2.0)
            cost_of_transport = total_energy / (ROBOT_WEIGHT * safe_disp)
            rms_joint_torque = torch.sqrt(torque_sq_accum / eval_time_s)

            sigma_height = torch.std(root_z_buf, dim=0)
            mean_height_drift = torch.abs(torch.mean(root_z_buf, dim=0) - settled_root_pos[:, 2])
            mean_tilt = torch.mean(root_tilt_buf, dim=0)
            mean_yaw_err = torch.mean(root_yaw_buf, dim=0)
            chassis_variability = sigma_height + mean_height_drift + mean_tilt + 2.0 * mean_yaw_err

            # 1. Speed Reward (Saturates at 0.20 m/s)
            speed_reward = 12.0 * torch.clamp(v_target, min=0.0, max=0.20)
            drift_penalty = torch.clamp(
                12.0 * cross_track_drift + 10.0 * mean_path_drift + 4.0 * drift_ratio + 60.0 * mean_yaw_err,
                max=4.0
            )

            # 2. Underwater CoT & Joint Torque Smoothness Penalty
            cot_excess = torch.clamp(cost_of_transport - 0.18, min=0.0)
            cot_penalty = torch.clamp(
                0.5 * cost_of_transport + cot_excess * 3.5 + 0.25 * rms_joint_torque,
                max=5.0
            )

            # 3. Underwater Impact Penalty (Baseline filtered peak is ~24.5 N; penalize > 22.0 N)
            excess_force = torch.clamp(peak_impact_forces - 18.5, min=0.0)
            impact_penalty = torch.clamp(
                0.04 * (peak_impact_forces - 15.0).clamp(min=0.0) + (excess_force / 6.0) ** 1.5,
                max=5.0
            )

            # 4. Benthic Traction / Unloading Penalty (Expected Sum Fz ≈ 17.1 N; penalize < 12.0 N floating/hopping)
            mean_sum_fz = sum_fz_accum / float(max(1, post_ramp_steps))
            unloading_deficit = torch.clamp(12.0 - mean_sum_fz, min=0.0)
            benthic_penalty = torch.clamp(unloading_deficit * 0.4, max=3.0)

            # 5. Stability & Stall Penalties
            chassis_variability_penalty = torch.clamp(chassis_variability * 40.0, max=2.5)
            stall_mask = (target_displacement < 0.25).float()
            stall_penalty = stall_mask * 6.0

            fitness = (
                speed_reward
                - drift_penalty
                - cot_penalty
                - chassis_variability_penalty
                - impact_penalty
                - benthic_penalty
                - stall_penalty
            )

            best_gen_idx = torch.argmax(fitness).item()
            best_gen_fitness = fitness[best_gen_idx].item()
            mean_fitness = fitness.mean().item()
            std_fitness = fitness.std().item()

            best_dist = target_displacement[best_gen_idx].item()
            best_drift = cross_track_drift[best_gen_idx].item()
            best_cot = cost_of_transport[best_gen_idx].item()
            best_impact = peak_impact_forces[best_gen_idx].item()
            best_var = chassis_variability[best_gen_idx].item()
            best_gen_genome = population[best_gen_idx].clone()

            # 1. Update the active stage's local champion
            if best_gen_fitness > stage_best_fitness:
                stage_best_fitness = best_gen_fitness
                current_champion_genome = best_gen_genome.clone()

            # 2. Update the run-wide global record (strictly monotonic)
            if best_gen_fitness > best_overall_fitness:
                best_overall_fitness = best_gen_fitness
                best_overall_genome = best_gen_genome.clone()

            row = [
                global_gen,
                round(best_gen_fitness, 4),
                round(mean_fitness, 4),
                round(std_fitness, 4),
                round(best_dist, 4),
                round(best_drift, 4),
                round(best_cot, 2),
                round(best_impact, 2),
                round(best_var, 5)
            ] + [round(float(val), 5) for val in best_gen_genome]

            with open(CSV_LOG_PATH, mode="a", newline="") as f:
                csv.writer(f).writerow(row)

            save_checkpoint(global_gen, population, fitness, best_overall_genome, best_overall_fitness)

            print(
                f"[{stage_name[:6].upper()}] G{global_gen:03d} (P{stage_gen+1:02d}/{stage_generations}) | "
                f"Fit: {best_gen_fitness:7.4f} (Top: {best_overall_fitness:7.4f}) | "
                f"Dist: {best_dist:5.3f}m | Drift: {best_drift:5.3f}m | CoT: {best_cot:4.2f} | "
                f"Peak F: {best_impact:5.1f}N | Var: {best_var:6.4f}",
                flush=True
            )

            # ----------------------------------------------------------------------
            # Genetic Operations: Crossover, Mutation & Elitism
            # ----------------------------------------------------------------------
            if active_mask.any():
                k_elites = 2
                _, elite_indices = torch.topk(fitness, k=k_elites)
                elites = population[elite_indices].clone()

                p1 = torch.randint(0, num_envs, (num_envs,), device=DEVICE)
                p2 = torch.randint(0, num_envs, (num_envs,), device=DEVICE)
                parent_a = torch.where((fitness[p1] > fitness[p2]).unsqueeze(1), population[p1], population[p2])

                p3 = torch.randint(0, num_envs, (num_envs,), device=DEVICE)
                p4 = torch.randint(0, num_envs, (num_envs,), device=DEVICE)
                parent_b = torch.where((fitness[p3] > fitness[p4]).unsqueeze(1), population[p3], population[p4])

                num_genes = population.shape[1]
                blend_weights = torch.rand((num_envs, num_genes), device=DEVICE) * 1.4 - 0.20
                crossed_offspring = blend_weights * parent_a + (1.0 - blend_weights) * parent_b

                mutation_rate = max(0.02, 0.06 * (1.0 - (stage_gen + 1) / stage_generations))
                raw_noise = torch.randn_like(crossed_offspring) * (bounds_max - bounds_min) * mutation_rate
                masked_noise = raw_noise * active_mask.float().unsqueeze(0)

                offspring = torch.clamp(crossed_offspring + masked_noise, bounds_min, bounds_max)
                if frozen_mask.any():
                    offspring[:, frozen_mask] = current_champion_genome[frozen_mask]
                offspring[:k_elites] = elites
                population = offspring
                cpg_controller.set_population_genomes(population)

    # --------------------------------------------------------------------------
    # Post-Evaluation Summary & Export
    # --------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print(f"[CHAMPION BENCHMARK SUMMARY across {num_envs} Parallel Instances]")
    print("=" * 90)
    print(f"Unified Fitness : {fitness.mean().item():6.2f} ± {fitness.std().item():.2f}")
    print(f"Target Dist (m) : {target_displacement.mean().item():6.3f} ± {target_displacement.std().item():.3f} m")
    print(f"Cross-Track (m) : {cross_track_drift.mean().item():6.3f} ± {cross_track_drift.std().item():.3f} m")
    print(f"Cost of Transp. : {cost_of_transport.mean().item():6.2f} ± {cost_of_transport.std().item():.2f}")
    print(f"Peak Force  (N) : {peak_impact_forces.mean().item():6.1f} ± {peak_impact_forces.std().item():.1f} N")
    print(f"Chassis Var     : {chassis_variability.mean().item():6.4f} ± {chassis_variability.std().item():.4f}")
    print("=" * 90 + "\n")

    export_best_parameters(best_overall_genome, best_overall_fitness, TOTAL_GENERATIONS, num_envs)

    world.stop()
    simulation_app.close()

if __name__ == "__main__":
    main()