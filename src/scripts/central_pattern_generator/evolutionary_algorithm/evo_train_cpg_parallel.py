RENDER_SIMULATION = False  # Headless execution for maximum training throughput

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

import sys
from pathlib import Path

# Resolve path to 'src/scripts' (2 levels up from 'newton/')
SCRIPTS_DIR = Path(__file__).resolve().parents[2]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from silver2_constants.isaac_constants import *
from evo_warp_cpg_controller import VectorizedWarpHexapodCPGController

USD_PATH = "/home/jorge/Documents/Code/silver2_isaacsim/src/scenes/newton/silver2_training_grid.usd"
STAGE_NAME = "silver2_training_grid.usd"

SIM_DT = 1.0 / 420.0
DECIMATION = 4
CPG_DT = SIM_DT * DECIMATION

# --------------------------------------------------------------------------
# Checkpoint & Telemetry Paths
# --------------------------------------------------------------------------
OUTPUT_BASE_DIR = "/home/jorge/Documents/Code/silver2_isaacsim/src/scripts/central_pattern_generator/runs"
RUN_ID = datetime.now().strftime("cpg_ga_%Y%m%d_%H%M%S")
RUN_DIR = os.path.join(OUTPUT_BASE_DIR, RUN_ID)
os.makedirs(RUN_DIR, exist_ok=True)

CSV_LOG_PATH = os.path.join(RUN_DIR, "history.csv")
CHECKPOINT_PATH = os.path.join(RUN_DIR, "checkpoint_latest.pt")
FINAL_JSON_PATH = os.path.join(RUN_DIR, "best_cpg_parameters.json")
FINAL_PT_PATH = os.path.join(RUN_DIR, "best_cpg_genome.pt")

GENOME_PARAM_NAMES = [
    "l2_clearance",
    "l1_preload",
    "k1_stride_gain",
    "k2_splay_gain",
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
            "preload_offset_l1_m": round(g[1], 5),
            "preload_offset_l1_mm": round(g[1] * 1000.0, 2),
            "stride_gain_k1": round(g[2], 4),
            "lateral_splay_gain_k2": round(g[3], 4),
            "vertical_gain_k3": round(g[4], 4),
            "duty_factor_epsilon": round(g[5], 4),
            "gait_period_T_s": round(g[6], 3),
            "coupling_strength_w": round(g[7], 4),
            "sigmoid_slope_b": round(g[8], 3),
            "touchdown_force_F_td_N": round(g[9], 2),
            "release_force_F_rel_N": round(g[10], 2)
        },
        "python_snippet_constants": {
            "CPG_OPTIMIZED_GAINS": {
                "l2": round(g[0], 5),
                "l1": round(g[1], 5),
                "k1": round(g[2], 4),
                "k2": round(g[3], 4),
                "k3": round(g[4], 4),
                "epsilon": round(g[5], 4),
                "period": round(g[6], 3),
                "coupling": round(g[7], 4),
                "b": round(g[8], 3),
                "f_touch": round(g[9], 2),
                "f_release": round(g[10], 2)
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

def main():
    usd_context = omni.usd.get_context()
    if STAGE_NAME not in (usd_context.get_stage_url() or ""):
        usd_context.open_stage(USD_PATH)

    stage = get_current_stage()

    # 1. Setup Newton Physics Scene
    scene_prim = stage.GetPrimAtPath("/physicsScene")
    if not scene_prim.IsValid():
        scene = UsdPhysics.Scene.Define(stage, "/physicsScene")
        scene_prim = scene.GetPrim()
    scene = UsdPhysics.Scene(scene_prim)
    scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr().Set(9.81)
    if not scene_prim.HasAttribute("physics:engine"):
        scene_prim.CreateAttribute("physics:engine", Sdf.ValueTypeNames.Token).Set("Newton")
    else:
        scene_prim.GetAttribute("physics:engine").Set("Newton")

    SimulationManager.switch_physics_engine("newton")
    ns = newton_ext.acquire_stage()
    if ns is not None:
        ns.cfg.solver_cfg.nconmax = 20000
        ns.cfg.solver_cfg.njmax = 12000

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
        "best_forward_dist_m",
        "best_lateral_drift_m",
        "best_peak_impact_N"
    ] + GENOME_PARAM_NAMES

    with open(CSV_LOG_PATH, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(csv_headers)

    world = World.instance()
    if world is None:
        world = World(physics_dt=SIM_DT, stage_units_in_meters=1.0, backend="torch", device=DEVICE)
    world.initialize_physics()

    ns = newton_ext.acquire_stage()
    if ns is not None:
        ns.cfg.solver_cfg.nconmax = 20000
        ns.cfg.solver_cfg.njmax = 12000

    # 3. Vectorized Articulation View
    robot_view = Articulation(
        prim_paths_expr=target_robot_paths,
        name="silver2_population_view",
        reset_xform_properties=False
    )
    world.scene.add(robot_view)

    # 4. Discover the Exact Tibia Link Names and Casing from the First Robot
    first_robot_prim = stage.GetPrimAtPath(target_robot_paths[0])
    discovered_tibia_names = []
    for prim in Usd.PrimRange(first_robot_prim):
        name = prim.GetName()
        # Match only rigid articulation links (not child meshes or joint primitives)
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

    if not world.is_playing():
        world.play()
    world.reset()

    try:
        joint_kps, joint_kds = robot_view.get_gains()
        if joint_kps is None or joint_kps.abs().sum() == 0:
            raise ValueError
    except Exception:
        # Nominal SILVER2 PD drive gains in Newton (stiffness=120.0, damping=5.0)
        joint_kps = torch.full((num_envs, 18), 120.0, device=DEVICE)
        joint_kds = torch.full((num_envs, 18), 5.0, device=DEVICE)

    MAX_JOINT_TORQUE = 60.0  # N*m (SILVER2 peak drive saturation limit)

    # 5. Joint & Foot Hardware Mappings
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
        dt=CPG_DT,
        gait="tripod",
        device=DEVICE
    )
    robot_weight_n = SILVER2_TOTAL_MASS * 9.81

    standing_rad = np.deg2rad(np.array(SILVER2_STANDING_ANGLES_DEG, dtype=np.float32).flatten())
    standing_tensor = torch.as_tensor(standing_rad, dtype=torch.float32, device=DEVICE)
    standing_targets_pop = standing_tensor[canonical_to_newton].unsqueeze(0).repeat(num_envs, 1)

    robot_view.set_joint_positions(standing_targets_pop)
    robot_view.set_joint_velocities(torch.zeros_like(standing_targets_pop))

    print(f"[INFO] Settling {num_envs} parallel robots across 500 physics steps...")
    for _ in range(500):
        robot_view.set_joint_position_targets(standing_targets_pop)
        world.step(render=False)

    # Save stable settled spawn poses to reset floating base every generation
    default_root_pos, default_root_rot = robot_view.get_world_poses()
    default_root_pos = default_root_pos.clone()
    default_root_rot = default_root_rot.clone()

    settled_q = robot_view.get_joint_positions()[:, newton_to_canonical].cpu().numpy()
    angles_rad = np.deg2rad(np.rad2deg(settled_q[0]).reshape(6, 3)).astype(np.float32)
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

    # --------------------------------------------------------------------------
    # 7. Evolutionary Optimization (11 Parameters)
    # Genome: [l2, l1, k1, k2, k3, epsilon, total_period, coupling_strength, b, f_touch, f_release]
    # --------------------------------------------------------------------------
    GENOME_MIN = torch.tensor([0.005, -0.025, 0.50, 0.00, 0.50, 0.50, 1.00, 0.05, 1.0, 15.0, 2.0], device=DEVICE)
    GENOME_MAX = torch.tensor([0.045,  0.000, 1.60, 0.40, 1.60, 0.70, 2.50, 0.80, 5.0, 50.0, 15.0], device=DEVICE)

    CURRICULUM_MASKS = {
        # Stage 1: Spatial footprint & stroke [l2, l1, k1, k2, k3]
        "stage1_spatial": torch.tensor(
            [True, True, True, True, True, False, False, False, False, False, False],
            dtype=torch.bool, device=DEVICE
        ),
        # Stage 2: Oscillator cadence & gait rhythm [epsilon, T, w, b]
        "stage2_temporal": torch.tensor(
            [False, False, False, False, False, True, True, True, True, False, False],
            dtype=torch.bool, device=DEVICE
        ),
        # Stage 3: Force thresholds for contact reflexes [F_td, F_rel]
        "stage3_sensory": torch.tensor(
            [False, False, False, False, False, False, False, False, False, True, True],
            dtype=torch.bool, device=DEVICE
        ),
    }

    # Reference baseline genome (used to hold inactive stages static)
    BASELINE_GENOME = torch.tensor([
        0.01,   # l2 (m)
        -0.01,  # l1 (m)
        1.00,   # k1
        0.00,   # k2
        1.00,   # k3
        0.50,   # epsilon
        2.00,   # period T (s)
        0.40,   # coupling w
        2.00,   # slope b
        15.00,  # F_td (N)
        5.00    # F_rel (N)
    ], device=DEVICE)

    # Select active stage: "stage1_spatial" | "stage2_temporal" | "stage3_sensory"
    ACTIVE_STAGE = "stage1_spatial"
    active_mask = CURRICULUM_MASKS[ACTIVE_STAGE]
    frozen_mask = ~active_mask

    # Broadcast baseline across entire population [num_envs, 11]
    population = BASELINE_GENOME.repeat(num_envs, 1)

    # Randomize only active parameters within their respective bounds
    random_active = GENOME_MIN + torch.rand((num_envs, 11), device=DEVICE) * (GENOME_MAX - GENOME_MIN)
    population = torch.where(active_mask.unsqueeze(0), random_active, population)

    # Seed Individual 0 with the baseline (or previous stage champion)
    population[0] = BASELINE_GENOME.clone()
    cpg_controller.set_population_genomes(population)

    NUM_GENERATIONS = 50
    STEPS_PER_EVAL = 1500

    best_overall_fitness = -float("inf")
    best_overall_genome = population[0].clone()

    print(f"\n[INFO] Starting Genetic Algorithm: {NUM_GENERATIONS} generations x {num_envs} parallel robots")
    print("=" * 115)

    # --------------------------------------------------------------------------
    # 8. Optimization Loop
    # --------------------------------------------------------------------------
    for gen in range(NUM_GENERATIONS):
        # 1. Reset floating base pose, velocities, and motor targets
        robot_view.set_world_poses(positions=default_root_pos, orientations=default_root_rot)
        robot_view.set_velocities(torch.zeros((num_envs, 6), device=DEVICE))
        robot_view.set_joint_positions(standing_targets_pop)
        robot_view.set_joint_velocities(torch.zeros_like(standing_targets_pop))
        cpg_controller.reset_all_states()

        start_poses, _ = robot_view.get_world_poses()
        start_pos = start_poses.clone()
        peak_impact_forces = torch.zeros(num_envs, device=DEVICE)
        total_energy = torch.zeros(num_envs, device=DEVICE)

        # 2. Parallel Episode Stepping
        for step in range(STEPS_PER_EVAL):
            if step % DECIMATION == 0:
                ramp = min(1.0, (step + 1) / 200.0)

                raw_forces = feet_view.get_net_contact_forces(clone=False)
                if raw_forces.dim() == 3:
                    raw_forces = raw_forces.squeeze(0)

                canonical_forces = raw_forces[feet_to_canonical].contiguous()
                contact_forces_wp = wp.from_torch(canonical_forces, dtype=wp.vec3)

                warp_targets = cpg_controller.compute_joint_targets_vec(
                    default_feet_pos_body=default_feet_body_wp,
                    dir_angle_rad=0.0,
                    stride_forward=0.005,
                    stride_lateral=0.005,
                    ramp=ramp,
                    contact_forces=contact_forces_wp
                )

                torch_targets = wp.to_torch(warp_targets).reshape(num_envs, 18)
                newton_targets = torch_targets[:, canonical_to_newton]

                # Accumulate mechanical work
                q_curr = robot_view.get_joint_positions(clone=False)
                q_vels = robot_view.get_joint_velocities(clone=False)

                # Attempt native applied efforts; fall back to analytical PD drive torque
                tau_efforts = None
                if hasattr(robot_view, "get_applied_joint_efforts"):
                    try:
                        tau_efforts = robot_view.get_applied_joint_efforts(clone=False)
                    except Exception:
                        tau_efforts = None

                if tau_efforts is None or tau_efforts.abs().sum() == 0:
                    # Analytical motor effort: tau = clamp(Kp*(q_des - q) - Kd*q_dot, -tau_max, tau_max)
                    tau_efforts = torch.clamp(
                        joint_kps * (newton_targets - q_curr) - joint_kds * q_vels,
                        -MAX_JOINT_TORQUE,
                        MAX_JOINT_TORQUE
                    )

                # Instantaneous mechanical power: P = sum(|tau * q_dot|) (Watts)
                step_power = torch.sum(torch.abs(tau_efforts * q_vels), dim=-1)
                
                # Integrate energy: dE = P * dt (Joules)
                dt_step = DECIMATION * SIM_DT
                total_energy += step_power * dt_step

                if step >= 40:
                    fn_mags = torch.norm(canonical_forces.reshape(num_envs, 6, 3), dim=-1)
                    peak_impact_forces = torch.maximum(peak_impact_forces, fn_mags.max(dim=1).values)

            robot_view.set_joint_position_targets(newton_targets)
            world.step(render=False)

        # 3. Multi-Objective Scale-Aligned Fitness Evaluation
        end_poses, end_quats = robot_view.get_world_poses()
        forward_displacement = end_poses[:, 0] - start_pos[:, 0]
        lateral_drift = torch.abs(end_poses[:, 1] - start_pos[:, 1])

        # Compute chassis tilt penalty from base quaternion (w, x, y, z)
        # Up-vector projection on global Z axis: R_33 = 1 - 2*(x^2 + y^2)
        tilt_deviation = 2.0 * (end_quats[:, 1]**2 + end_quats[:, 2]**2)  # 0 when upright, >0 when pitched/rolled
        tilt_penalty = torch.clamp(tilt_deviation * 10.0, max=5.0)

        # Impact penalty based on true physical Newtons (nominal threshold 350 N)
        excess_force = torch.clamp(peak_impact_forces - 350.0, min=0.0)
        impact_penalty = torch.clamp(excess_force / 50.0, max=5.0)

        # Cost of transport
        safe_distance = torch.clamp(forward_displacement, min=0.05)
        cost_of_transport = total_energy / (robot_weight_n * safe_distance)
        # Penalty: Baseline smooth walking is ~1.5 - 3.5; hopping spikes to > 8.0
        cot_excess = torch.clamp(cost_of_transport - 2.5, min=0.0)
        cot_penalty = torch.clamp(cot_excess * 1.5, max=10.0)
        """
        # Total fitness score
        fitness = (
            forward_displacement * 10.0   # Dominant drive: 3 m -> +30 pts
            - lateral_drift * 5.0          # Drift penalty
            - tilt_penalty                 # Chassis stability penalty (max -5 pts)
            - impact_penalty              # Ground collision penalty (max -5 pts)
        )
        """

        # Stage 1 Fitness
        eval_time_s = STEPS_PER_EVAL * SIM_DT
        # Forward velocity in m/s
        v_forward = forward_displacement / eval_time_s
        # Straightness: ratio of lateral drift to forward progress (bounded)
        safe_fwd = torch.clamp(forward_displacement, min=0.10)
        drift_ratio = torch.clamp(lateral_drift / safe_fwd, max=2.0)
        # Dimensionless Cost of Transport
        cost_of_transport = total_energy / (robot_weight_n * safe_fwd)
        # Bounded Stage 1 fitness (Clean magnitude: nominal ~0.0 to 1.0)
        fitness = (
            v_forward              # Direct forward speed (0.5 to 1.2 pts)
            - 0.5 * drift_ratio    # Drift penalty (0.0 straight, up to -1.0)
            - 0.1 * cost_of_transport # Energy efficiency penalty (~ -0.2 to -0.4 for good walks)
        )

        best_gen_idx = torch.argmax(fitness).item()
        best_gen_fitness = fitness[best_gen_idx].item()
        mean_fitness = fitness.mean().item()
        std_fitness = fitness.std().item()

        best_dist = forward_displacement[best_gen_idx].item()
        best_drift = lateral_drift[best_gen_idx].item()
        best_cot = cost_of_transport[best_gen_idx].item()
        best_impact = peak_impact_forces[best_gen_idx].item()
        best_gen_genome = population[best_gen_idx].clone()

        if best_gen_fitness > best_overall_fitness:
            best_overall_fitness = best_gen_fitness
            best_overall_genome = best_gen_genome.clone()

        # 4. Stream Metrics to CSV Log
        row = [
            gen + 1,
            round(best_gen_fitness, 4),
            round(mean_fitness, 4),
            round(std_fitness, 4),
            round(best_dist, 4),
            round(best_drift, 4),
            round(best_impact, 2)
        ] + [round(float(val), 5) for val in best_gen_genome]

        with open(CSV_LOG_PATH, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)

        # 5. Save Rolling Checkpoint
        save_checkpoint(gen + 1, population, fitness, best_overall_genome, best_overall_fitness)

        print(
            f"Gen {gen+1:02d}/{NUM_GENERATIONS} | "
            f"Fit: {best_gen_fitness:6.2f} (All-Time: {best_overall_fitness:6.2f}) | "
            f"Dist: {best_dist:5.2f} m | Drift: {best_drift:4.2f} m | "
            f"CoT: {best_cot:5.2f} | Peak F: {best_impact:5.1f} N"
        )

        # 6. Selection, Crossover & Mutation for Next Generation
        k_elites = 8
        _, elite_indices = torch.topk(fitness, k=k_elites)
        elites = population[elite_indices].clone()

        # Binary Tournament Selection
        p1 = torch.randint(0, num_envs, (num_envs,), device=DEVICE)
        p2 = torch.randint(0, num_envs, (num_envs,), device=DEVICE)
        parent_a = torch.where((fitness[p1] > fitness[p2]).unsqueeze(1), population[p1], population[p2])

        p3 = torch.randint(0, num_envs, (num_envs,), device=DEVICE)
        p4 = torch.randint(0, num_envs, (num_envs,), device=DEVICE)
        parent_b = torch.where((fitness[p3] > fitness[p4]).unsqueeze(1), population[p3], population[p4])

        # Per-Gene BLX-alpha (alpha=0.15)
        num_genes = population.shape[1]
        blend_weights = torch.rand((num_envs, num_genes), device=DEVICE) * 1.3 - 0.15
        crossed_offspring = blend_weights * parent_a + (1.0 - blend_weights) * parent_b

        offspring = torch.where(active_mask.unsqueeze(0), crossed_offspring, BASELINE_GENOME.unsqueeze(0))

        # Annealed mutation rate with exploration floor
        mutation_rate = max(0.02, 0.12 * (1.0 - (gen + 1) / NUM_GENERATIONS))
        raw_noise = torch.randn_like(offspring) * (GENOME_MAX - GENOME_MIN) * mutation_rate

        # Zero out mutation noise for all frozen parameters
        masked_noise = raw_noise * active_mask.float().unsqueeze(0)

        # Apply noise and clamp
        offspring = torch.clamp(offspring + masked_noise, GENOME_MIN, GENOME_MAX)

        # Re-enforce exact baseline values on frozen parameters to eliminate precision drift
        offspring[:, frozen_mask] = BASELINE_GENOME[frozen_mask]

        # Preserve top elites without mutation
        offspring[:k_elites] = elites
        population = offspring
        cpg_controller.set_population_genomes(population)

    # --------------------------------------------------------------------------
    # 9. Export Optimal Genome Configuration
    # --------------------------------------------------------------------------
    export_best_parameters(best_overall_genome, best_overall_fitness, NUM_GENERATIONS, num_envs)

    world.stop()
    simulation_app.close()


if __name__ == "__main__":
    main()