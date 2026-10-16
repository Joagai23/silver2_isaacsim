import json
from pathlib import Path
from typing import Sequence, Union
import warp as wp
from warp_hydrodynamics import solve_hydrodynamics_kernel

class WarpHydrodynamicsWrapper:
    """
    A GPU-accelerated wrapper that stores physical constants in Warp arrays
    and calls the Warp kernel for calculation.
    """
    def __init__(
        self,
        width: Union[float, Sequence[float]],
        depth: Union[float, Sequence[float]],
        height: Union[float, Sequence[float]],
        linear_drag_coefficient: Union[float, Sequence[float]],
        angular_drag_coefficient: Union[float, Sequence[float]],
        linear_damping: Union[float, Sequence[float]],
        angular_damping: Union[float, Sequence[float]],
        linear_mass_coeff: Union[float, Sequence[float]],
        angular_mass_coeff: Union[float, Sequence[float]],
        lift_coefficient: Union[float, Sequence[float]],
        water_density: float = 1025.0,
        gravity: float = 9.81,
        num_envs: int = 1,
        device: str = "cuda:0",
    ):
        
        self.device = device
        self.num_envs = num_envs
        self.graph = None

        # Normalize scalar vs per-link sequence inputs
        def _to_list(val):
            return [float(v) for v in val] if isinstance(val, (list, tuple)) else [float(val)]

        def _broadcast(lst, name):
            if len(lst) == 1 and self.num_links > 1:
                return lst * self.num_links
            if len(lst) != self.num_links:
                raise ValueError(f"Expected 1 or {self.num_links} values for {name}, got {len(lst)}")
            return lst

        widths = _to_list(width)
        depths = _to_list(depth)
        heights = _to_list(height)
        lin_drags = _to_list(linear_drag_coefficient)
        ang_drags = _to_list(angular_drag_coefficient)
        lin_damps = _to_list(linear_damping)
        ang_damps = _to_list(angular_damping)
        lin_masses = _to_list(linear_mass_coeff)
        ang_masses = _to_list(angular_mass_coeff)
        lifts = _to_list(lift_coefficient)

        self.num_links = max(
            len(widths), len(depths), len(heights),
            len(lin_drags), len(ang_drags), len(lin_damps),
            len(ang_damps), len(lin_masses), len(ang_masses), len(lifts)
        )

        widths = _broadcast(widths, "width")
        depths = _broadcast(depths, "depth")
        heights = _broadcast(heights, "height")
        lin_drags = _broadcast(lin_drags, "linear_drag_coefficient")
        ang_drags = _broadcast(ang_drags, "angular_drag_coefficient")
        lin_damps = _broadcast(lin_damps, "linear_damping")
        ang_damps = _broadcast(ang_damps, "angular_damping")
        lin_masses = _broadcast(lin_masses, "linear_mass_coeff")
        ang_masses = _broadcast(ang_masses, "angular_mass_coeff")
        lifts = _broadcast(lifts, "lift_coefficient")

        self.total_bodies = self.num_envs * self.num_links

        # Pre-allocate fixed-address input buffers for CUDA Graph
        self.in_position = wp.zeros(self.total_bodies, dtype=wp.vec3, device=device)
        self.in_orientation = wp.zeros(self.total_bodies, dtype=wp.quat, device=device)
        self.in_lin_vel = wp.zeros(self.total_bodies, dtype=wp.vec3, device=device)
        self.in_ang_vel = wp.zeros(self.total_bodies, dtype=wp.vec3, device=device)
        self.in_lin_acc = wp.zeros(self.total_bodies, dtype=wp.vec3, device=device)
        self.in_ang_acc = wp.zeros(self.total_bodies, dtype=wp.vec3, device=device)

        # Shared local face normals for any axis-aligned local box
        local_face_normals = [
            [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0], [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0], [0.0, 0.0, -1.0],
        ]
        self.wp_normals = wp.array(local_face_normals, dtype=wp.vec3, device=device)
        
        # Build per-link geometry & physics tables
        all_keypoints = []
        all_areas = []
        all_centers = []
        all_params = []
        all_added_mass = []

        nx, ny, nz = 3, 3, 3

        for _ in range(self.num_envs):
            for idx in range(self.num_links):
                w, d, h = widths[idx], depths[idx], heights[idx]
                x, y, z = w / 2.0, d / 2.0, h / 2.0
                vol = w * d * h
                displaced_mass = vol * water_density

                # 27 Uniform Voxel Centroids for this link
                link_kp = []
                for i in range(nx):
                    px = -x + (i + 0.5) * (w / nx)
                    for j in range(ny):
                        py = -y + (j + 0.5) * (d / ny)
                        for k in range(nz):
                            pz = -z + (k + 0.5) * (h / nz)
                            link_kp.append([px, py, pz])
                all_keypoints.append(link_kp)

                # Face Areas and Face Centers for this link
                all_areas.append([d * h, d * h, w * h, w * h, w * d, w * d])
                all_centers.append([
                    [x, 0.0, 0.0], [-x, 0.0, 0.0],
                    [0.0, y, 0.0], [0.0, -y, 0.0],
                    [0.0, 0.0, z], [0.0, 0.0, -z],
                ])

                # Parameters Vector
                all_params.append([
                    water_density,
                    gravity,
                    lin_drags[idx],
                    ang_drags[idx],
                    lin_damps[idx],
                    ang_damps[idx],
                    lifts[idx],
                    vol,
                ])

                # Anisotropic Linear & Rotational Added Mass
                lm_c = lin_masses[idx]
                am_c = ang_masses[idx]

                ref_area = (vol ** (2.0 / 3.0)) + 1e-6
                added_mass_x = displaced_mass * lm_c * ((d * h) / ref_area)
                added_mass_y = displaced_mass * lm_c * ((w * h) / ref_area)
                added_mass_z = displaced_mass * lm_c * ((w * d) / ref_area)
                lin_am_vec = [added_mass_x, added_mass_y, added_mass_z]

                ang_am_x = (1.0 / 12.0) * displaced_mass * (d**2 + h**2) * am_c
                ang_am_y = (1.0 / 12.0) * displaced_mass * (w**2 + h**2) * am_c
                ang_am_z = (1.0 / 12.0) * displaced_mass * (w**2 + d**2) * am_c
                ang_am_vec = [ang_am_x, ang_am_y, ang_am_z]

                all_added_mass.append([lin_am_vec, ang_am_vec])
        
        # Allocate 2D Per-Link GPU Arrays
        self.wp_keypoints = wp.array(all_keypoints, dtype=wp.vec3, device=device)
        self.wp_areas = wp.array(all_areas, dtype=wp.float32, device=device)
        self.wp_centers = wp.array(all_centers, dtype=wp.vec3, device=device)
        self.wp_params = wp.array(all_params, dtype=wp.float32, device=device)
        self.wp_added_mass = wp.array(all_added_mass, dtype=wp.vec3, device=device)

        # Output Buffers: shape (total_bodies,)
        self.out_buoyancy_force = wp.zeros(self.total_bodies, dtype=wp.vec3, device=device)
        self.out_drag_force = wp.zeros(self.total_bodies, dtype=wp.vec3, device=device)
        self.out_lift_force = wp.zeros(self.total_bodies, dtype=wp.vec3, device=device)
        self.out_drag_torque = wp.zeros(self.total_bodies, dtype=wp.vec3, device=device)
        self.out_added_mass_force = wp.zeros(self.total_bodies, dtype=wp.vec3, device=device)
        self.out_added_mass_torque = wp.zeros(self.total_bodies, dtype=wp.vec3, device=device)
        self.out_center_of_buoyancy = wp.zeros(self.total_bodies, dtype=wp.vec3, device=device)
        self.out_center_of_pressure = wp.zeros(self.total_bodies, dtype=wp.vec3, device=device)

        # Pre-bind zero-copy PyTorch tensor views
        self.t_out_buoyancy_force = wp.to_torch(self.out_buoyancy_force)
        self.t_out_drag_force = wp.to_torch(self.out_drag_force)
        self.t_out_lift_force = wp.to_torch(self.out_lift_force)
        self.t_out_drag_torque = wp.to_torch(self.out_drag_torque)
        self.t_out_added_mass_force = wp.to_torch(self.out_added_mass_force)
        self.t_out_added_mass_torque = wp.to_torch(self.out_added_mass_torque)
        self.t_out_center_of_buoyancy = wp.to_torch(self.out_center_of_buoyancy)
        self.t_out_center_of_pressure = wp.to_torch(self.out_center_of_pressure)

    @classmethod
    def from_json(
        cls,
        json_path: Union[str, Path],
        link_names_or_paths: Sequence[str],
        num_envs: int = 1,
        device: str = "cuda:0",
    ) -> "WarpHydrodynamicsWrapper":
        """
        Builds a multi-link WarpHydrodynamicsWrapper directly from the SILVER2 JSON configuration
        and a list of link names or USD prim paths (e.g., from `rigid_prim.paths`).
        """
        with open(json_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        globals_cfg = cfg.get("globals", {})
        water_density = float(globals_cfg.get("waterDensity", 1025.0))
        gravity = float(globals_cfg.get("gravity", 9.81))
        parts_cfg = cfg["parts"]

        widths, depths, heights = [], [], []
        lin_drags, ang_drags = [], []
        lin_damps, ang_damps = [], []
        lin_masses, ang_masses = [], []
        lifts = []

        for path_or_name in link_names_or_paths:
            prim_name = path_or_name.split("/")[-1].lower()
            matched_part = None
            for part_key, part_data in parts_cfg.items():
                if part_key.lower() in prim_name:
                    matched_part = part_data
                    break

            if matched_part is None:
                raise KeyError(
                    f"Could not match link '{path_or_name}' to any part key in {list(parts_cfg.keys())}"
                )

            widths.append(matched_part["xDimension"])
            depths.append(matched_part["yDimension"])
            heights.append(matched_part["zDimension"])
            lin_drags.append(matched_part["linearDragCoefficient"])
            ang_drags.append(matched_part["angularDragCoefficient"])
            lin_damps.append(matched_part["linearDamping"])
            ang_damps.append(matched_part["angularDamping"])
            lin_masses.append(matched_part["linearAddedMassCoefficient"])
            ang_masses.append(matched_part["angularAddedMassCoefficient"])
            lifts.append(matched_part["liftCoefficient"])

        return cls(
            width=widths,
            depth=depths,
            height=heights,
            linear_drag_coefficient=lin_drags,
            angular_drag_coefficient=ang_drags,
            linear_damping=lin_damps,
            angular_damping=ang_damps,
            linear_mass_coeff=lin_masses,
            angular_mass_coeff=ang_masses,
            lift_coefficient=lifts,
            water_density=water_density,
            gravity=gravity,
            num_envs=num_envs,
            device=device,
        )

    def calculate_hydrodynamic_forces(
        self,
        t_position,
        t_orientation,
        t_linear_velocity,
        t_angular_velocity,
        t_linear_acceleration,
        t_angular_acceleration,
    ):
        """
        Copies PyTorch input tensors of shape (total_bodies, 3) / (total_bodies, 4) into
        fixed CUDA Graph buffers, replays the graph, and returns zero-copy PyTorch tensor views.
        """
        self.in_position.assign(wp.from_torch(t_position, dtype=wp.vec3))
        self.in_orientation.assign(wp.from_torch(t_orientation, dtype=wp.quat))
        self.in_lin_vel.assign(wp.from_torch(t_linear_velocity, dtype=wp.vec3))
        self.in_ang_vel.assign(wp.from_torch(t_angular_velocity, dtype=wp.vec3))
        self.in_lin_acc.assign(wp.from_torch(t_linear_acceleration, dtype=wp.vec3))
        self.in_ang_acc.assign(wp.from_torch(t_angular_acceleration, dtype=wp.vec3))

        # Use CUDA graphs
        if self.graph is None:
            wp.capture_begin(device=self.device)
            wp.launch(
                kernel=solve_hydrodynamics_kernel,
                dim=self.total_bodies,
                inputs=[
                    self.in_position,
                    self.in_orientation,
                    self.in_lin_vel,
                    self.in_ang_vel,
                    self.in_lin_acc,
                    self.in_ang_acc,
                    self.wp_keypoints,
                    self.wp_normals,
                    self.wp_centers,
                    self.wp_areas,
                    self.wp_params,
                    self.wp_added_mass,
                ],
                outputs=[
                    self.out_buoyancy_force,
                    self.out_drag_force,
                    self.out_lift_force,
                    self.out_drag_torque,
                    self.out_added_mass_force,
                    self.out_added_mass_torque,
                    self.out_center_of_buoyancy,
                    self.out_center_of_pressure,
                ],
                device=self.device,
            )
            self.graph = wp.capture_end(device=self.device)

        wp.capture_launch(self.graph)
        
        return (
            self.t_out_buoyancy_force,
            self.t_out_drag_force,
            self.t_out_lift_force,
            self.t_out_drag_torque,
            self.t_out_added_mass_force,
            self.t_out_added_mass_torque,
            self.t_out_center_of_buoyancy,
            self.t_out_center_of_pressure,
        )