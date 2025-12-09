import carb
import omni.kit.window.property
import torch
import omni.physx
import warp as wp
import json
import os
from isaacsim.replicator.behavior.global_variables import EXPOSED_ATTR_NS
from isaacsim.replicator.behavior.utils.behavior_utils import (
    check_if_exposed_variables_should_be_removed,
    create_exposed_variables,
    get_exposed_variable,
    remove_exposed_variables,
)
from omni.kit.scripting import BehaviorScript
from pxr import Sdf, UsdPhysics
from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.simulation_context import SimulationContext
from .warp_hydrodynamics_wrapper import WarpHydrodynamicsWrapper

class HydrodynamicsBehavior(BehaviorScript):
    """
    Behavior script that applies hydrodynamics using a Zero-Copy GPU Pipeline.
    Data flow: Isaac Sim (GPU) -> PyTorch (GPU) -> Warp (GPU) -> Isaac Sim (GPU).
    """
    BEHAVIOR_NS = "hydrodynamicsBehavior"

    VARIABLES_TO_EXPOSE = [
        # General Properties
        {"attr_name": "waterDensity", "attr_type": Sdf.ValueTypeNames.Float, "default_value": 1025.0, "doc": "Density of the fluid in kg/m^3."},
        {"attr_name": "gravity", "attr_type": Sdf.ValueTypeNames.Float, "default_value": 9.81, "doc": "Gravitational acceleration in m/s^2."},
        # Object Dimensions
        {"attr_name": "xDimension", "attr_type": Sdf.ValueTypeNames.Float, "default_value": 1.0, "doc": "Object dimension along its local X-axis (m)."},
        {"attr_name": "yDimension", "attr_type": Sdf.ValueTypeNames.Float, "default_value": 1.0, "doc": "Object dimension along its local Y-axis (m)."},
        {"attr_name": "zDimension", "attr_type": Sdf.ValueTypeNames.Float, "default_value": 1.0, "doc": "Object dimension along its local Z-axis (m)."},
        # Drag Properties
        {"attr_name": "linearDragCoefficient", "attr_type": Sdf.ValueTypeNames.Float, "default_value": 1.2, "doc": "Quadratic linear drag coefficient (Cd)."},
        {"attr_name": "angularDragCoefficient", "attr_type": Sdf.ValueTypeNames.Float, "default_value": 0.8, "doc": "Quadratic angular drag coefficient."},
        {"attr_name": "linearDamping", "attr_type": Sdf.ValueTypeNames.Float, "default_value": 300.0, "doc": "Linear damping multiplier for low-speed stability."},
        {"attr_name": "angularDamping", "attr_type": Sdf.ValueTypeNames.Float, "default_value": 150.0, "doc": "Linear angular damping for low-speed stability."},
        # Added Mass Properties
        {"attr_name": "linearAddedMassCoefficient", "attr_type": Sdf.ValueTypeNames.Float, "default_value": 0.05, "doc": "Added mass coefficient for surge, sway, and heave acceleration."},
        {"attr_name": "angularAddedMassCoefficient", "attr_type": Sdf.ValueTypeNames.Float, "default_value": 0.02, "doc": "Added mass coefficient for roll, pitch, and yaw acceleration."},
        # Lift Properties
        {"attr_name": "liftCoefficient", "attr_type": Sdf.ValueTypeNames.Float, "default_value": 1.0, "doc": "A multiplier for the overall strength of the lift force."}
    ]

    def on_init(self):
        # Initialize Simulation Context with Torch Backend
        self._device = "cuda:0"
        self._sim_context = SimulationContext(backend="torch", device=self._device)

        # Initialize class variables
        self._hydro_calculator = None
        self._rigid_prim_view = None
        self._last_linear_velocity = None
        self._last_angular_velocity = None
        
        # Physics-based update
        self._physx_subscription = None
        # Initialize Warp Global
        try:
            if not wp.is_mempool_enabled(device=self._device):
                wp.init()
        except Exception as e:
            carb.log_error(f"Failed to initialize NVIDIA Warp: {e}")

        create_exposed_variables(self.prim, EXPOSED_ATTR_NS, self.BEHAVIOR_NS, self.VARIABLES_TO_EXPOSE)
        self._apply_json_config()
        omni.kit.window.property.get_window().request_rebuild()

    def _apply_json_config(self):
        """
        Loads config and matches Prim name (e.g. 'Coxa_0') to JSON category (e.g. 'coxa').
        """
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "hydrodynamics_config.json")

        if not os.path.exists(config_path):
            carb.log_warn(f"[Hydro] Config missing at {config_path}")
            return

        try:
            with open(config_path, 'r') as f:
                data = json.load(f)
            
            if "globals" in data:
                for k, v in data["globals"].items():
                    self._set_attr(k, v)

            prim_name = self.prim.GetName().lower()
            part_type = None
            
            if "parts" in data:
                for category in data["parts"].keys():
                    if category.lower() in prim_name:
                        part_type = category
                        break
            
            if part_type is None and "body" in prim_name:
                 part_type = "body"

            if part_type:
                carb.log_info(f"[Hydro] Config: Applied '{part_type}' settings to {self.prim.GetName()}")
                part_data = data["parts"][part_type]
                for k, v in part_data.items():
                    self._set_attr(k, v)
            else:
                carb.log_warn(f"[Hydro] Config: No matching part found for {self.prim.GetName()}. Using defaults.")

        except Exception as e:
            carb.log_error(f"[Hydro] JSON Error: {e}")
    
    def _set_attr(self, name, value):
        full_attr_name = f"{EXPOSED_ATTR_NS}:{self.BEHAVIOR_NS}:{name}"
        attr = self.prim.GetAttribute(full_attr_name)
        
        if attr and attr.IsValid():
            attr.Set(float(value))
        else:
            carb.log_warn(f"[Hydro] Failed to set attribute: {full_attr_name}")

    def on_destroy(self):
        self._reset()
        if check_if_exposed_variables_should_be_removed(self.prim, __file__):
            remove_exposed_variables(self.prim, EXPOSED_ATTR_NS, self.BEHAVIOR_NS, self.VARIABLES_TO_EXPOSE)
            omni.kit.window.property.get_window().request_rebuild()

    def on_play(self):
        self._setup()
        physx_interface = omni.physx.get_physx_interface()
        self._physx_subscription = physx_interface.subscribe_physics_step_events(self._on_physics_step)

    def on_stop(self):
        self._reset()

    # FixedUpdate
    def _on_physics_step(self, delta_time:float):
        if delta_time <= 1e-6 or self._rigid_prim_view is None or not self._rigid_prim_view.is_valid():
            return
        self._apply_behavior(delta_time)

    def _setup(self):
        if not self.prim.HasAPI(UsdPhysics.RigidBodyAPI):
            carb.log_warn(f"HydrodynamicsBehavior on prim {self.prim_path} requires a RigidBody component.")
            return
        
        view_name = f"hydro_view_{self.prim_path.name}"
        self._rigid_prim_view = RigidPrimView(
            prim_paths_expr=str(self.prim_path),
            name=view_name
        )
        self._rigid_prim_view.initialize()

        self._hydro_calculator = WarpHydrodynamicsWrapper(
            width=self._get_exposed_variable("xDimension"), 
            depth=self._get_exposed_variable("yDimension"), 
            height=self._get_exposed_variable("zDimension"),
            linear_drag_coefficient=self._get_exposed_variable("linearDragCoefficient"),
            angular_drag_coefficient=self._get_exposed_variable("angularDragCoefficient"),
            linear_damping=self._get_exposed_variable("linearDamping"),
            angular_damping=self._get_exposed_variable("angularDamping"),
            water_density=self._get_exposed_variable("waterDensity"), 
            gravity=self._get_exposed_variable("gravity"),
            linear_mass_coeff = self._get_exposed_variable("linearAddedMassCoefficient"),
            angular_mass_coeff = self._get_exposed_variable("angularAddedMassCoefficient"),
            lift_coefficient=self._get_exposed_variable("liftCoefficient"),
            device=self._device
        )

        # Fetch mass to avoid explosions
        masses = self._rigid_prim_view.get_masses(clone=False)
        self._mass = masses[0]
        carb.log_info(f"HydrodynamicsBehavior (Warp GPU) initialized for {self.prim_path}")

    def _apply_behavior(self, delta_time):
        try:
            positions, orientations = self._rigid_prim_view.get_world_poses(clone=False)
            full_velocities = self._rigid_prim_view.get_velocities(clone=False)
            
            if full_velocities is None or full_velocities.shape[0] == 0:
                return

            positions = positions.to(self._device)
            orientations = orientations.to(self._device)
            full_velocities = full_velocities.to(self._device)
            
            linear_velocity = full_velocities[:, 0:3]
            angular_velocity = full_velocities[:, 3:6]
            
        except (UnboundLocalError, IndexError, RuntimeError, AttributeError):
            return

        orientation = orientations[:, [1, 2, 3, 0]]

        if self._last_linear_velocity is None:
            self._last_linear_velocity = torch.zeros_like(linear_velocity, device=self._device)
            self._last_angular_velocity = torch.zeros_like(angular_velocity, device=self._device)

        # Estimate acceleration
        linear_acceleration = (linear_velocity - self._last_linear_velocity) / delta_time
        angular_acceleration = (angular_velocity - self._last_angular_velocity) / delta_time

        # Run GPU Kernel 
        (t_buoyancy_force, t_drag_force, t_lift_force, t_drag_torque, t_added_mass_force, 
         t_added_mass_torque, t_center_of_buoyancy, t_center_of_pressure) = self._hydro_calculator.calculate_hydrodynamic_forces(
            positions, orientation, linear_velocity, angular_velocity, 
            linear_acceleration, angular_acceleration
        )

        # Calculate torques generated by off-center forces
        t_torque_buoyancy = torch.cross(t_center_of_buoyancy - positions, t_buoyancy_force, dim=-1)
        t_torque_drag = torch.cross(t_center_of_pressure - positions, t_drag_force, dim=-1)
        t_torque_lift = torch.cross(t_center_of_pressure - positions, t_lift_force, dim=-1)

        # Calculate final force ad torque
        net_force = t_buoyancy_force + t_drag_force + t_lift_force + t_added_mass_force
        net_torque = t_torque_buoyancy + t_torque_drag + t_torque_lift + t_drag_torque + t_added_mass_torque

        # Safety Clamp (Prevent Explosions)
        MAX_ACCEL = 500.0 
        max_force = self._mass * MAX_ACCEL
        force_mag = torch.norm(net_force, dim=-1, keepdim=True)
        scale = torch.clamp(max_force / (force_mag + 1e-6), max=1.0)
        net_force = net_force * scale
        net_torque = net_torque * scale
        
        # Apply to Simulation
        self._rigid_prim_view.apply_forces_and_torques_at_pos(
            forces=net_force,
            torques=net_torque,
            positions=positions,
            is_global=True
        )

        # Update stored velocities for the next frame
        self._last_linear_velocity = linear_velocity.clone()
        self._last_angular_velocity = angular_velocity.clone()

    def _reset(self):
        self._hydro_calculator = None
        self._rigid_prim_view = None
        self._last_linear_velocity = None
        self._last_angular_velocity = None
        self._physx_subscription = None

    def _get_exposed_variable(self, attr_name):
        full_attr_name = f"{EXPOSED_ATTR_NS}:{self.BEHAVIOR_NS}:{attr_name}"
        return get_exposed_variable(self.prim, full_attr_name)