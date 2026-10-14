import time
import carb
import omni.physx
from omni.kit.scripting import BehaviorScript


class BenchmarkRtf(BehaviorScript):
    """
    A standalone script to measure Real-Time Factor (RTF) globally.
    """
    def on_init(self):
        self._physx_subscription = None
        self._running = False
        print("[RTF Benchmark] Initialized. Ready to run.")

    def on_destroy(self):
        self._physx_subscription = None

    def on_play(self):
        # Reset counters
        self._start_wall_time = time.time()
        self._total_sim_time = 0.0
        self._frame_count = 0
        self._running = True
        
        # Subscribe to physics steps (accurate simulation time)
        physx_interface = omni.physx.get_physx_interface()
        self._physx_subscription = physx_interface.subscribe_physics_step_events(self._on_physics_step)
        
        carb.log_info("[RTF Benchmark] Benchmarking started...")

    def on_stop(self):
        self._report_final_stats()
        self._running = False
        self._physx_subscription = None

    def _on_physics_step(self, delta_time: float):
        if not self._running:
            return
            
        # Accumulate exact simulation time stepped by PhysX
        self._total_sim_time += delta_time
        self._frame_count += 1
        
        if self._frame_count % 600 == 0:
            self._print_live_stats()

    def _report_final_stats(self):
        if not hasattr(self, '_start_wall_time'):
            return

        end_wall_time = time.time()
        total_wall_time = end_wall_time - self._start_wall_time
        
        # Avoid division by zero
        if total_wall_time < 0.001:
            return

        avg_rtf = self._total_sim_time / total_wall_time
        fps = self._frame_count / total_wall_time

        print(f"\n========================================")
        print(f"BENCHMARK RESULTS")
        print(f"----------------------------------------")
        print(f"Total Wall Time:  {total_wall_time:.4f} s")
        print(f"Total Sim Time:   {self._total_sim_time:.4f} s")
        print(f"Physics Steps:    {self._frame_count}")
        print(f"----------------------------------------")
        print(f"AVERAGE RTF:      {avg_rtf:.4f} x")
        print(f"AVERAGE FPS:      {fps:.2f}")
        print(f"========================================\n")

    def _print_live_stats(self):
        current_wall = time.time() - self._start_wall_time
        rtf = self._total_sim_time / current_wall
        print(f"[RTF Live] Sim: {self._total_sim_time:.2f}s | RTF: {rtf:.3f}")