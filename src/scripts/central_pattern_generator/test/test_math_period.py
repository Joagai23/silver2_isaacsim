"""
Diagnostic Script: Test 1 - CPG Mathematical Period & Zero-Crossing Verification
Evaluates limit cycle timing, zero-crossings, and tripod phase locking in isolation.
"""

import numpy as np
from silver2_isaac_constants import SILVER2_MOUNTS, SILVER2_LINKS
from numpy_cpg_controller import NumpyHexapodCPGController


def run_period_diagnostic(dt: float, total_period: float = 2.0, num_cycles: int = 10, label: str = ""):
    print("\n" + "=" * 80)
    print(f"RUNNING TEST: {label}")
    print(f"Parameters: dt = {dt:.6f} s ({1.0 / dt:.2f} Hz) | Commanded Period T = {total_period:.3f} s")
    print("=" * 80)

    # 1. Instantiate isolated controller
    cpg = NumpyHexapodCPGController(
        leg_mounts=SILVER2_MOUNTS,
        link_lengths=SILVER2_LINKS,
        dt=dt,
        total_period=total_period,
        gait="tripod"
    )

    total_time = num_cycles * total_period
    total_steps = int(np.ceil(total_time / dt))

    # Recording arrays
    timestamps = np.zeros(total_steps)
    y0_history = np.zeros(total_steps)
    y1_history = np.zeros(total_steps)
    r0_history = np.zeros(total_steps)

    # 2. Integration loop
    for step in range(total_steps):
        t = step * dt
        timestamps[step] = t
        
        # Step Hopf ODEs
        cpg.step_cpg()

        y0_history[step] = cpg.y[0]  # Leg 0 (Tripod 1)
        y1_history[step] = cpg.y[1]  # Leg 1 (Tripod 2)
        r0_history[step] = np.sqrt(cpg.x[0]**2 + cpg.y[0]**2)

    # 3. Detect zero-crossings (negative to positive transition: Swing -> Stance)
    # Using linear interpolation between discrete steps for sub-timestep precision:
    # t_cross = t[k-1] + (-y[k-1] / (y[k] - y[k-1])) * dt
    crossings_leg0 = []
    for k in range(1, total_steps):
        if y0_history[k - 1] < 0.0 and y0_history[k] >= 0.0:
            dy = y0_history[k] - y0_history[k - 1]
            frac = -y0_history[k - 1] / dy if dy != 0.0 else 0.0
            t_exact = timestamps[k - 1] + frac * dt
            crossings_leg0.append(t_exact)

    crossings_leg1 = []
    for k in range(1, total_steps):
        if y1_history[k - 1] < 0.0 and y1_history[k] >= 0.0:
            dy = y1_history[k] - y1_history[k - 1]
            frac = -y1_history[k - 1] / dy if dy != 0.0 else 0.0
            t_exact = timestamps[k - 1] + frac * dt
            crossings_leg1.append(t_exact)

    crossings_leg0 = np.array(crossings_leg0)
    crossings_leg1 = np.array(crossings_leg1)

    if len(crossings_leg0) < 2:
        print("[ERROR] Not enough zero-crossings detected to measure period.")
        return

    # Calculate period between successive cycles
    periods = np.diff(crossings_leg0)
    mean_period = np.mean(periods)
    std_period = np.std(periods)
    period_error_pct = ((mean_period - total_period) / total_period) * 100.0

    # 4. Phase-offset calculation between Leg 0 and Leg 1 (Tripod anti-phase)
    # Ideal difference should be exactly T / 2 (pi phase shift)
    phase_shifts = []
    for t0 in crossings_leg0:
        # Find closest subsequent crossing in leg 1
        t1_candidates = crossings_leg1[crossings_leg1 > t0]
        if len(t1_candidates) > 0:
            phase_shifts.append(t1_candidates[0] - t0)

    mean_phase_shift = np.mean(phase_shifts) if len(phase_shifts) > 0 else 0.0
    expected_phase_shift = total_period / 2.0
    phase_error_pct = ((mean_phase_shift - expected_phase_shift) / expected_phase_shift) * 100.0

    # 5. Output Telemetry Table
    print(f"{'Cycle #':<10} | {'Cycle Start (s)':<16} | {'Measured Period (s)':<22} | {'Error vs Target'}")
    print("-" * 80)
    for idx, (t_start, p_val) in enumerate(zip(crossings_leg0[:-1], periods)):
        err = p_val - total_period
        print(f"Cycle {idx + 1:<4} | {t_start:12.4f} s    | {p_val:14.4f} s          | {err:+.4f} s ({err / total_period * 100:+.2f}%)")

    print("-" * 80)
    print(f"Summary Metrics:")
    print(f"  • Commanded Period       : {total_period:.4f} s")
    print(f"  • Mean Measured Period   : {mean_period:.4f} s ± {std_period:.5f} s")
    print(f"  • Period Error           : {period_error_pct:+.2f}%")
    print(f"  • Limit Cycle Radius (R) : {np.mean(r0_history):.3f} (Theoretical sqrt(mu) = {np.sqrt(cpg.mu):.3f})")
    print(f"  • Tripod 1-2 Phase Shift : {mean_phase_shift:.4f} s (Target: {expected_phase_shift:.4f} s, Error: {phase_error_pct:+.2f}%)")

    if abs(period_error_pct) > 1.0:
        print("\n[RESULT: FAIL] The numerical oscillator period drifts from the commanded physical period.")
    else:
        print("\n[RESULT: PASS] The numerical oscillator matches the physical clock within < 1% error.")

def main():
    #Test Case A: Fixed 100 Hz timestep (the controller's default)
    run_period_diagnostic(
        dt=0.01,
        total_period=2.0,
        num_cycles=8,
        label="Case A: Controller Default dt = 0.01 s (100 Hz)"
    )

    # Test Case B: Decimated 420 Hz timestep (4 * 1/420 s ≈ 0.009524 s)
    run_period_diagnostic(
        dt=(1.0 / 420.0) * 4.0,
        total_period=2.0,
        num_cycles=8,
        label="Case B: Decimated Decoupled dt = 4/420 s (105 Hz)"
    )

if __name__ == "__main__":
    main()