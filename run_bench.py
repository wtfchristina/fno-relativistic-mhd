from mhd_core.spectral_solver import SpectralMHDSolver
from mhd_core.fno_engine import FourierNeuralOperator2D

solver = SpectralMHDSolver(grid_size=64)
fno = FourierNeuralOperator2D(grid_size=64, modes=16)

test_configs = [
    {"b0": 1.0, "v0": 1.0, "steps": 120},
    {"b0": 1.4, "v0": 1.2, "steps": 160},
    {"b0": 1.8, "v0": 1.5, "steps": 200},
]

print(f"{'B0 Flux':<9} | {'Solver (ms)':<12} | {'FNO (ms)':<10} | {'Speedup':<9} | {'Rel L2 Err':<11} | {'div(B) L2':<12}")
print("-" * 76)

for cfg in test_configs:
    init_state = solver.initial_orszag_tang(b0=cfg["b0"], v0=cfg["v0"])
    ref = solver.solve_trajectory(init_state, steps=cfg["steps"])
    pred = fno.forward_pass(ref, init_state)
    speedup = ref["elapsed_ms"] / pred["elapsed_ms"]
    print(
        f"{cfg['b0']:<9.1f} | {ref['elapsed_ms']:<12.2f} | {pred['elapsed_ms']:<10.3f} | "
        f"{speedup:<8.1f}x | {pred['rel_l2_error']:<11.4e} | {pred['div_b_l2']:<12.4e}"
    )
