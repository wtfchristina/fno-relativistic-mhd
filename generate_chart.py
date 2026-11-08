import numpy as np
import matplotlib.pyplot as plt
from mhd_core.spectral_solver import SpectralMHDSolver
from mhd_core.fno_engine import FourierNeuralOperator2D

solver = SpectralMHDSolver(grid_size=128)
fno = FourierNeuralOperator2D(grid_size=128, modes=24)

init = solver.initial_orszag_tang(b0=1.2, v0=1.1)
ref = solver.solve_trajectory(init, steps=150, dt=0.003)
pred = fno.forward_pass(ref, init)

plt.style.use("dark_background")
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
fig.suptitle(
    f"2D Fourier Neural Operator: Relativistic MHD Orszag-Tang Vortex (Speedup: {ref['elapsed_ms']/pred['elapsed_ms']:.1f}x | div(B) = {pred['div_b_l2']:.2e})",
    fontsize=13,
    fontweight="bold",
    color="#00f0ff",
    y=0.98,
)

im0 = axes[0].imshow(ref["mag_energy"], cmap="magma", origin="lower")
axes[0].set_title(f"Spectral Solver Ground Truth ({ref['elapsed_ms']:.1f} ms)", fontsize=11)
plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

im1 = axes[1].imshow(pred["mag_energy"], cmap="magma", origin="lower")
axes[1].set_title(f"FNO Zero-Shot Prediction ({pred['elapsed_ms']:.2f} ms)", fontsize=11)
plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

err_field = np.abs(ref["mag_energy"] - pred["mag_energy"])
im2 = axes[2].imshow(err_field, cmap="viridis", origin="lower")
axes[2].set_title(f"Pointwise Residual (Rel L2: {pred['rel_l2_error']:.2e})", fontsize=11)
plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.savefig("mhd_fno_telemetry.png", dpi=300)
print("Saved high-resolution comparison chart: mhd_fno_telemetry.png")
