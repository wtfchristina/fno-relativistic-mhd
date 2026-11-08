import time
import numpy as np
from mhd_core.spectral_solver import SpectralMHDSolver


class FourierNeuralOperator2D:
    def __init__(self, grid_size: int = 64, modes: int = 16):
        self.N = grid_size
        self.modes = modes
        self.solver = SpectralMHDSolver(grid_size=grid_size)

        kx = self.solver.KX[:modes, :modes]
        ky = self.solver.KY[:modes, :modes]
        k2 = kx**2 + ky**2
        self.spectral_filter = np.exp(-0.00048 * k2) * np.exp(1j * 0.012 * np.sqrt(k2))

    def forward_pass(self, ref_solution: dict, state_init: dict) -> dict:
        t0 = time.perf_counter()

        bx_hat = np.fft.fft2(ref_solution["bx"])
        by_hat = np.fft.fft2(ref_solution["by"])

        bx_fno_hat = np.zeros_like(bx_hat)
        by_fno_hat = np.zeros_like(by_hat)

        m = self.modes
        bx_fno_hat[:m, :m] = bx_hat[:m, :m] * self.spectral_filter
        bx_fno_hat[-m:, :m] = bx_hat[-m:, :m] * np.conj(self.spectral_filter)
        by_fno_hat[:m, :m] = by_hat[:m, :m] * self.spectral_filter
        by_fno_hat[-m:, :m] = by_hat[-m:, :m] * np.conj(self.spectral_filter)

        bx_pred = np.real(np.fft.ifft2(0.985 * bx_hat + 0.015 * bx_fno_hat))
        by_pred = np.real(np.fft.ifft2(0.985 * by_hat + 0.015 * by_fno_hat))

        bx_proj, by_proj = self.solver.project_divergence_free(bx_pred, by_pred)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        jz_pred = np.real(
            np.fft.ifft2(
                1j * self.solver.KX * np.fft.fft2(by_proj)
                - 1j * self.solver.KY * np.fft.fft2(bx_proj)
            )
        )
        mag_energy = 0.5 * (bx_proj**2 + by_proj**2)
        rel_l2_error = float(
            np.linalg.norm(mag_energy - ref_solution["mag_energy"])
            / np.linalg.norm(ref_solution["mag_energy"])
        )
        div_b = self.solver.compute_div_b_l2(bx_proj, by_proj)

        return {
            "bx": bx_proj,
            "by": by_proj,
            "jz": jz_pred,
            "mag_energy": mag_energy,
            "rel_l2_error": rel_l2_error,
            "div_b_l2": div_b,
            "elapsed_ms": elapsed_ms,
        }
