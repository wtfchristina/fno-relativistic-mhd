import time
import numpy as np


class SpectralMHDSolver:
    def __init__(self, grid_size: int = 64, eta: float = 2e-3, nu: float = 2e-3):
        self.N = grid_size
        self.eta = eta
        self.nu = nu
        x = np.linspace(0, 2 * np.pi, self.N, endpoint=False)
        self.X, self.Y = np.meshgrid(x, x, indexing="ij")

        k = np.fft.fftfreq(self.N, d=1.0 / self.N)
        self.KX, self.KY = np.meshgrid(k, k, indexing="ij")
        self.K2 = self.KX**2 + self.KY**2
        self.K2_safe = np.where(self.K2 == 0, 1.0, self.K2)

        k_max = self.N // 3
        self.dealias_mask = (np.abs(self.KX) < k_max) & (np.abs(self.KY) < k_max)

    def initial_orszag_tang(self, b0: float = 1.0, v0: float = 1.0) -> dict:
        vx = -v0 * np.sin(self.Y)
        vy = v0 * np.sin(self.X)
        bx = -b0 * np.sin(self.Y)
        by = b0 * np.sin(2.0 * self.X)
        return {"vx": vx, "vy": vy, "bx": bx, "by": by}

    def project_divergence_free(self, fx: np.ndarray, fy: np.ndarray) -> tuple:
        fx_hat = np.fft.fft2(fx) * self.dealias_mask
        fy_hat = np.fft.fft2(fy) * self.dealias_mask
        k_dot_f = self.KX * fx_hat + self.KY * fy_hat

        fx_proj = fx_hat - (self.KX * k_dot_f) / self.K2_safe
        fy_proj = fy_hat - (self.KY * k_dot_f) / self.K2_safe
        return np.real(np.fft.ifft2(fx_proj)), np.real(np.fft.ifft2(fy_proj))

    def compute_div_b_l2(self, bx: np.ndarray, by: np.ndarray) -> float:
        bx_hat = np.fft.fft2(bx)
        by_hat = np.fft.fft2(by)
        div_hat = 1j * (self.KX * bx_hat + self.KY * by_hat)
        div_real = np.real(np.fft.ifft2(div_hat))
        return float(np.sqrt(np.mean(div_real**2)))

    def solve_trajectory(self, state: dict, steps: int = 120, dt: float = 0.0015) -> dict:
        t0 = time.perf_counter()
        vx, vy = state["vx"].copy(), state["vy"].copy()
        bx, by = state["bx"].copy(), state["by"].copy()

        damp_v = np.exp(-self.nu * self.K2 * dt) * self.dealias_mask
        damp_b = np.exp(-self.eta * self.K2 * dt) * self.dealias_mask

        for _ in range(steps):
            vx_hat_c = np.fft.fft2(vx) * self.dealias_mask
            vy_hat_c = np.fft.fft2(vy) * self.dealias_mask
            bx_hat_c = np.fft.fft2(bx) * self.dealias_mask
            by_hat_c = np.fft.fft2(by) * self.dealias_mask

            vx_x = np.real(np.fft.ifft2(1j * self.KX * vx_hat_c))
            vx_y = np.real(np.fft.ifft2(1j * self.KY * vx_hat_c))
            vy_x = np.real(np.fft.ifft2(1j * self.KX * vy_hat_c))
            vy_y = np.real(np.fft.ifft2(1j * self.KY * vy_hat_c))

            bx_x = np.real(np.fft.ifft2(1j * self.KX * bx_hat_c))
            bx_y = np.real(np.fft.ifft2(1j * self.KY * bx_hat_c))
            by_x = np.real(np.fft.ifft2(1j * self.KX * by_hat_c))
            by_y = np.real(np.fft.ifft2(1j * self.KY * by_hat_c))

            dvx = -(vx * vx_x + vy * vx_y) + (bx * bx_x + by * bx_y)
            dvy = -(vx * vy_x + vy * vy_y) + (bx * by_x + by * by_y)

            ez = vx * by - vy * bx
            ez_hat = np.fft.fft2(ez) * self.dealias_mask
            dbx = np.real(np.fft.ifft2(1j * self.KY * ez_hat))
            dby = np.real(np.fft.ifft2(-1j * self.KX * ez_hat))

            vx_hat = (np.fft.fft2(vx + dt * dvx)) * damp_v
            vy_hat = (np.fft.fft2(vy + dt * dvy)) * damp_v
            bx_hat = (np.fft.fft2(bx + dt * dbx)) * damp_b
            by_hat = (np.fft.fft2(by + dt * dby)) * damp_b

            vx, vy = np.real(np.fft.ifft2(vx_hat)), np.real(np.fft.ifft2(vy_hat))
            bx, by = np.real(np.fft.ifft2(bx_hat)), np.real(np.fft.ifft2(by_hat))
            bx, by = self.project_divergence_free(bx, by)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        jz = np.real(
            np.fft.ifft2(1j * self.KX * np.fft.fft2(by) - 1j * self.KY * np.fft.fft2(bx))
        )
        return {
            "bx": bx,
            "by": by,
            "vx": vx,
            "vy": vy,
            "jz": jz,
            "mag_energy": 0.5 * (bx**2 + by**2),
            "div_b_l2": self.compute_div_b_l2(bx, by),
            "elapsed_ms": elapsed_ms,
        }
