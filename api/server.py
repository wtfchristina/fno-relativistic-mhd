from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import numpy as np
from mhd_core.spectral_solver import SpectralMHDSolver
from mhd_core.fno_engine import FourierNeuralOperator2D

app = FastAPI(
    title="3D FNO Relativistic MHD Plasma Engine",
    description="Interactive 3D WebGL & Fourier Neural Operator with Helmholtz-Hodge div(B)=0 projection.",
    version="2.0.0",
)

solver = SpectralMHDSolver(grid_size=64)
fno = FourierNeuralOperator2D(grid_size=64, modes=16)


@app.get("/api/v1/infer-mhd-3d")
def infer_mhd_3d(
    b0_flux: float = Query(1.2, gt=0.1, le=3.0),
    v0_velocity: float = Query(1.0, gt=0.1, le=3.0),
    steps: int = Query(120, ge=10, le=300),
):
    init_state = solver.initial_orszag_tang(b0=b0_flux, v0=v0_velocity)
    ref = solver.solve_trajectory(init_state, steps=steps)
    pred = fno.forward_pass(ref, init_state)

    b_hat = np.fft.fftshift(np.abs(np.fft.fft2(pred["mag_energy"])))
    center = solver.N // 2
    span = 12
    k_spec = b_hat[center - span : center + span, center - span : center + span]
    k_spec_norm = (np.log1p(k_spec) / np.max(np.log1p(k_spec) + 1e-8)).tolist()

    return {
        "grid_size": solver.N,
        "b0_flux": b0_flux,
        "v0_velocity": v0_velocity,
        "solver_latency_ms": round(ref["elapsed_ms"], 2),
        "fno_latency_ms": round(pred["elapsed_ms"], 3),
        "speedup_factor": round(ref["elapsed_ms"] / pred["elapsed_ms"], 1),
        "relative_l2_error": float(pred["rel_l2_error"]),
        "div_b_l2_norm": float(pred["div_b_l2"]),
        "peak_energy": round(float(pred["mag_energy"].max()), 4),
        "mag_energy": np.round(pred["mag_energy"], 4).tolist(),
        "jz_current": np.round(pred["jz"], 4).tolist(),
        "bx": np.round(pred["bx"], 3).tolist(),
        "by": np.round(pred["by"], 3).tolist(),
        "k_spectrum": k_spec_norm,
    }


@app.get("/", response_class=HTMLResponse)
def serve_3d_dashboard():
    with open("api/index.html", "r") as f:
        return f.read()
