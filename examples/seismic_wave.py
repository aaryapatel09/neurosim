"""Seismic wave propagation in heterogeneous media.

Demonstrates the acoustic wave solver with a layered velocity model
and a Ricker-wavelet point source.
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt

from neurosim.waves.acoustic import AcousticSolver, GaussianPulse, PointSource

nx, ny = 200, 200

# Build a layered velocity model (top: slow, bottom: fast)
speed = jnp.ones((nx, ny))
speed = speed.at[80:, :].set(1.5)
speed = speed.at[140:, :].set(2.0)

# --- Ricker source simulation ---
solver = AcousticSolver(size=(nx, ny), speed_field=speed, boundary="absorbing")
solver.add_source(PointSource(position=(50, 100), frequency=5.0, wavelet="ricker"))
result = solver.simulate(n_steps=600, dt=0.1, save_every=30)

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
n_snap = result.n_snapshots
indices = [0, n_snap // 3, 2 * n_snap // 3, n_snap - 1]

for ax, idx in zip(axes, indices):
    im = ax.imshow(result.u[idx], cmap="seismic", vmin=-0.02, vmax=0.02, origin="lower")
    ax.set_title(f"t = {float(result.t[idx]):.1f}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

plt.suptitle("Seismic wave in layered medium")
plt.tight_layout()
plt.savefig("seismic_wave.png", dpi=150)
plt.show()

# --- Gaussian pulse with periodic boundaries ---
solver2 = AcousticSolver(size=(100, 100), speed=1.0, boundary="periodic")
solver2.add_pulse(GaussianPulse(center=(50, 50), width=4.0))
result2 = solver2.simulate(n_steps=300, dt=0.1, save_every=50)
print(f"Periodic simulation: {result2.n_snapshots} snapshots saved")
