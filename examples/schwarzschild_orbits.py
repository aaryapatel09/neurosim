"""Geodesic orbits around a Schwarzschild black hole.

Demonstrates the relativity module by computing and plotting
circular, precessing, and photon orbits.
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt

from neurosim.relativity.geodesic import GeodesicSolver

# --- Circular orbit at r = 10M ---
solver = GeodesicSolver(M=1.0, metric="schwarzschild")
r0 = 10.0
L_circ = float(jnp.sqrt(1.0 * r0**2 / (r0 - 3.0)))

traj_circ = solver.integrate(r0=r0, L=L_circ, ur0=0.0, tau_span=(0, 500), dtau=0.1)

# --- Precessing elliptical orbit ---
traj_prec = solver.integrate(r0=12.0, L=4.2, ur0=0.0, tau_span=(0, 800), dtau=0.1)

# --- Photon orbit (null geodesic) ---
traj_photon = solver.integrate(
    r0=20.0,
    L=5.5,
    E=1.0,
    massless=True,
    tau_span=(0, 150),
    dtau=0.02,
)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, traj, title in zip(
    axes,
    [traj_circ, traj_prec, traj_photon],
    ["Circular orbit", "Precessing orbit", "Photon trajectory"],
):
    ax.plot(traj.x, traj.y, lw=0.5)
    ax.plot(0, 0, "ko", ms=8)
    circle = plt.Circle(
        (0, 0), solver.schwarzschild_radius, color="k", fill=False, ls="--"
    )
    ax.add_patch(circle)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("x / M")
    ax.set_ylabel("y / M")

plt.tight_layout()
plt.savefig("schwarzschild_orbits.png", dpi=150)
plt.show()
print(f"ISCO radius: {solver.isco:.2f} M")
