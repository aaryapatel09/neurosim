"""Geodesic integration in curved spacetimes.

Solves the geodesic equation for massive and massless particles in
Schwarzschild and Kerr spacetimes using Hamiltonian formulation with
symplectic integration.

The geodesic equation in Hamiltonian form:

    dx^mu/dtau = dH/dp_mu
    dp_mu/dtau = -dH/dx^mu

where H = (1/2) g^{mu nu} p_mu p_nu is the super-Hamiltonian.
For geodesics, H = -1/2 (massive) or H = 0 (massless/photons).

Coordinates: (t, r, theta, phi) in Boyer-Lindquist form.

References:
    - Misner, Thorne, Wheeler. "Gravitation" (1973), Ch. 25, 33
    - Chandrasekhar. "The Mathematical Theory of Black Holes" (1983)
    - Levin & Perez-Giz. "A Periodic Table for Black Hole Orbits" (2008)
"""

from __future__ import annotations

import logging
from typing import Literal

import jax
import jax.numpy as jnp
from jax import Array

from neurosim.config import RelativityConfig
from neurosim.exceptions import ConfigurationError, PhysicsError
from neurosim.state import GeodesicTrajectory

logger = logging.getLogger(__name__)


class GeodesicSolver:
    """Geodesic integrator for Schwarzschild and Kerr spacetimes.

    Integrates the geodesic equations using the Hamiltonian formulation
    with a symplectic (leapfrog) integrator to preserve the constraint
    g^{mu nu} p_mu p_nu = const.

    Example:
        >>> solver = GeodesicSolver(M=1.0, metric="schwarzschild")
        >>> # Circular orbit at r=10M
        >>> traj = solver.integrate(
        ...     r0=10.0, phi0=0.0, ur0=0.0, L=3.5,
        ...     tau_span=(0, 500), dtau=0.1,
        ... )

    Args:
        M: Central mass in geometric units (G=c=1).
        a: Kerr spin parameter (0 <= a < M). Only used for metric="kerr".
        metric: Spacetime metric.
    """

    def __init__(
        self,
        M: float = 1.0,
        a: float = 0.0,
        metric: Literal["schwarzschild", "kerr"] = "schwarzschild",
    ) -> None:
        self._config = RelativityConfig(M=M, a=a, metric=metric)
        self._M = M
        self._a = a
        self._metric = metric

        if metric == "schwarzschild" and a != 0.0:
            raise ConfigurationError(
                "Schwarzschild metric requires a=0. Use metric='kerr' for spinning black holes."
            )
        if metric == "kerr" and a >= M:
            raise PhysicsError(
                f"Kerr spin parameter a={a} must be < M={M} (extremal limit)."
            )

    @property
    def schwarzschild_radius(self) -> float:
        """Schwarzschild radius r_s = 2M."""
        return 2.0 * self._M

    @property
    def isco(self) -> float:
        """Innermost stable circular orbit radius.

        For Schwarzschild: r_isco = 6M.
        For Kerr (prograde): r_isco depends on spin.
        """
        if self._metric == "schwarzschild":
            return 6.0 * self._M
        # Kerr ISCO (prograde)
        M, a = self._M, self._a
        z1 = 1.0 + (1.0 - a**2 / M**2) ** (1.0 / 3.0) * (
            (1.0 + a / M) ** (1.0 / 3.0) + (1.0 - a / M) ** (1.0 / 3.0)
        )
        z2 = jnp.sqrt(3.0 * a**2 / M**2 + z1**2)
        return float(M * (3.0 + z2 - jnp.sqrt((3.0 - z1) * (3.0 + z1 + 2.0 * z2))))

    def integrate(
        self,
        r0: float = 10.0,
        theta0: float = jnp.pi / 2,
        phi0: float = 0.0,
        ur0: float = 0.0,
        L: float = 4.0,
        E: float | None = None,
        massless: bool = False,
        tau_span: tuple[float, float] = (0.0, 500.0),
        dtau: float = 0.1,
        save_every: int = 1,
    ) -> GeodesicTrajectory:
        """Integrate geodesic equations.

        Args:
            r0: Initial radial coordinate.
            theta0: Initial polar angle (default: equatorial plane).
            phi0: Initial azimuthal angle.
            ur0: Initial radial velocity dr/dtau.
            L: Angular momentum (conserved quantity).
            E: Energy per unit mass. If None, computed for circular orbit.
            massless: If True, trace null geodesic (photon).
            tau_span: (tau_start, tau_end) proper time range.
            dtau: Proper time step.
            save_every: Save every N steps.

        Returns:
            GeodesicTrajectory with coordinates and momenta.
        """
        M = self._M
        rs = 2.0 * M

        if r0 <= rs and not massless:
            raise PhysicsError(
                f"Initial radius r0={r0} is inside the event horizon r_s={rs}."
            )

        if r0 <= 0:
            raise ConfigurationError(f"Initial radius must be positive, got r0={r0}")

        # Compute energy for circular orbit if not specified
        if E is None:
            if massless:
                E = 1.0
            else:
                E = float(jnp.sqrt((1.0 - rs / r0) * (1.0 + L**2 / r0**2)))

        tau_start, tau_end = tau_span
        n_steps = int((tau_end - tau_start) / dtau)

        logger.info(
            "Integrating %s geodesic: r0=%.2f, L=%.2f, E=%.4f, n_steps=%d",
            self._metric,
            r0,
            L,
            E,
            n_steps,
        )

        if self._metric == "schwarzschild":
            return self._integrate_schwarzschild(
                r0,
                theta0,
                phi0,
                ur0,
                float(L),
                float(E),
                massless,
                tau_start,
                dtau,
                n_steps,
                save_every,
            )
        return self._integrate_kerr(
            r0,
            theta0,
            phi0,
            ur0,
            float(L),
            float(E),
            massless,
            tau_start,
            dtau,
            n_steps,
            save_every,
        )

    def _integrate_schwarzschild(
        self,
        r0: float,
        theta0: float,
        phi0: float,
        ur0: float,
        L: float,
        E: float,
        massless: bool,
        tau_start: float,
        dtau: float,
        n_steps: int,
        save_every: int,
    ) -> GeodesicTrajectory:
        """Integrate geodesics in Schwarzschild spacetime."""
        M = self._M
        rs = 2.0 * M
        # Initial conditions in (t, r, theta, phi)
        # p_t = -E, p_phi = L (conserved)
        # p_r from the mass-shell constraint
        r0_val = r0
        f0 = 1.0 - rs / r0_val  # metric function

        # p_r^2 = E^2/f - f*(mu2 + L^2/r^2)  =>  p_r = ... (covariant)
        # Using contravariant: dr/dtau = p^r = f * p_r
        # p_r (covariant) = (1/f) * ur0
        pr0 = ur0 / f0

        coords0 = jnp.array([0.0, r0_val, theta0, phi0])
        momenta0 = jnp.array([-E, pr0, 0.0, L])

        def step(
            carry: tuple[Array, Array, int],
            _: None,
        ) -> tuple[tuple[Array, Array, int], tuple[Array, Array]]:
            coords, mom, idx = carry
            r = coords[1]
            f = 1.0 - rs / r

            # Half-step momenta update (dp/dtau = -dH/dx)
            # dH/dr = (rs/(2*r^2*f)) * mom[0]^2 + (rs/(2*r^2)) * mom[1]^2
            #         - mom[2]^2/r^3 - mom[3]^2/(r^3 * sin^2(th))
            dH_dr = (
                (rs / (2.0 * r**2)) * mom[0] ** 2 / f**2
                + (rs / (2.0 * r**2)) * mom[1] ** 2
                - mom[2] ** 2 / r**3
                - mom[3] ** 2 / (r**3 * jnp.sin(coords[2]) ** 2)
            )

            dH_dth = mom[3] ** 2 * jnp.cos(coords[2]) / (r**2 * jnp.sin(coords[2]) ** 3)

            mom_half = mom.at[1].set(mom[1] - 0.5 * dtau * dH_dr)
            mom_half = mom_half.at[2].set(mom_half[2] - 0.5 * dtau * dH_dth)

            # Full-step coords update (dx/dtau = dH/dp)
            r_new = coords[1]
            f_new = 1.0 - rs / r_new

            dt_dtau = -mom_half[0] / f_new
            dr_dtau = f_new * mom_half[1]
            dth_dtau = mom_half[2] / r_new**2
            dphi_dtau = mom_half[3] / (r_new**2 * jnp.sin(coords[2]) ** 2)

            coords_new = coords + dtau * jnp.array(
                [dt_dtau, dr_dtau, dth_dtau, dphi_dtau]
            )

            # Second half-step momenta
            r2 = coords_new[1]
            f2 = 1.0 - rs / r2

            dH_dr2 = (
                (rs / (2.0 * r2**2)) * mom_half[0] ** 2 / f2**2
                + (rs / (2.0 * r2**2)) * mom_half[1] ** 2
                - mom_half[2] ** 2 / r2**3
                - mom_half[3] ** 2 / (r2**3 * jnp.sin(coords_new[2]) ** 2)
            )

            dH_dth2 = (
                mom_half[3] ** 2
                * jnp.cos(coords_new[2])
                / (r2**2 * jnp.sin(coords_new[2]) ** 3)
            )

            mom_new = mom_half.at[1].set(mom_half[1] - 0.5 * dtau * dH_dr2)
            mom_new = mom_new.at[2].set(mom_new[2] - 0.5 * dtau * dH_dth2)

            return (coords_new, mom_new, idx + 1), (coords_new, mom_new)

        init = (coords0, momenta0, 0)
        _, (coords_all, momenta_all) = jax.lax.scan(step, init, None, length=n_steps)

        # Subsample
        if save_every > 1:
            indices = jnp.arange(0, n_steps, save_every)
            coords_all = coords_all[indices]
            momenta_all = momenta_all[indices]

        tau = jnp.arange(coords_all.shape[0], dtype=jnp.float64) * dtau * save_every

        return GeodesicTrajectory(
            tau=tau,
            coords=coords_all,
            momenta=momenta_all,
            metric_name="schwarzschild",
        )

    def _integrate_kerr(
        self,
        r0: float,
        theta0: float,
        phi0: float,
        ur0: float,
        L: float,
        E: float,
        massless: bool,
        tau_start: float,
        dtau: float,
        n_steps: int,
        save_every: int,
    ) -> GeodesicTrajectory:
        """Integrate geodesics in Kerr spacetime (Boyer-Lindquist)."""
        M = self._M
        a = self._a

        coords0 = jnp.array([0.0, r0, theta0, phi0])
        pr0 = ur0  # approximate
        momenta0 = jnp.array([-E, pr0, 0.0, L])

        def kerr_metric_inv(
            r: Array | float, theta: Array | float
        ) -> tuple[Array, Array, Array, Array, Array]:
            """Compute inverse metric components for Kerr."""
            sigma = r**2 + a**2 * jnp.cos(theta) ** 2
            delta = r**2 - 2.0 * M * r + a**2
            A = (r**2 + a**2) ** 2 - a**2 * delta * jnp.sin(theta) ** 2

            gtt_inv = -A / (sigma * delta)
            grr_inv = delta / sigma
            gthth_inv = 1.0 / sigma
            gphph_inv = (delta - a**2 * jnp.sin(theta) ** 2) / (
                sigma * delta * jnp.sin(theta) ** 2
            )
            gtph_inv = -2.0 * M * a * r / (sigma * delta)

            return gtt_inv, grr_inv, gthth_inv, gphph_inv, gtph_inv

        def step(
            carry: tuple[Array, Array, int],
            _: None,
        ) -> tuple[tuple[Array, Array, int], tuple[Array, Array]]:
            coords, mom, idx = carry
            r = coords[1]
            theta = coords[2]

            gtt_inv, grr_inv, gthth_inv, gphph_inv, gtph_inv = kerr_metric_inv(r, theta)

            # dx/dtau = g^{mu nu} p_nu
            dt_dtau = gtt_inv * mom[0] + gtph_inv * mom[3]
            dr_dtau = grr_inv * mom[1]
            dth_dtau = gthth_inv * mom[2]
            dphi_dtau = gphph_inv * mom[3] + gtph_inv * mom[0]

            # Numerical derivatives for dp/dtau (finite difference on metric)
            eps = 1e-6
            # dr
            gtt_p, grr_p, gthth_p, gphph_p, gtph_p = kerr_metric_inv(r + eps, theta)
            gtt_m, grr_m, gthth_m, gphph_m, gtph_m = kerr_metric_inv(r - eps, theta)

            dH_dr = (
                0.5 * (gtt_p - gtt_m) / (2.0 * eps) * mom[0] ** 2
                + 0.5 * (grr_p - grr_m) / (2.0 * eps) * mom[1] ** 2
                + 0.5 * (gthth_p - gthth_m) / (2.0 * eps) * mom[2] ** 2
                + 0.5 * (gphph_p - gphph_m) / (2.0 * eps) * mom[3] ** 2
                + (gtph_p - gtph_m) / (2.0 * eps) * mom[0] * mom[3]
            )

            # dtheta
            gtt_p2, grr_p2, gthth_p2, gphph_p2, gtph_p2 = kerr_metric_inv(
                r, theta + eps
            )
            gtt_m2, grr_m2, gthth_m2, gphph_m2, gtph_m2 = kerr_metric_inv(
                r, theta - eps
            )

            dH_dth = (
                0.5 * (gtt_p2 - gtt_m2) / (2.0 * eps) * mom[0] ** 2
                + 0.5 * (grr_p2 - grr_m2) / (2.0 * eps) * mom[1] ** 2
                + 0.5 * (gthth_p2 - gthth_m2) / (2.0 * eps) * mom[2] ** 2
                + 0.5 * (gphph_p2 - gphph_m2) / (2.0 * eps) * mom[3] ** 2
                + (gtph_p2 - gtph_m2) / (2.0 * eps) * mom[0] * mom[3]
            )

            # Leapfrog: half-step mom, full-step coords, half-step mom
            mom_half = mom.at[1].set(mom[1] - 0.5 * dtau * dH_dr)
            mom_half = mom_half.at[2].set(mom_half[2] - 0.5 * dtau * dH_dth)

            coords_new = coords + dtau * jnp.array(
                [dt_dtau, dr_dtau, dth_dtau, dphi_dtau]
            )

            mom_new = mom_half.at[1].set(mom_half[1] - 0.5 * dtau * dH_dr)
            mom_new = mom_new.at[2].set(mom_new[2] - 0.5 * dtau * dH_dth)

            return (coords_new, mom_new, idx + 1), (coords_new, mom_new)

        init = (coords0, momenta0, 0)
        _, (coords_all, momenta_all) = jax.lax.scan(step, init, None, length=n_steps)

        if save_every > 1:
            indices = jnp.arange(0, n_steps, save_every)
            coords_all = coords_all[indices]
            momenta_all = momenta_all[indices]

        tau = jnp.arange(coords_all.shape[0], dtype=jnp.float64) * dtau * save_every

        return GeodesicTrajectory(
            tau=tau,
            coords=coords_all,
            momenta=momenta_all,
            metric_name="kerr",
        )
