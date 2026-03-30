"""Tests for general relativity module."""

import jax
import jax.numpy as jnp
import pytest

from neurosim.exceptions import ConfigurationError, PhysicsError
from neurosim.relativity.geodesic import GeodesicSolver
from neurosim.relativity.lensing import GravitationalLens
from neurosim.relativity.lorentz import (
    lorentz_boost,
    lorentz_factor,
    proper_time,
    velocity_addition,
)

jax.config.update("jax_enable_x64", True)


class TestGeodesicSolver:
    """Tests for geodesic integration."""

    def test_circular_orbit(self) -> None:
        """A particle at the right energy should stay near constant r."""
        solver = GeodesicSolver(M=1.0, metric="schwarzschild")
        r0 = 10.0
        # Circular orbit: L^2 = M*r^2/(r-3M), E^2 = (1-2M/r)^2/(1-3M/r)
        L = float(jnp.sqrt(1.0 * r0**2 / (r0 - 3.0)))
        E = float(jnp.sqrt((1.0 - 2.0 / r0) ** 2 / (1.0 - 3.0 / r0)))
        traj = solver.integrate(
            r0=r0,
            L=L,
            E=E,
            ur0=0.0,
            tau_span=(0, 100),
            dtau=0.01,
        )

        # r should stay approximately constant
        r = traj.r
        assert float(jnp.std(r)) / r0 < 0.1, "Orbit radius drifted too much"

    def test_trajectory_properties(self) -> None:
        """GeodesicTrajectory should have correct properties."""
        solver = GeodesicSolver(M=1.0)
        traj = solver.integrate(r0=10.0, L=4.0, tau_span=(0, 100), dtau=0.5)

        assert traj.n_steps > 0
        assert traj.r.shape[0] == traj.n_steps
        assert traj.phi.shape[0] == traj.n_steps
        assert traj.x.shape[0] == traj.n_steps
        assert traj.y.shape[0] == traj.n_steps
        assert traj.metric_name == "schwarzschild"

    def test_photon_orbit(self) -> None:
        """Null geodesic should run without errors."""
        solver = GeodesicSolver(M=1.0)
        traj = solver.integrate(
            r0=10.0,
            L=5.5,
            E=1.0,
            massless=True,
            tau_span=(0, 100),
            dtau=0.05,
        )
        assert traj.n_steps > 0

    def test_isco_schwarzschild(self) -> None:
        """ISCO for Schwarzschild should be 6M."""
        solver = GeodesicSolver(M=1.0)
        assert solver.isco == pytest.approx(6.0, rel=1e-10)

    def test_schwarzschild_radius(self) -> None:
        solver = GeodesicSolver(M=2.0)
        assert solver.schwarzschild_radius == pytest.approx(4.0)

    def test_kerr_metric(self) -> None:
        """Kerr geodesic should run without errors."""
        solver = GeodesicSolver(M=1.0, a=0.5, metric="kerr")
        traj = solver.integrate(
            r0=10.0,
            L=4.0,
            tau_span=(0, 100),
            dtau=0.5,
        )
        assert traj.metric_name == "kerr"
        assert traj.n_steps > 0

    def test_inside_horizon_error(self) -> None:
        solver = GeodesicSolver(M=1.0)
        with pytest.raises(PhysicsError):
            solver.integrate(r0=1.5, L=4.0)

    def test_negative_radius_error(self) -> None:
        solver = GeodesicSolver(M=1.0)
        with pytest.raises((ConfigurationError, PhysicsError)):
            solver.integrate(r0=-1.0, L=4.0)

    def test_schwarzschild_spin_error(self) -> None:
        with pytest.raises(ConfigurationError):
            GeodesicSolver(M=1.0, a=0.5, metric="schwarzschild")

    def test_extremal_kerr_error(self) -> None:
        with pytest.raises(PhysicsError):
            GeodesicSolver(M=1.0, a=1.0, metric="kerr")

    def test_save_every(self) -> None:
        solver = GeodesicSolver(M=1.0)
        traj = solver.integrate(
            r0=10.0, L=4.0, tau_span=(0, 100), dtau=0.5, save_every=5
        )
        assert traj.n_steps == 40  # 200 steps / 5


class TestGravitationalLens:
    """Tests for gravitational lensing."""

    def test_deflection_angle(self) -> None:
        """Weak-field deflection should be ~ 4M/b."""
        lens = GravitationalLens(M=1.0)
        b = jnp.array([100.0])
        delta = lens.deflection_angle(b)
        assert float(delta[0]) == pytest.approx(0.04, rel=0.05)

    def test_photon_sphere(self) -> None:
        lens = GravitationalLens(M=1.0)
        assert float(lens.photon_sphere) == pytest.approx(3.0)

    def test_shadow_radius(self) -> None:
        lens = GravitationalLens(M=1.0)
        assert float(lens.shadow_radius) == pytest.approx(3.0 * jnp.sqrt(3.0), rel=1e-6)

    def test_einstein_radius(self) -> None:
        lens = GravitationalLens(M=1.0)
        theta_e = lens.einstein_radius(d_lens=50.0, d_source=100.0)
        assert theta_e > 0

    def test_einstein_radius_ordering_error(self) -> None:
        lens = GravitationalLens(M=1.0)
        with pytest.raises(ConfigurationError):
            lens.einstein_radius(d_lens=100.0, d_source=50.0)

    def test_ray_trace_image(self) -> None:
        lens = GravitationalLens(M=1.0)
        image = lens.ray_trace_image(observer_distance=100.0, image_size=50, fov=0.1)
        assert image.shape == (50, 50)

    def test_shadow_in_image(self) -> None:
        """Center of image should show shadow (infinite deflection)."""
        lens = GravitationalLens(M=1.0)
        image = lens.ray_trace_image(observer_distance=50.0, image_size=50, fov=0.2)
        center = image[25, 25]
        assert jnp.isinf(center)

    def test_negative_mass_error(self) -> None:
        with pytest.raises(ConfigurationError):
            GravitationalLens(M=-1.0)


class TestLorentz:
    """Tests for special relativistic transformations."""

    def test_lorentz_factor_zero(self) -> None:
        """gamma(v=0) = 1."""
        assert float(lorentz_factor(0.0)) == pytest.approx(1.0)

    def test_lorentz_factor_known(self) -> None:
        """gamma(v=0.6c) = 1.25."""
        assert float(lorentz_factor(0.6)) == pytest.approx(1.25)

    def test_superluminal_error(self) -> None:
        with pytest.raises(PhysicsError):
            lorentz_factor(1.0)

    def test_boost_preserves_interval(self) -> None:
        """Lorentz boost should preserve the spacetime interval."""
        v = 0.5
        four_vec = jnp.array([10.0, 3.0, 0.0, 0.0])
        boosted = lorentz_boost(four_vec, v)

        interval_orig = four_vec[0] ** 2 - jnp.sum(four_vec[1:] ** 2)
        interval_boost = boosted[0] ** 2 - jnp.sum(boosted[1:] ** 2)
        assert float(interval_orig) == pytest.approx(float(interval_boost), rel=1e-10)

    def test_boost_3d(self) -> None:
        """General 3D boost should work."""
        four_vec = jnp.array([10.0, 1.0, 2.0, 3.0])
        v = jnp.array([0.3, 0.0, 0.0])
        boosted = lorentz_boost(four_vec, v)
        assert boosted.shape == (4,)

    def test_velocity_addition(self) -> None:
        """Relativistic addition of v1=v2=0.5c should give 0.8c."""
        v = velocity_addition(0.5, 0.5)
        assert v == pytest.approx(0.8)

    def test_velocity_addition_light(self) -> None:
        """Adding c to anything should give c."""
        v = velocity_addition(0.999, 0.999)
        assert v < 1.0

    def test_proper_time(self) -> None:
        """Proper time should be dilated."""
        dt = jnp.array([1.0])
        v = jnp.array([0.6])
        dtau = proper_time(v, dt)
        assert float(dtau[0]) == pytest.approx(0.8)
