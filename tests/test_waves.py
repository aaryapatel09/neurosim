"""Tests for wave equation / acoustics module."""

import jax
import jax.numpy as jnp
import pytest

from neurosim.exceptions import ConfigurationError
from neurosim.waves.acoustic import AcousticSolver, GaussianPulse, PointSource

jax.config.update("jax_enable_x64", True)


class TestAcousticSolver:
    """Tests for the 2D wave equation solver."""

    def test_basic_simulation(self) -> None:
        """Solver should run without errors."""
        solver = AcousticSolver(size=(50, 50), speed=1.0)
        solver.add_source(PointSource(position=(25, 25), frequency=5.0))
        result = solver.simulate(n_steps=100, dt=0.1, save_every=20)

        assert result.u.shape[1] == 50
        assert result.u.shape[2] == 50
        assert result.t.shape[0] > 0

    def test_gaussian_pulse(self) -> None:
        """Gaussian pulse should propagate outward."""
        solver = AcousticSolver(size=(60, 60), speed=1.0)
        solver.add_pulse(GaussianPulse(center=(30, 30), width=3.0))
        result = solver.simulate(n_steps=100, dt=0.1, save_every=50)

        # Initial pulse should have energy at center
        # After propagation, energy should spread
        initial_center = float(jnp.abs(result.u[0, 30, 30]))
        final_center = float(jnp.abs(result.u[-1, 30, 30]))
        assert initial_center > final_center  # Pulse moved away from center

    def test_ricker_source(self) -> None:
        """Ricker wavelet source should inject energy."""
        solver = AcousticSolver(size=(40, 40), speed=1.0)
        solver.add_source(
            PointSource(position=(20, 20), frequency=3.0, wavelet="ricker")
        )
        result = solver.simulate(n_steps=200, dt=0.1, save_every=50)

        max_amp = float(jnp.max(jnp.abs(result.u[-1])))
        assert max_amp > 0.0

    def test_sinusoidal_source(self) -> None:
        """Sinusoidal source should inject energy."""
        solver = AcousticSolver(size=(40, 40), speed=1.0)
        solver.add_source(
            PointSource(position=(20, 20), frequency=3.0, wavelet="sinusoidal")
        )
        result = solver.simulate(n_steps=200, dt=0.1, save_every=50)
        max_amp = float(jnp.max(jnp.abs(result.u[-1])))
        assert max_amp > 0.0

    def test_heterogeneous_media(self) -> None:
        """Different wave speeds should affect propagation."""
        nx, ny = 60, 60
        # Left half: speed=1, right half: speed=2
        speed_field = jnp.ones((nx, ny))
        speed_field = speed_field.at[nx // 2 :, :].set(2.0)

        solver_uniform = AcousticSolver(size=(nx, ny), speed=1.0)
        solver_uniform.add_pulse(GaussianPulse(center=(15, 30), width=3.0))

        solver_hetero = AcousticSolver(size=(nx, ny), speed_field=speed_field)
        solver_hetero.add_pulse(GaussianPulse(center=(15, 30), width=3.0))

        result_uni = solver_uniform.simulate(n_steps=200, dt=0.1, save_every=100)
        result_het = solver_hetero.simulate(n_steps=200, dt=0.1, save_every=100)

        # Fields should differ due to speed contrast
        assert not jnp.allclose(result_uni.u[-1], result_het.u[-1])

    def test_periodic_boundary(self) -> None:
        """Periodic and absorbing boundaries should give different results."""
        solver_abs = AcousticSolver(size=(40, 40), speed=1.0, boundary="absorbing")
        solver_per = AcousticSolver(size=(40, 40), speed=1.0, boundary="periodic")

        pulse = GaussianPulse(center=(20, 20), width=3.0)
        solver_abs.add_pulse(pulse)
        solver_per.add_pulse(pulse)

        result_abs = solver_abs.simulate(n_steps=200, dt=0.1, save_every=100)
        result_per = solver_per.simulate(n_steps=200, dt=0.1, save_every=100)

        assert not jnp.allclose(result_abs.u[-1], result_per.u[-1])

    def test_reflecting_boundary(self) -> None:
        """Reflecting boundary should conserve more energy than absorbing."""
        solver_ref = AcousticSolver(size=(40, 40), speed=1.0, boundary="reflecting")
        solver_abs = AcousticSolver(size=(40, 40), speed=1.0, boundary="absorbing")

        pulse = GaussianPulse(center=(20, 20), width=3.0)
        solver_ref.add_pulse(pulse)
        solver_abs.add_pulse(pulse)

        result_ref = solver_ref.simulate(n_steps=300, dt=0.1, save_every=150)
        result_abs = solver_abs.simulate(n_steps=300, dt=0.1, save_every=150)

        energy_ref = float(jnp.sum(result_ref.u[-1] ** 2))
        energy_abs = float(jnp.sum(result_abs.u[-1] ** 2))
        assert energy_ref >= energy_abs

    def test_speed_field_stored(self) -> None:
        speed = jnp.ones((30, 30)) * 2.0
        solver = AcousticSolver(size=(30, 30), speed_field=speed)
        solver.add_pulse(GaussianPulse(center=(15, 15)))
        result = solver.simulate(n_steps=50, dt=0.05, save_every=25)
        assert result.speed_field is not None
        assert float(jnp.mean(result.speed_field)) == pytest.approx(2.0)

    def test_amplitude_property(self) -> None:
        solver = AcousticSolver(size=(30, 30), speed=1.0)
        solver.add_pulse(GaussianPulse(center=(15, 15)))
        result = solver.simulate(n_steps=50, dt=0.1, save_every=25)
        assert result.amplitude.shape[0] == result.n_snapshots

    def test_cfl_error(self) -> None:
        solver = AcousticSolver(size=(30, 30), speed=1.0)
        solver.add_pulse(GaussianPulse(center=(15, 15)))
        with pytest.raises(ConfigurationError):
            solver.simulate(n_steps=10, dt=10.0)

    def test_no_source_error(self) -> None:
        solver = AcousticSolver(size=(30, 30), speed=1.0)
        with pytest.raises(ConfigurationError):
            solver.simulate()

    def test_small_grid_error(self) -> None:
        with pytest.raises(ConfigurationError):
            AcousticSolver(size=(2, 2))

    def test_source_out_of_bounds(self) -> None:
        solver = AcousticSolver(size=(30, 30))
        with pytest.raises(ConfigurationError):
            solver.add_source(PointSource(position=(50, 15)))

    def test_speed_field_shape_mismatch(self) -> None:
        with pytest.raises(ConfigurationError):
            AcousticSolver(size=(30, 30), speed_field=jnp.ones((20, 20)))

    def test_multiple_sources(self) -> None:
        solver = AcousticSolver(size=(50, 50), speed=1.0)
        solver.add_source(PointSource(position=(15, 25), frequency=3.0))
        solver.add_source(PointSource(position=(35, 25), frequency=5.0))
        result = solver.simulate(n_steps=100, dt=0.1, save_every=50)
        assert result.n_snapshots > 0
