"""2D acoustic wave equation solver.

Solves the scalar wave equation in 2D:

    d²u/dt² = c(x,y)² * (d²u/dx² + d²u/dy²)

where u is the displacement (pressure) field and c(x,y) is the
spatially varying wave speed. Uses a second-order finite difference
scheme in both space and time.

Supports:
    - Heterogeneous media (variable speed of sound)
    - Point sources and Gaussian pulse initial conditions
    - Absorbing (sponge layer), periodic, and reflecting boundaries
    - Source time functions (Ricker wavelet, sinusoidal)

References:
    - Aki & Richards. "Quantitative Seismology" (2002), Ch. 4
    - Virieux. "P-SV wave propagation in heterogeneous media" (1986)
    - Clayton & Engquist. "Absorbing boundary conditions for acoustic
      and elastic wave equations" (1977)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
from jax import Array

from neurosim.config import WaveConfig
from neurosim.exceptions import ConfigurationError
from neurosim.state import WaveResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PointSource:
    """Point source for wave excitation.

    Attributes:
        position: Source location (ix, iy) in grid indices.
        frequency: Source frequency in Hz (for Ricker/sinusoidal).
        amplitude: Peak amplitude.
        wavelet: Source time function type.
    """

    position: tuple[int, int]
    frequency: float = 10.0
    amplitude: float = 1.0
    wavelet: Literal["ricker", "sinusoidal"] = "ricker"


@dataclass(frozen=True)
class GaussianPulse:
    """Gaussian pulse initial condition.

    Attributes:
        center: Pulse center (ix, iy) in grid indices.
        width: Gaussian width in grid units.
        amplitude: Peak amplitude.
    """

    center: tuple[int, int]
    width: float = 5.0
    amplitude: float = 1.0


class AcousticSolver:
    """2D acoustic wave equation solver.

    Solves the wave equation on a uniform grid with optional
    heterogeneous wave speed, multiple sources, and absorbing
    boundaries.

    Example:
        >>> solver = AcousticSolver(size=(200, 200), speed=1.0)
        >>> solver.add_source(PointSource(position=(100, 100), frequency=5.0))
        >>> result = solver.simulate(n_steps=1000, dt=0.1, save_every=20)

    Args:
        size: Grid dimensions (nx, ny).
        speed: Default wave speed (uniform). Overridden by speed_field.
        dx: Grid spacing.
        boundary: Boundary condition type.
        speed_field: Optional 2D array of spatially varying wave speed,
            shape (nx, ny).
    """

    def __init__(
        self,
        size: tuple[int, int] = (200, 200),
        speed: float = 1.0,
        dx: float = 1.0,
        boundary: Literal["absorbing", "periodic", "reflecting"] = "absorbing",
        speed_field: Array | None = None,
    ) -> None:
        self._nx, self._ny = size
        self._dx = dx
        self._config = WaveConfig(speed=speed, boundary=boundary)  # type: ignore[call-arg]
        self._sources: list[PointSource] = []
        self._pulses: list[GaussianPulse] = []

        if speed_field is not None:
            if speed_field.shape != (self._nx, self._ny):
                raise ConfigurationError(
                    f"Speed field shape {speed_field.shape} doesn't match "
                    f"grid size ({self._nx}, {self._ny})"
                )
            self._speed_field = speed_field
        else:
            self._speed_field = jnp.full((self._nx, self._ny), speed)

        if min(size) < 4:
            raise ConfigurationError(f"Grid must be at least 4x4, got {size}")

    @property
    def size(self) -> tuple[int, int]:
        """Grid dimensions (nx, ny)."""
        return (self._nx, self._ny)

    def add_source(self, source: PointSource) -> None:
        """Add a point source."""
        ix, iy = source.position
        if not (0 <= ix < self._nx and 0 <= iy < self._ny):
            raise ConfigurationError(
                f"Source position {source.position} out of grid bounds "
                f"({self._nx}, {self._ny})"
            )
        self._sources.append(source)

    def add_pulse(self, pulse: GaussianPulse) -> None:
        """Add a Gaussian pulse initial condition."""
        ix, iy = pulse.center
        if not (0 <= ix < self._nx and 0 <= iy < self._ny):
            raise ConfigurationError(
                f"Pulse center {pulse.center} out of grid bounds "
                f"({self._nx}, {self._ny})"
            )
        self._pulses.append(pulse)

    def simulate(
        self,
        n_steps: int = 1000,
        dt: float = 0.1,
        save_every: int = 20,
    ) -> WaveResult:
        """Run the wave equation simulation.

        Args:
            n_steps: Number of time steps.
            dt: Time step size.
            save_every: Save snapshots every N steps.

        Returns:
            WaveResult with displacement field snapshots.
        """
        nx, ny = self._nx, self._ny
        dx = self._dx
        c = self._speed_field
        boundary = self._config.boundary

        # CFL stability: dt <= dx / (c_max * sqrt(2))
        c_max = float(jnp.max(c))
        cfl = c_max * dt / dx
        if cfl > 1.0 / jnp.sqrt(2.0):
            raise ConfigurationError(
                f"CFL number {cfl:.3f} exceeds 1/sqrt(2) ~ 0.707. "
                "Reduce dt or increase dx."
            )

        if not self._sources and not self._pulses:
            raise ConfigurationError(
                "At least one source or pulse must be added before simulation"
            )

        logger.info(
            "Starting wave simulation: grid=%dx%d, c_max=%.2f, dt=%.4f, n_steps=%d",
            nx,
            ny,
            c_max,
            dt,
            n_steps,
        )

        # Precompute coefficient
        c2dt2_dx2 = (c * dt / dx) ** 2

        # Initialize fields
        u_curr = jnp.zeros((nx, ny))
        u_prev = jnp.zeros((nx, ny))

        # Apply Gaussian pulse initial conditions
        grid_x, grid_y = jnp.meshgrid(
            jnp.arange(nx, dtype=jnp.float64),
            jnp.arange(ny, dtype=jnp.float64),
            indexing="ij",
        )
        for pulse in self._pulses:
            cx, cy = pulse.center
            r2 = (grid_x - cx) ** 2 + (grid_y - cy) ** 2
            u_curr = u_curr + pulse.amplitude * jnp.exp(-r2 / (2.0 * pulse.width**2))

        u_prev = u_curr.copy()

        # Build absorbing boundary damping
        if boundary == "absorbing":
            damping = _build_sponge_layer(nx, ny, n_layers=20)
        else:
            damping = jnp.ones((nx, ny))

        # Source time functions
        source_positions = [(s.position[0], s.position[1]) for s in self._sources]
        source_freqs = [s.frequency for s in self._sources]
        source_amps = [s.amplitude for s in self._sources]
        source_types = [s.wavelet for s in self._sources]

        def step(
            carry: tuple[Array, Array, int],
            _: None,
        ) -> tuple[tuple[Array, Array, int], Array]:
            u_c, u_p, step_idx = carry
            t = step_idx * dt

            # Laplacian (5-point stencil)
            if boundary == "periodic":
                lap = (
                    jnp.roll(u_c, 1, axis=0)
                    + jnp.roll(u_c, -1, axis=0)
                    + jnp.roll(u_c, 1, axis=1)
                    + jnp.roll(u_c, -1, axis=1)
                    - 4.0 * u_c
                )
            else:
                # Interior Laplacian with zero-padding at edges
                lap = jnp.zeros_like(u_c)
                lap = lap.at[1:-1, 1:-1].set(
                    u_c[2:, 1:-1]
                    + u_c[:-2, 1:-1]
                    + u_c[1:-1, 2:]
                    + u_c[1:-1, :-2]
                    - 4.0 * u_c[1:-1, 1:-1]
                )

            # Time step: u_next = 2*u_curr - u_prev + c^2*dt^2/dx^2 * laplacian
            u_next = 2.0 * u_c - u_p + c2dt2_dx2 * lap

            # Add sources
            for i, (ix, iy) in enumerate(source_positions):
                freq = source_freqs[i]
                amp = source_amps[i]
                wtype = source_types[i]
                if wtype == "ricker":
                    # Ricker wavelet: (1 - 2*pi^2*f^2*t^2) * exp(-pi^2*f^2*t^2)
                    arg = jnp.pi**2 * freq**2 * t**2
                    src_val = amp * (1.0 - 2.0 * arg) * jnp.exp(-arg)
                else:
                    src_val = amp * jnp.sin(2.0 * jnp.pi * freq * t)
                u_next = u_next.at[ix, iy].add(src_val * dt**2)

            # Boundary conditions
            if boundary == "reflecting":
                u_next = u_next.at[0, :].set(u_next[1, :])
                u_next = u_next.at[-1, :].set(u_next[-2, :])
                u_next = u_next.at[:, 0].set(u_next[:, 1])
                u_next = u_next.at[:, -1].set(u_next[:, -2])
            elif boundary == "absorbing":
                u_next = u_next * damping

            return (u_next, u_c, step_idx + 1), u_next

        init = (u_curr, u_prev, 0)
        _, u_all = jax.lax.scan(step, init, None, length=n_steps)

        # Subsample
        if save_every > 1:
            indices = jnp.arange(0, n_steps, save_every)
            u_all = u_all[indices]

        t = jnp.arange(u_all.shape[0], dtype=jnp.float64) * save_every * dt

        return WaveResult(
            t=t,
            u=u_all,
            grid_x=jnp.arange(nx, dtype=jnp.float64) * dx,
            grid_y=jnp.arange(ny, dtype=jnp.float64) * dx,
            speed_field=self._speed_field,
        )


def _build_sponge_layer(nx: int, ny: int, n_layers: int = 20) -> Array:
    """Build a sponge-layer damping profile for absorbing boundaries.

    Returns a 2D array of damping coefficients in [0, 1], with values
    decaying toward 0 at the edges.
    """
    damping = jnp.ones((nx, ny))

    for i in range(n_layers):
        val = 1.0 - 0.5 * ((n_layers - i) / n_layers) ** 2

        # Apply to all 4 edges
        damping = damping.at[i, :].set(jnp.minimum(damping[i, :], val))
        damping = damping.at[nx - 1 - i, :].set(
            jnp.minimum(damping[nx - 1 - i, :], val)
        )
        damping = damping.at[:, i].set(jnp.minimum(damping[:, i], val))
        damping = damping.at[:, ny - 1 - i].set(
            jnp.minimum(damping[:, ny - 1 - i], val)
        )

    return damping
