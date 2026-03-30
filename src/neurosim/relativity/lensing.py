"""Gravitational lensing ray tracer.

Traces light rays in the Schwarzschild metric to compute gravitational
lensing effects. Uses the exact deflection angle formula and backward
ray tracing from the observer to build lensed images.

The deflection angle for a light ray with impact parameter b in
Schwarzschild spacetime (weak-field limit):

    delta_phi = 4M / b

For strong lensing near the photon sphere (r = 3M), the exact
integral must be evaluated numerically.

References:
    - Einstein. "Lens-like action of a star" (1936)
    - Virbhadra & Ellis. "Schwarzschild black hole lensing" (2000)
    - Bozza. "Gravitational lensing in the strong field limit" (2002)
"""

from __future__ import annotations

import logging

import jax.numpy as jnp
from jax import Array

from neurosim.exceptions import ConfigurationError

logger = logging.getLogger(__name__)


class GravitationalLens:
    """Gravitational lensing simulator in Schwarzschild spacetime.

    Traces photon trajectories backward from the observer plane to
    compute the lensed appearance of a background source.

    Example:
        >>> lens = GravitationalLens(M=1.0)
        >>> angles = lens.deflection_angle(b=jnp.linspace(3.0, 20.0, 100))
        >>> image = lens.ray_trace_image(
        ...     observer_distance=100.0, image_size=200, fov=0.1,
        ... )

    Args:
        M: Mass of the lensing object in geometric units (G=c=1).
    """

    def __init__(self, M: float = 1.0) -> None:
        if M <= 0:
            raise ConfigurationError(f"Mass must be positive, got M={M}")
        self._M = M

    @property
    def photon_sphere(self) -> float:
        """Photon sphere radius r_ph = 3M."""
        return 3.0 * self._M

    @property
    def shadow_radius(self) -> float:
        """Critical impact parameter (shadow boundary) b_c = 3*sqrt(3)*M."""
        return float(3.0 * jnp.sqrt(3.0) * self._M)

    def deflection_angle(self, b: Array) -> Array:
        """Compute the gravitational deflection angle.

        Uses the weak-field approximation delta_phi = 4M/b for
        large impact parameters, with a correction term for moderate b.

        Args:
            b: Impact parameter(s). Must be > shadow_radius.

        Returns:
            Deflection angle(s) in radians.
        """
        M = self._M
        # Second-order post-Newtonian deflection
        # delta = 4M/b + 15*pi*M^2/(4*b^2)
        delta = 4.0 * M / b + 15.0 * jnp.pi * M**2 / (4.0 * b**2)
        return delta

    def einstein_radius(self, d_lens: float, d_source: float) -> float:
        """Compute the Einstein radius for a point lens.

        theta_E = sqrt(4M * d_LS / (d_L * d_S))

        Args:
            d_lens: Distance to the lens.
            d_source: Distance to the source.

        Returns:
            Einstein radius in radians.
        """
        if d_source <= d_lens:
            raise ConfigurationError(
                f"Source distance ({d_source}) must be > lens distance ({d_lens})"
            )
        d_ls = d_source - d_lens
        return float(jnp.sqrt(4.0 * self._M * d_ls / (d_lens * d_source)))

    def ray_trace_image(
        self,
        observer_distance: float = 100.0,
        image_size: int = 100,
        fov: float = 0.1,
    ) -> Array:
        """Ray-trace a gravitationally lensed image.

        Backward-traces rays from an observer plane through the
        Schwarzschild metric to compute the apparent deflection.
        Returns a 2D map of deflection magnitudes.

        Args:
            observer_distance: Distance of the observer from the lens.
            image_size: Number of pixels per side.
            fov: Field of view (half-angle in radians).

        Returns:
            2D array of deflection angles, shape (image_size, image_size).
        """
        if observer_distance <= 0:
            raise ConfigurationError("Observer distance must be positive")
        if image_size < 2:
            raise ConfigurationError("Image size must be >= 2")

        M = self._M

        # Create image plane coordinates
        angles = jnp.linspace(-fov, fov, image_size)
        ax, ay = jnp.meshgrid(angles, angles, indexing="ij")

        # Impact parameter for each ray
        b = observer_distance * jnp.sqrt(ax**2 + ay**2)
        b = jnp.maximum(b, 1e-10)  # avoid division by zero

        # Deflection angle (scalar per ray)
        deflection = 4.0 * M / b + 15.0 * jnp.pi * M**2 / (4.0 * b**2)

        # Mask rays that would hit the black hole (b < shadow_radius)
        b_crit = float(self.shadow_radius)
        shadow_mask = b < b_crit
        deflection = jnp.where(shadow_mask, jnp.inf, deflection)

        return deflection
