"""Special relativistic Lorentz transformations.

Provides utilities for Lorentz boosts, time dilation, length
contraction, and relativistic velocity addition.

All functions use natural units where c = 1 unless otherwise noted.

References:
    - Einstein. "On the Electrodynamics of Moving Bodies" (1905)
    - Jackson. "Classical Electrodynamics" (1999), Ch. 11
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from neurosim.exceptions import PhysicsError


def lorentz_factor(v: Array | float) -> Array:
    """Compute the Lorentz factor gamma = 1/sqrt(1 - v^2/c^2).

    Args:
        v: Velocity (or array of velocities) in units of c.

    Returns:
        Lorentz factor gamma.

    Raises:
        PhysicsError: If |v| >= 1 (superluminal).
    """
    v = jnp.asarray(v, dtype=jnp.float64)
    v2 = jnp.sum(v**2) if v.ndim > 0 and v.shape[-1] > 1 else v**2
    if jnp.any(v2 >= 1.0):
        raise PhysicsError(
            f"Velocity |v|={float(jnp.sqrt(jnp.max(v2))):.4f} >= c (superluminal)"
        )
    return 1.0 / jnp.sqrt(1.0 - v2)


def lorentz_boost(four_vector: Array, v: Array | float) -> Array:
    """Apply a Lorentz boost to a four-vector.

    Boosts the four-vector (ct, x, y, z) by velocity v along the
    x-direction (if scalar) or along the direction of v (if 3-vector).

    Args:
        four_vector: Four-vector [ct, x, y, z], shape (4,).
        v: Boost velocity in units of c. Scalar for x-boost, or
            3-vector for arbitrary direction.

    Returns:
        Boosted four-vector, shape (4,).
    """
    four_vector = jnp.asarray(four_vector, dtype=jnp.float64)
    v = jnp.asarray(v, dtype=jnp.float64)

    if v.ndim == 0:
        # Boost along x-axis
        gamma = 1.0 / jnp.sqrt(1.0 - v**2)
        ct, x, y, z = four_vector
        ct_new = gamma * (ct - v * x)
        x_new = gamma * (x - v * ct)
        return jnp.array([ct_new, x_new, y, z])

    # General boost along direction of v
    v_mag = jnp.linalg.norm(v)
    gamma = 1.0 / jnp.sqrt(1.0 - v_mag**2)
    n = v / jnp.maximum(v_mag, 1e-15)  # unit vector

    ct = four_vector[0]
    r = four_vector[1:]

    r_parallel = jnp.dot(r, n) * n
    r_perp = r - r_parallel

    ct_new = gamma * (ct - jnp.dot(v, r))
    r_new = r_perp + gamma * (r_parallel - v * ct)

    return jnp.concatenate([jnp.array([ct_new]), r_new])


def velocity_addition(v1: float, v2: float) -> float:
    """Relativistic velocity addition.

    Computes the combined velocity of two collinear boosts:
        v = (v1 + v2) / (1 + v1*v2/c^2)

    Args:
        v1: First velocity in units of c.
        v2: Second velocity in units of c.

    Returns:
        Combined velocity in units of c.
    """
    return float((v1 + v2) / (1.0 + v1 * v2))


def proper_time(v: Array, dt: Array) -> Array:
    """Compute proper time interval for a moving observer.

    dtau = dt * sqrt(1 - v^2/c^2)

    Args:
        v: Velocity at each time step, shape (n_steps,) or scalar.
        dt: Coordinate time intervals, shape (n_steps,) or scalar.

    Returns:
        Proper time intervals dtau.
    """
    v = jnp.asarray(v, dtype=jnp.float64)
    dt = jnp.asarray(dt, dtype=jnp.float64)
    return dt * jnp.sqrt(1.0 - v**2)
