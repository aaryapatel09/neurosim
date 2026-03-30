"""Quantum gate definitions.

Provides standard single-qubit and two-qubit gate matrices as JAX arrays.
All parameterized gates are differentiable through JAX autodiff.

Gate conventions follow the standard quantum computing literature:
    - Computational basis: |0> = [1, 0], |1> = [0, 1]
    - Matrices act on state vectors via left-multiplication
    - Two-qubit gates use the tensor product convention |q0 q1>

References:
    - Nielsen & Chuang. "Quantum Computation and Quantum Information" (2000)
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

# ── Single-qubit gates ──────────────────────────────────────────────


def H() -> Array:
    """Hadamard gate."""
    return jnp.array([[1, 1], [1, -1]], dtype=jnp.complex128) / jnp.sqrt(2)


def X() -> Array:
    """Pauli-X (NOT) gate."""
    return jnp.array([[0, 1], [1, 0]], dtype=jnp.complex128)


def Y() -> Array:
    """Pauli-Y gate."""
    return jnp.array([[0, -1j], [1j, 0]], dtype=jnp.complex128)


def Z() -> Array:
    """Pauli-Z gate."""
    return jnp.array([[1, 0], [0, -1]], dtype=jnp.complex128)


def S() -> Array:
    """S (phase pi/2) gate."""
    return jnp.array([[1, 0], [0, 1j]], dtype=jnp.complex128)


def T() -> Array:
    """T (phase pi/4) gate."""
    return jnp.array([[1, 0], [0, jnp.exp(1j * jnp.pi / 4)]], dtype=jnp.complex128)


def Rx(theta: float | Array) -> Array:
    """Rotation about X-axis by angle theta.

    Rx(theta) = exp(-i * theta/2 * X) = [[cos(t/2), -i*sin(t/2)],
                                          [-i*sin(t/2), cos(t/2)]]

    Args:
        theta: Rotation angle in radians.

    Returns:
        2x2 unitary matrix.
    """
    c = jnp.cos(theta / 2)
    s = jnp.sin(theta / 2)
    return jnp.array([[c, -1j * s], [-1j * s, c]], dtype=jnp.complex128)


def Ry(theta: float | Array) -> Array:
    """Rotation about Y-axis by angle theta.

    Ry(theta) = exp(-i * theta/2 * Y) = [[cos(t/2), -sin(t/2)],
                                          [sin(t/2), cos(t/2)]]
    """
    c = jnp.cos(theta / 2)
    s = jnp.sin(theta / 2)
    return jnp.array([[c, -s], [s, c]], dtype=jnp.complex128)


def Rz(theta: float | Array) -> Array:
    """Rotation about Z-axis by angle theta.

    Rz(theta) = exp(-i * theta/2 * Z) = [[e^{-it/2}, 0],
                                          [0, e^{it/2}]]
    """
    return jnp.array(
        [[jnp.exp(-1j * theta / 2), 0], [0, jnp.exp(1j * theta / 2)]],
        dtype=jnp.complex128,
    )


def phase(phi: float | Array) -> Array:
    """Phase gate with arbitrary angle.

    P(phi) = [[1, 0], [0, e^{i*phi}]]
    """
    return jnp.array([[1, 0], [0, jnp.exp(1j * phi)]], dtype=jnp.complex128)


# ── Two-qubit gates ────────────────────────────────────────────────


def CNOT() -> Array:
    """Controlled-NOT (CX) gate.

    Acts on |control, target>. Flips target if control is |1>.
    """
    return jnp.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=jnp.complex128,
    )


def CZ() -> Array:
    """Controlled-Z gate.

    Applies Z to target if control is |1>.
    """
    return jnp.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]],
        dtype=jnp.complex128,
    )
