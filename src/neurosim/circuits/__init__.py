"""Quantum circuit simulator module.

Provides a differentiable quantum circuit simulator with common gates,
parameterized circuits for variational algorithms (VQE/QAOA), and
full JAX autodiff support for gradient-based circuit optimization.
"""

from neurosim.circuits.gates import (
    CNOT,
    CZ,
    H,
    Rx,
    Ry,
    Rz,
    S,
    T,
    X,
    Y,
    Z,
    phase,
)
from neurosim.circuits.simulator import QuantumCircuit

__all__ = [
    "QuantumCircuit",
    "H",
    "X",
    "Y",
    "Z",
    "S",
    "T",
    "Rx",
    "Ry",
    "Rz",
    "phase",
    "CNOT",
    "CZ",
]
