"""Quantum circuit simulator with differentiable execution.

Provides a circuit builder and state-vector simulator that is fully
compatible with JAX transformations (jit, grad, vmap). Supports
parameterized circuits for variational quantum algorithms.

The simulator maintains a 2^n dimensional complex state vector and
applies gates via matrix-vector products (using tensor reshaping
for efficient multi-qubit operations).

References:
    - Peruzzo et al. "A variational eigenvalue solver on a photonic
      quantum processor" (2014) — VQE
    - Farhi, Goldstone, Gutmann. "A Quantum Approximate Optimization
      Algorithm" (2014) — QAOA
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array

from neurosim.circuits import gates
from neurosim.exceptions import ConfigurationError, DimensionError
from neurosim.state import CircuitResult, CircuitState

logger = logging.getLogger(__name__)


@dataclass
class _GateOp:
    """Internal representation of a gate operation."""

    name: str
    qubits: tuple[int, ...]
    matrix: Array | None = None
    param_fn: str | None = None  # name of gates.Rx, gates.Ry, etc.
    param_idx: int | None = None  # index into parameter vector


class QuantumCircuit:
    """Differentiable quantum circuit simulator.

    Builds a circuit from a sequence of gate operations and executes
    it on a state-vector simulator. Supports parameterized gates for
    variational algorithms with JAX autodiff.

    Example:
        >>> qc = QuantumCircuit(2)
        >>> qc.h(0)
        >>> qc.cnot(0, 1)
        >>> state = qc.run()  # Bell state |00> + |11>

        >>> # Parameterized circuit
        >>> qc = QuantumCircuit(2)
        >>> qc.ry(0, param_idx=0)
        >>> qc.ry(1, param_idx=1)
        >>> qc.cnot(0, 1)
        >>> params = jnp.array([0.5, 0.3])
        >>> state = qc.run(params)

    Args:
        n_qubits: Number of qubits in the circuit.
    """

    def __init__(self, n_qubits: int) -> None:
        if n_qubits < 1:
            raise ConfigurationError(f"Need at least 1 qubit, got {n_qubits}")
        if n_qubits > 20:
            raise ConfigurationError(
                f"State vector simulation limited to 20 qubits, got {n_qubits}"
            )
        self._n_qubits = n_qubits
        self._ops: list[_GateOp] = []
        self._n_params = 0

    @property
    def n_qubits(self) -> int:
        """Number of qubits."""
        return self._n_qubits

    @property
    def n_params(self) -> int:
        """Number of variational parameters."""
        return self._n_params

    @property
    def dim(self) -> int:
        """Hilbert space dimension 2^n."""
        return int(2**self._n_qubits)

    def _validate_qubit(self, q: int) -> None:
        if q < 0 or q >= self._n_qubits:
            raise DimensionError(f"Qubit {q} out of range [0, {self._n_qubits})")

    # ── Single-qubit fixed gates ────────────────────────────────

    def h(self, qubit: int) -> None:
        """Add Hadamard gate."""
        self._validate_qubit(qubit)
        self._ops.append(_GateOp("H", (qubit,), matrix=gates.H()))

    def x(self, qubit: int) -> None:
        """Add Pauli-X gate."""
        self._validate_qubit(qubit)
        self._ops.append(_GateOp("X", (qubit,), matrix=gates.X()))

    def y(self, qubit: int) -> None:
        """Add Pauli-Y gate."""
        self._validate_qubit(qubit)
        self._ops.append(_GateOp("Y", (qubit,), matrix=gates.Y()))

    def z(self, qubit: int) -> None:
        """Add Pauli-Z gate."""
        self._validate_qubit(qubit)
        self._ops.append(_GateOp("Z", (qubit,), matrix=gates.Z()))

    def s(self, qubit: int) -> None:
        """Add S gate."""
        self._validate_qubit(qubit)
        self._ops.append(_GateOp("S", (qubit,), matrix=gates.S()))

    def t(self, qubit: int) -> None:
        """Add T gate."""
        self._validate_qubit(qubit)
        self._ops.append(_GateOp("T", (qubit,), matrix=gates.T()))

    # ── Parameterized single-qubit gates ────────────────────────

    def rx(
        self, qubit: int, angle: float | None = None, param_idx: int | None = None
    ) -> None:
        """Add Rx rotation gate."""
        self._validate_qubit(qubit)
        if angle is not None and param_idx is not None:
            raise ConfigurationError("Specify either angle or param_idx, not both")
        if angle is not None:
            self._ops.append(_GateOp("Rx", (qubit,), matrix=gates.Rx(angle)))
        elif param_idx is not None:
            self._n_params = max(self._n_params, param_idx + 1)
            self._ops.append(
                _GateOp("Rx", (qubit,), param_fn="Rx", param_idx=param_idx)
            )
        else:
            raise ConfigurationError("Rx requires either angle or param_idx")

    def ry(
        self, qubit: int, angle: float | None = None, param_idx: int | None = None
    ) -> None:
        """Add Ry rotation gate."""
        self._validate_qubit(qubit)
        if angle is not None and param_idx is not None:
            raise ConfigurationError("Specify either angle or param_idx, not both")
        if angle is not None:
            self._ops.append(_GateOp("Ry", (qubit,), matrix=gates.Ry(angle)))
        elif param_idx is not None:
            self._n_params = max(self._n_params, param_idx + 1)
            self._ops.append(
                _GateOp("Ry", (qubit,), param_fn="Ry", param_idx=param_idx)
            )
        else:
            raise ConfigurationError("Ry requires either angle or param_idx")

    def rz(
        self, qubit: int, angle: float | None = None, param_idx: int | None = None
    ) -> None:
        """Add Rz rotation gate."""
        self._validate_qubit(qubit)
        if angle is not None and param_idx is not None:
            raise ConfigurationError("Specify either angle or param_idx, not both")
        if angle is not None:
            self._ops.append(_GateOp("Rz", (qubit,), matrix=gates.Rz(angle)))
        elif param_idx is not None:
            self._n_params = max(self._n_params, param_idx + 1)
            self._ops.append(
                _GateOp("Rz", (qubit,), param_fn="Rz", param_idx=param_idx)
            )
        else:
            raise ConfigurationError("Rz requires either angle or param_idx")

    # ── Two-qubit gates ─────────────────────────────────────────

    def cnot(self, control: int, target: int) -> None:
        """Add CNOT gate."""
        self._validate_qubit(control)
        self._validate_qubit(target)
        if control == target:
            raise ConfigurationError("Control and target must be different qubits")
        self._ops.append(_GateOp("CNOT", (control, target), matrix=gates.CNOT()))

    def cz(self, q0: int, q1: int) -> None:
        """Add CZ gate."""
        self._validate_qubit(q0)
        self._validate_qubit(q1)
        if q0 == q1:
            raise ConfigurationError("CZ requires two different qubits")
        self._ops.append(_GateOp("CZ", (q0, q1), matrix=gates.CZ()))

    # ── Execution ───────────────────────────────────────────────

    def run(
        self,
        params: Array | None = None,
        initial_state: Array | None = None,
    ) -> CircuitState:
        """Execute the circuit and return the final state.

        Args:
            params: Parameter vector for variational gates, shape (n_params,).
            initial_state: Initial state vector. Defaults to |00...0>.

        Returns:
            CircuitState with the final state vector.
        """
        n = self._n_qubits
        dim = self.dim

        if initial_state is None:
            state = jnp.zeros(dim, dtype=jnp.complex128)
            state = state.at[0].set(1.0)
        else:
            if initial_state.shape != (dim,):
                raise DimensionError(
                    f"Initial state shape {initial_state.shape} doesn't match dim={dim}"
                )
            state = initial_state

        if self._n_params > 0 and params is None:
            raise ConfigurationError(
                f"Circuit has {self._n_params} parameters but no params provided"
            )

        for op in self._ops:
            if op.param_fn is not None and params is not None:
                # Build gate matrix from parameter
                gate_fn = {"Rx": gates.Rx, "Ry": gates.Ry, "Rz": gates.Rz}[op.param_fn]
                matrix = gate_fn(params[op.param_idx])
            else:
                matrix = op.matrix  # type: ignore[assignment]

            if len(op.qubits) == 1:
                state = _apply_single_qubit_gate(state, matrix, op.qubits[0], n)
            elif len(op.qubits) == 2:
                state = _apply_two_qubit_gate(
                    state, matrix, op.qubits[0], op.qubits[1], n
                )

        return CircuitState(
            state_vector=state,
            n_qubits=n,
        )

    def expectation(
        self,
        observable: Array,
        params: Array | None = None,
    ) -> float:
        """Compute expectation value <psi|O|psi> of an observable.

        Args:
            observable: Hermitian matrix, shape (dim, dim).
            params: Parameter vector for variational gates.

        Returns:
            Expectation value (real scalar).
        """
        state = self.run(params).state_vector
        return float(jnp.real(jnp.conj(state) @ observable @ state))

    def vqe(
        self,
        hamiltonian: Array,
        initial_params: Array,
        n_steps: int = 100,
        learning_rate: float = 0.1,
    ) -> CircuitResult:
        """Run the Variational Quantum Eigensolver (VQE).

        Minimizes <psi(theta)|H|psi(theta)> using gradient descent
        with JAX autodiff.

        Args:
            hamiltonian: Hermitian Hamiltonian matrix, shape (dim, dim).
            initial_params: Starting parameter values.
            n_steps: Number of optimization steps.
            learning_rate: Gradient descent step size.

        Returns:
            CircuitResult with optimized state and energy history.
        """
        if hamiltonian.shape != (self.dim, self.dim):
            raise DimensionError(
                f"Hamiltonian shape {hamiltonian.shape} doesn't match dim={self.dim}"
            )

        logger.info(
            "Starting VQE: n_qubits=%d, n_params=%d, n_steps=%d",
            self._n_qubits,
            self._n_params,
            n_steps,
        )

        def energy_fn(p: Array) -> Array:
            state = self.run(p).state_vector
            return jnp.real(jnp.conj(state) @ hamiltonian @ state)

        grad_fn = jax.grad(energy_fn)

        params = initial_params
        energies = []

        for _ in range(n_steps):
            e = energy_fn(params)
            energies.append(float(e))
            grad = grad_fn(params)
            params = params - learning_rate * grad

        final_state = self.run(params).state_vector

        return CircuitResult(
            state_vector=final_state,
            n_qubits=self._n_qubits,
            expectation=float(energy_fn(params)),
            parameters=params,
            energy_history=jnp.array(energies),
        )


def _apply_single_qubit_gate(
    state: Array, gate: Array, qubit: int, n_qubits: int
) -> Array:
    """Apply a single-qubit gate to the state vector.

    Reshapes the state to (2, 2, ..., 2) tensor, applies the gate
    along the target qubit axis, then flattens back.
    """
    dim = 2**n_qubits
    psi = state.reshape([2] * n_qubits)
    psi = jnp.tensordot(gate, psi, axes=([1], [qubit]))
    psi = jnp.moveaxis(psi, 0, qubit)
    return psi.reshape(dim)


def _apply_two_qubit_gate(
    state: Array, gate: Array, q0: int, q1: int, n_qubits: int
) -> Array:
    """Apply a two-qubit gate to the state vector.

    Uses the reshape-and-contract approach for arbitrary qubit pairs.
    """
    dim = 2**n_qubits
    psi = state.reshape([2] * n_qubits)
    gate_4 = gate.reshape(2, 2, 2, 2)

    # Contract gate indices (2,3) with qubit axes (q0, q1)
    psi = jnp.tensordot(gate_4, psi, axes=([2, 3], [q0, q1]))

    # Move gate output axes (0, 1) back to (q0, q1)
    # After tensordot, output axes 0,1 are at the front
    source = [0, 1]
    dest = sorted([q0, q1])
    psi = jnp.moveaxis(psi, source, dest)

    return psi.reshape(dim)
