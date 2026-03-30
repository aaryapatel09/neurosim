"""Tests for quantum circuit simulator."""

import jax
import jax.numpy as jnp
import pytest

from neurosim.circuits.gates import CNOT, CZ, H, Rx, Ry, Rz, S, T, X, Y, Z, phase
from neurosim.circuits.simulator import QuantumCircuit
from neurosim.exceptions import ConfigurationError, DimensionError

jax.config.update("jax_enable_x64", True)


class TestGates:
    """Tests for quantum gate matrices."""

    def test_hadamard_unitary(self) -> None:
        """H @ H^dag = I."""
        h = H()
        product = h @ jnp.conj(h.T)
        assert jnp.allclose(product, jnp.eye(2), atol=1e-12)

    def test_pauli_x_squared(self) -> None:
        """X^2 = I."""
        x = X()
        assert jnp.allclose(x @ x, jnp.eye(2), atol=1e-12)

    def test_pauli_y_squared(self) -> None:
        """Y^2 = I."""
        y = Y()
        assert jnp.allclose(y @ y, jnp.eye(2), atol=1e-12)

    def test_pauli_z_squared(self) -> None:
        """Z^2 = I."""
        z = Z()
        assert jnp.allclose(z @ z, jnp.eye(2), atol=1e-12)

    def test_rx_zero_is_identity(self) -> None:
        rx = Rx(0.0)
        assert jnp.allclose(rx, jnp.eye(2), atol=1e-12)

    def test_ry_pi_is_y(self) -> None:
        """Ry(pi) should be proportional to Y."""
        ry = Ry(jnp.pi)
        # Ry(pi) = [[0, -1], [1, 0]] = -i*Y (up to global phase)
        expected = jnp.array([[0.0, -1.0], [1.0, 0.0]], dtype=jnp.complex128)
        assert jnp.allclose(ry, expected, atol=1e-10)

    def test_rz_unitary(self) -> None:
        rz = Rz(1.23)
        product = rz @ jnp.conj(rz.T)
        assert jnp.allclose(product, jnp.eye(2), atol=1e-12)

    def test_cnot_shape(self) -> None:
        assert CNOT().shape == (4, 4)

    def test_cz_diagonal(self) -> None:
        cz = CZ()
        assert jnp.allclose(
            jnp.diag(cz), jnp.array([1, 1, 1, -1], dtype=jnp.complex128)
        )

    def test_s_squared_is_z(self) -> None:
        """S^2 = Z."""
        s = S()
        assert jnp.allclose(s @ s, Z(), atol=1e-12)

    def test_t_fourth_is_z(self) -> None:
        """T^4 = Z."""
        t = T()
        t4 = t @ t @ t @ t
        assert jnp.allclose(t4, Z(), atol=1e-10)

    def test_phase_gate(self) -> None:
        p = phase(jnp.pi / 2)
        assert jnp.allclose(p, S(), atol=1e-12)


class TestQuantumCircuit:
    """Tests for the circuit simulator."""

    def test_initial_state(self) -> None:
        """Default state should be |0>."""
        qc = QuantumCircuit(1)
        state = qc.run()
        assert jnp.allclose(state.state_vector, jnp.array([1, 0], dtype=jnp.complex128))

    def test_hadamard_superposition(self) -> None:
        """H|0> = (|0> + |1>)/sqrt(2)."""
        qc = QuantumCircuit(1)
        qc.h(0)
        state = qc.run()
        expected = jnp.array([1, 1], dtype=jnp.complex128) / jnp.sqrt(2)
        assert jnp.allclose(state.state_vector, expected, atol=1e-12)

    def test_bell_state(self) -> None:
        """H on q0 then CNOT should create Bell state."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cnot(0, 1)
        state = qc.run()

        probs = state.probabilities
        assert float(probs[0]) == pytest.approx(0.5, abs=1e-10)  # |00>
        assert float(probs[3]) == pytest.approx(0.5, abs=1e-10)  # |11>
        assert float(probs[1]) == pytest.approx(0.0, abs=1e-10)  # |01>
        assert float(probs[2]) == pytest.approx(0.0, abs=1e-10)  # |10>

    def test_x_gate_flips(self) -> None:
        """X|0> = |1>."""
        qc = QuantumCircuit(1)
        qc.x(0)
        state = qc.run()
        assert jnp.allclose(state.state_vector, jnp.array([0, 1], dtype=jnp.complex128))

    def test_norm_preserved(self) -> None:
        """Circuit should preserve state norm."""
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cnot(0, 1)
        qc.ry(2, angle=0.7)
        qc.cz(1, 2)
        state = qc.run()
        norm = float(jnp.sum(jnp.abs(state.state_vector) ** 2))
        assert norm == pytest.approx(1.0, rel=1e-10)

    def test_parameterized_circuit(self) -> None:
        """Parameterized Ry gates should respond to parameters."""
        qc = QuantumCircuit(1)
        qc.ry(0, param_idx=0)
        params = jnp.array([jnp.pi])
        state = qc.run(params)
        # Ry(pi)|0> = |1>
        assert float(jnp.abs(state.state_vector[1]) ** 2) == pytest.approx(
            1.0, abs=1e-10
        )

    def test_expectation_value(self) -> None:
        """<0|Z|0> = 1."""
        qc = QuantumCircuit(1)
        z_obs = jnp.array([[1, 0], [0, -1]], dtype=jnp.complex128)
        exp = qc.expectation(z_obs)
        assert exp == pytest.approx(1.0, abs=1e-10)

    def test_expectation_after_x(self) -> None:
        """<1|Z|1> = -1."""
        qc = QuantumCircuit(1)
        qc.x(0)
        z_obs = jnp.array([[1, 0], [0, -1]], dtype=jnp.complex128)
        exp = qc.expectation(z_obs)
        assert exp == pytest.approx(-1.0, abs=1e-10)

    def test_vqe_simple(self) -> None:
        """VQE should find ground state of Z (which is |0>, energy=-1... wait, Z has eigenvalues +1,-1)."""
        # Hamiltonian: -Z (ground state is |0> with energy -1)
        ham = -jnp.array([[1, 0], [0, -1]], dtype=jnp.complex128)

        qc = QuantumCircuit(1)
        qc.ry(0, param_idx=0)
        result = qc.vqe(
            ham, initial_params=jnp.array([1.0]), n_steps=50, learning_rate=0.3
        )

        # Should converge near -1
        assert result.expectation is not None
        assert result.expectation < -0.9
        assert result.energy_history is not None
        assert result.parameters is not None

    def test_differentiable(self) -> None:
        """Circuit should be differentiable through JAX."""
        qc = QuantumCircuit(1)
        qc.ry(0, param_idx=0)
        z_obs = jnp.array([[1, 0], [0, -1]], dtype=jnp.complex128)

        def cost(p: jnp.ndarray) -> jnp.ndarray:
            state = qc.run(p).state_vector
            return jnp.real(jnp.conj(state) @ z_obs @ state)

        grad = jax.grad(cost)(jnp.array([0.5]))
        assert grad.shape == (1,)
        assert float(jnp.abs(grad[0])) > 0  # Non-zero gradient

    def test_qubit_out_of_range(self) -> None:
        qc = QuantumCircuit(2)
        with pytest.raises(DimensionError):
            qc.h(3)

    def test_cnot_same_qubit_error(self) -> None:
        qc = QuantumCircuit(2)
        with pytest.raises(ConfigurationError):
            qc.cnot(0, 0)

    def test_too_many_qubits_error(self) -> None:
        with pytest.raises(ConfigurationError):
            QuantumCircuit(25)

    def test_missing_params_error(self) -> None:
        qc = QuantumCircuit(1)
        qc.ry(0, param_idx=0)
        with pytest.raises(ConfigurationError):
            qc.run()  # No params provided

    def test_n_qubits_property(self) -> None:
        qc = QuantumCircuit(3)
        assert qc.n_qubits == 3
        assert qc.dim == 8
