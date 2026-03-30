"""Variational Quantum Eigensolver (VQE) demo.

Uses the quantum circuit simulator to find the ground-state energy
of a simple Hamiltonian via parameterized circuits and JAX autodiff.
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt

from neurosim.circuits.simulator import QuantumCircuit

# Hamiltonian: -Z (eigenvalues +1, -1 → ground state energy = -1)
ham = -jnp.array([[1, 0], [0, -1]], dtype=jnp.complex128)

# Build a single-qubit parameterized ansatz: Ry(theta)|0>
qc = QuantumCircuit(1)
qc.ry(0, param_idx=0)

# Run VQE optimization
result = qc.vqe(ham, initial_params=jnp.array([1.0]), n_steps=100, learning_rate=0.3)

print(f"Ground-state energy: {result.expectation:.6f}  (exact: -1.0)")
print(f"Optimal parameter:   {float(result.parameters[0]):.6f}")

# Plot convergence
plt.figure(figsize=(8, 4))
plt.plot(result.energy_history)
plt.axhline(-1.0, color="r", ls="--", label="Exact ground state")
plt.xlabel("Optimization step")
plt.ylabel("Energy")
plt.title("VQE Convergence")
plt.legend()
plt.tight_layout()
plt.savefig("vqe_convergence.png", dpi=150)
plt.show()

# --- Bell state demo ---
bell = QuantumCircuit(2)
bell.h(0)
bell.cnot(0, 1)
state = bell.run()
print(f"\nBell state probabilities: {state.probabilities}")
