"""
Example: VQE simulation for H2 molecule
"""

import numpy as np
from vqe_algorithm import VQE
from ansatz import RotationLayerAnsatz, HartreeFockAnsatz
from hamiltonian import H2Hamiltonian
from utils import VQEAnalyzer, compute_fidelity

# Setup
n_qubits = 4
n_electrons = 2

# Create ansatz (hardware-efficient)
ansatz = RotationLayerAnsatz(n_qubits=n_qubits, n_layers=2)

# Create H2 Hamiltonian
hamiltonian = H2Hamiltonian(bond_length=0.735)

# Create VQE solver
vqe = VQE(
    ansatz=ansatz,
    hamiltonian=hamiltonian,
    optimizer='COBYLA',
    max_iterations=200,
    tol=1e-6
)

# Run optimization
initial_params = np.random.randn(ansatz.n_params) * 0.1
optimal_params, min_energy = vqe.minimize(initial_params)

print(f"\n=== VQE Results for H2 ===")
print(f"Optimized Energy: {min_energy:.8f} Ha")
print(f"Circuit Parameters: {optimal_params}")
print(f"Number of Parameters: {ansatz.n_params}")

# Analyze results
history = vqe.get_energy_history()
analyzer = VQEAnalyzer()
analysis = analyzer.analyze_optimization(history, true_energy=-1.137)

print(f"\n=== Analysis ===")
for key, value in analysis.items():
    print(f"{key}: {value}")

# Plot convergence
analyzer.plot_convergence(history, true_energy=-1.137)