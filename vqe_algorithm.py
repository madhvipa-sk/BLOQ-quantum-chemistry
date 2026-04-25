"""
Variational Quantum Eigensolver (VQE) - Main Implementation
Solves ground state energy of molecular Hamiltonians using quantum-classical hybrid approach
"""

import numpy as np
from scipy.optimize import minimize
from typing import Callable, Tuple, List
import warnings

class VQE:
    """
    Variational Quantum Eigensolver (VQE) solver for quantum chemistry problems.
    Uses a parameterized quantum circuit and classical optimization.
    """
    
    def __init__(
        self,
        ansatz: 'QuantumAnsatz',
        hamiltonian: 'MolecularHamiltonian',
        backend: str = 'statevector',
        optimizer: str = 'COBYLA',
        max_iterations: int = 1000,
        tol: float = 1e-6
    ):
        """
        Initialize VQE solver.
        
        Args:
            ansatz: Parameterized quantum circuit
            hamiltonian: Molecular Hamiltonian to minimize
            backend: Quantum simulator backend ('statevector', 'qasm')
            optimizer: Classical optimizer ('COBYLA', 'SLSQP', 'Powell')
            max_iterations: Maximum optimization iterations
            tol: Convergence tolerance
        """
        self.ansatz = ansatz
        self.hamiltonian = hamiltonian
        self.backend = backend
        self.optimizer_name = optimizer
        self.max_iterations = max_iterations
        self.tol = tol
        self.history = []
        self.optimal_params = None
        self.min_energy = None
        
    def cost_function(self, params: np.ndarray) -> float:
        """
        Compute energy expectation value for given circuit parameters.
        
        Args:
            params: Circuit parameters
            
        Returns:
            Energy expectation value
        """
        # Get quantum state from ansatz
        state = self.ansatz.construct_circuit(params)
        
        # Compute energy expectation value
        energy = 0.0
        for coeff, pauli_str in self.hamiltonian.get_terms():
            expectation = self._measure_pauli_expectation(state, pauli_str)
            energy += coeff * expectation
        
        self.history.append(energy)
        return energy
    
    def _measure_pauli_expectation(
        self,
        state: np.ndarray,
        pauli_str: str
    ) -> float:
        """
        Compute expectation value of Pauli operator.
        
        Args:
            state: Quantum state vector
            pauli_str: Pauli string (e.g., 'ZZII')
            
        Returns:
            Expectation value
        """
        n_qubits = len(pauli_str)
        
        # Build Pauli matrix
        pauli_matrix = self._build_pauli_matrix(pauli_str, n_qubits)
        
        # Compute <ψ|P|ψ>
        expectation = np.real(np.conj(state) @ pauli_matrix @ state)
        
        return expectation
    
    @staticmethod
    def _build_pauli_matrix(pauli_str: str, n_qubits: int) -> np.ndarray:
        """Build full Pauli matrix from Pauli string."""
        pauli_matrices = {
            'I': np.array([[1, 0], [0, 1]], dtype=complex),
            'X': np.array([[0, 1], [1, 0]], dtype=complex),
            'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
            'Z': np.array([[1, 0], [0, -1]], dtype=complex)
        }
        
        matrix = pauli_matrices[pauli_str[0]]
        for pauli in pauli_str[1:]:
            matrix = np.kron(matrix, pauli_matrices[pauli])
        
        return matrix
    
    def minimize(self, initial_params: np.ndarray = None) -> Tuple[np.ndarray, float]:
        """
        Perform VQE optimization.
        
        Args:
            initial_params: Initial circuit parameters
            
        Returns:
            Tuple of (optimal_params, minimum_energy)
        """
        if initial_params is None:
            initial_params = np.random.randn(self.ansatz.n_params) * 0.1
        
        print(f"Starting VQE optimization...")
        print(f"Optimizer: {self.optimizer_name}")
        print(f"Circuit parameters: {self.ansatz.n_params}")
        
        # Define optimization options
        options = {
            'maxiter': self.max_iterations,
            'tol': self.tol,
            'disp': True
        }
        
        # Run optimization
        result = minimize(
            self.cost_function,
            initial_params,
            method=self.optimizer_name,
            options=options
        )
        
        self.optimal_params = result.x
        self.min_energy = result.fun
        
        print(f"\nOptimization complete!")
        print(f"Minimum energy: {self.min_energy:.8f}")
        print(f"Iterations: {len(self.history)}")
        
        return self.optimal_params, self.min_energy
    
    def get_final_state(self) -> np.ndarray:
        """Get final quantum state with optimal parameters."""
        if self.optimal_params is None:
            raise ValueError("Must run minimize() first")
        return self.ansatz.construct_circuit(self.optimal_params)
    
    def get_energy_history(self) -> List[float]:
        """Get optimization energy history."""
        return self.history.copy()