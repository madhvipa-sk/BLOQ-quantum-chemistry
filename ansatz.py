"""
Quantum Circuit Ansätze for VQE
Various parameterized circuit designs for molecular simulations
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import List

class QuantumAnsatz(ABC):
    """Abstract base class for quantum ansatze."""
    
    def __init__(self, n_qubits: int):
        """
        Initialize ansatz.
        
        Args:
            n_qubits: Number of qubits
        """
        self.n_qubits = n_qubits
        self.n_params = 0
    
    @abstractmethod
    def construct_circuit(self, params: np.ndarray) -> np.ndarray:
        """Construct and return quantum state vector."""
        pass


class HartreeFockAnsatz(QuantumAnsatz):
    """Hartree-Fock reference state for quantum chemistry."""
    
    def __init__(self, n_qubits: int, n_electrons: int):
        """
        Initialize HF ansatz.
        
        Args:
            n_qubits: Number of qubits
            n_electrons: Number of electrons
        """
        super().__init__(n_qubits)
        self.n_electrons = n_electrons
        self.n_params = 0
    
    def construct_circuit(self, params: np.ndarray = None) -> np.ndarray:
        """
        Construct Hartree-Fock reference state.
        Initialize qubits corresponding to occupied orbitals.
        """
        state = np.zeros(2**self.n_qubits, dtype=complex)
        
        # Create computational basis state with occupied orbitals
        occupied_state = 0
        for i in range(self.n_electrons):
            occupied_state |= (1 << i)
        
        state[occupied_state] = 1.0
        return state


class UCCAnsatzSingle(QuantumAnsatz):
    """
    Unitary Coupled Cluster ansatz with single excitations (UCCSD-like).
    Parameterized circuit for quantum chemistry.
    """
    
    def __init__(self, n_qubits: int, n_electrons: int, excitation_level: int = 1):
        """
        Initialize UCC ansatz.
        
        Args:
            n_qubits: Number of qubits
            n_electrons: Number of electrons
            excitation_level: 1 for singles, 2 for doubles
        """
        super().__init__(n_qubits)
        self.n_electrons = n_electrons
        self.excitation_level = excitation_level
        
        # Count excitations
        n_occupied = n_electrons
        n_virtual = n_qubits - n_electrons
        
        if excitation_level >= 1:
            self.n_single_excitations = n_occupied * n_virtual
        if excitation_level >= 2:
            n_double_excitations = (n_occupied * (n_occupied - 1) // 2) * \
                                   (n_virtual * (n_virtual - 1) // 2)
            self.n_params = self.n_single_excitations + n_double_excitations
        else:
            self.n_params = self.n_single_excitations
    
    def construct_circuit(self, params: np.ndarray) -> np.ndarray:
        """
        Construct UCC ansatz state.
        |ψ⟩ = exp(-i * Σ θ_i T_i) |HF⟩
        """
        # Start with HF state
        state = self._hartree_fock_state()
        
        # Apply single excitation rotations
        param_idx = 0
        n_occupied = self.n_electrons
        n_virtual = self.n_qubits - self.n_electrons
        
        for i in range(n_occupied):
            for a in range(n_virtual):
                theta = params[param_idx] if param_idx < len(params) else 0
                # Apply controlled rotation for excitation i -> a + n_occupied
                state = self._apply_single_excitation(state, i, a + n_occupied, theta)
                param_idx += 1
        
        return state
    
    def _hartree_fock_state(self) -> np.ndarray:
        """Generate Hartree-Fock reference state."""
        state = np.zeros(2**self.n_qubits, dtype=complex)
        hf_config = (1 << self.n_electrons) - 1
        state[hf_config] = 1.0
        return state
    
    def _apply_single_excitation(
        self,
        state: np.ndarray,
        i: int,
        j: int,
        theta: float
    ) -> np.ndarray:
        """Apply single excitation operator e^(-i θ (a_i† a_j - a_j† a_i))."""
        new_state = state.copy()
        dim = len(state)
        
        for config in range(dim):
            # Check if configuration has qubit i occupied and j unoccupied
            if ((config >> i) & 1) and not ((config >> j) & 1):
                # Create excited configuration
                excited_config = config ^ (1 << i) ^ (1 << j)
                
                # Apply rotation
                c = np.cos(theta / 2)
                s = np.sin(theta / 2)
                
                new_state[config] = c * state[config] - 1j * s * state[excited_config]
                new_state[excited_config] = -1j * s * state[config] + c * state[excited_config]
        
        return new_state


class RotationLayerAnsatz(QuantumAnsatz):
    """
    Hardware-efficient ansatz with rotation layers.
    Suitable for NISQ devices.
    """
    
    def __init__(self, n_qubits: int, n_layers: int = 2):
        """
        Initialize rotation layer ansatz.
        
        Args:
            n_qubits: Number of qubits
            n_layers: Number of rotation layers
        """
        super().__init__(n_qubits)
        self.n_layers = n_layers
        # 3 rotation angles per qubit per layer + entangling angles
        self.n_params = n_layers * (3 * n_qubits + (n_qubits - 1))
    
    def construct_circuit(self, params: np.ndarray) -> np.ndarray:
        """
        Construct rotation layer ansatz.
        Alternating single-qubit rotations and entangling gates.
        """
        dim = 2**self.n_qubits
        state = np.zeros(dim, dtype=complex)
        state[0] = 1.0  # Initialize to |0...0⟩
        
        param_idx = 0
        
        for layer in range(self.n_layers):
            # Single-qubit rotations (RY(θ) RZ(φ))
            for q in range(self.n_qubits):
                theta = params[param_idx]
                phi = params[param_idx + 1]
                gamma = params[param_idx + 2]
                param_idx += 3
                
                state = self._apply_u3(state, q, theta, phi, gamma)
            
            # Entangling gates (CNOT ladder)
            for q in range(self.n_qubits - 1):
                angle = params[param_idx]
                param_idx += 1
                state = self._apply_entangler(state, q, q + 1, angle)
        
        return state
    
    def _apply_u3(
        self,
        state: np.ndarray,
        qubit: int,
        theta: float,
        phi: float,
        gamma: float
    ) -> np.ndarray:
        """Apply U3 gate: U3(θ,φ,λ) = [[cos(θ/2), -e^(iλ)sin(θ/2)], [e^(iφ)sin(θ/2), e^(i(φ+λ))cos(θ/2)]]"""
        new_state = state.copy()
        
        c = np.cos(theta / 2)
        s = np.sin(theta / 2)
        
        for config in range(len(state)):
            if not (config >> qubit & 1):
                excited = config ^ (1 << qubit)
                new_state[config] = c * state[config] - np.exp(1j * gamma) * s * state[excited]
                new_state[excited] = np.exp(1j * phi) * s * state[config] + np.exp(1j * (phi + gamma)) * c * state[excited]
        
        return new_state
    
    def _apply_entangler(
        self,
        state: np.ndarray,
        q1: int,
        q2: int,
        angle: float
    ) -> np.ndarray:
        """Apply ZZ entangling gate: e^(-i θ Z⊗Z)"""
        new_state = state.copy()
        
        for config in range(len(state)):
            z1 = 1 if (config >> q1) & 1 else -1
            z2 = 1 if (config >> q2) & 1 else -1
            phase = np.exp(-1j * angle * z1 * z2 / 2)
            new_state[config] *= phase
        
        return new_state