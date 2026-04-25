"""
Utility functions for VQE simulation
Includes visualization, analysis, and helper functions
"""

import numpy as np
import warnings
from typing import List, Tuple
import matplotlib.pyplot as plt

class VQEAnalyzer:
    """Analyzer class for VQE results."""
    
    @staticmethod
    def compute_ground_state_energy_error(
        vqe_energy: float,
        true_energy: float
    ) -> float:
        """
        Compute error in ground state energy.
        
        Args:
            vqe_energy: VQE computed energy
            true_energy: Exact ground state energy
            
        Returns:
            Energy error in millihartree
        """
        return abs(vqe_energy - true_energy) * 1000
    
    @staticmethod
    def plot_convergence(
        history: List[float],
        true_energy: float = None,
        save_path: str = None
    ):
        """
        Plot VQE convergence.
        
        Args:
            history: Energy history from optimization
            true_energy: Exact ground state energy for reference
            save_path: Path to save figure
        """
        iterations = range(len(history))
        
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, history, 'b-', linewidth=2, label='VQE Energy')
        
        if true_energy is not None:
            plt.axhline(y=true_energy, color='r', linestyle='--', 
                       linewidth=2, label='Exact Ground State')
            plt.fill_between(iterations, true_energy, history, 
                            alpha=0.3, where=np.array(history)>=true_energy)
        
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Energy (Ha)', fontsize=12)
        plt.title('VQE Convergence', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
        
        plt.show()
    
    @staticmethod
    def analyze_optimization(
        history: List[float],
        true_energy: float = None
    ) -> dict:
        """
        Analyze optimization results.
        
        Args:
            history: Energy history
            true_energy: Exact ground state energy
            
        Returns:
            Dictionary with analysis metrics
        """
        analysis = {
            'final_energy': history[-1],
            'min_energy': min(history),
            'max_energy': max(history),
            'n_iterations': len(history),
            'energy_range': max(history) - min(history),
        }
        
        if true_energy is not None:
            analysis['error'] = abs(history[-1] - true_energy) * 1000
            analysis['error_progress'] = (abs(history[0] - true_energy) - 
                                         abs(history[-1] - true_energy)) * 1000
        
        return analysis


class CircuitBuilder:
    """Helper class for building quantum circuits."""
    
    @staticmethod
    def create_hardware_efficient_circuit(
        n_qubits: int,
        n_layers: int,
        reps: int = 1
    ) -> 'RotationLayerAnsatz':
        """
        Create hardware-efficient ansatz circuit.
        
        Args:
            n_qubits: Number of qubits
            n_layers: Number of rotation layers
            reps: Number of repetitions
            
        Returns:
            RotationLayerAnsatz instance
        """
        from ansatz import RotationLayerAnsatz
        return RotationLayerAnsatz(n_qubits, n_layers * reps)
    
    @staticmethod
    def create_ucc_circuit(
        n_qubits: int,
        n_electrons: int,
        singles: bool = True,
        doubles: bool = False
    ) -> 'UCC AnsatzSingle':
        """
        Create UCC ansatz circuit.
        
        Args:
            n_qubits: Number of qubits
            n_electrons: Number of electrons
            singles: Include single excitations
            doubles: Include double excitations
            
        Returns:
            UCC ansatz instance
        """
        from ansatz import UCC AnsatzSingle
        excitation_level = 1 if singles else 0
        if doubles:
            excitation_level = 2
        return UCC AnsatzSingle(n_qubits, n_electrons, excitation_level)


def compute_fidelity(state1: np.ndarray, state2: np.ndarray) -> float:
    """
    Compute fidelity between two quantum states.
    Fidelity = |⟨ψ1|ψ2⟩|²
    
    Args:
        state1: First quantum state
        state2: Second quantum state
        
    Returns:
        Fidelity (0 to 1)
    """
    overlap = np.abs(np.conj(state1) @ state2)**2
    return np.real(overlap)


def measure_expectation_value(
    state: np.ndarray,
    observable: np.ndarray
) -> float:
    """
    Measure expectation value of observable.
    ⟨O⟩ = ⟨ψ|O|ψ⟩
    
    Args:
        state: Quantum state vector
        observable: Observable matrix
        
    Returns:
        Expectation value
    """
    return np.real(np.conj(state) @ observable @ state)


def normalize_state(state: np.ndarray) -> np.ndarray:
    """Normalize quantum state to unit norm."""
    norm = np.linalg.norm(state)
    if norm > 0:
        return state / norm
    return state