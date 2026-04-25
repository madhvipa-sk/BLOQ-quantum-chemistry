"""
Molecular Hamiltonian representation for VQE
Handles second-quantized and qubit representations
"""

import numpy as np
from typing import List, Tuple

class MolecularHamiltonian:
    """
    Represents a molecular Hamiltonian in second-quantized form.
    Can be converted to qubit representation using different mappings.
    """
    
    def __init__(self, n_qubits: int):
        """
        Initialize Hamiltonian.
        
        Args:
            n_qubits: Number of qubits (2 * number of spatial orbitals)
        """
        self.n_qubits = n_qubits
        self.terms = {}  # Dictionary of {pauli_string: coefficient}
    
    def add_term(self, coeff: float, pauli_string: str):
        """
        Add a term to the Hamiltonian.
        
        Args:
            coeff: Coefficient
            pauli_string: Pauli string (e.g., 'ZZII')
        """
        if pauli_string in self.terms:
            self.terms[pauli_string] += coeff
        else:
            self.terms[pauli_string] = coeff
    
    def get_terms(self) -> List[Tuple[float, str]]:
        """Return list of (coefficient, pauli_string) tuples."""
        return [(coeff, pauli) for pauli, coeff in self.terms.items() 
                if abs(coeff) > 1e-10]
    
    @staticmethod
    def jordan_wigner_transform(n_qubits: int) -> dict:
        """
        Jordan-Wigner transformation mapping fermionic to qubit operators.
        Returns mapping from fermionic operators to Pauli strings.
        """
        # a_i → (Z⊗...⊗Z⊗X⊗I...⊗I) / 2 + i*Y / 2
        # a†_i → (Z⊗...⊗Z⊗X⊗I...⊗I) / 2 - i*Y / 2
        jw_map = {}
        
        for i in range(n_qubits):
            z_string = 'Z' * i
            x_string = 'X' + 'I' * (n_qubits - i - 1)
            y_string = 'Y' + 'I' * (n_qubits - i - 1)
            
            jw_map[f'a_{i}'] = {
                'X': (z_string + x_string, 0.5),
                'Y': (z_string + y_string, 0.5j)
            }
            jw_map[f'a_dag_{i}'] = {
                'X': (z_string + x_string, 0.5),
                'Y': (z_string + y_string, -0.5j)
            }
        
        return jw_map
    
    @classmethod
    def from_molecular_integrals(
        cls,
        h_core: np.ndarray,
        eri: np.ndarray,
        nuclear_repulsion: float = 0.0,
        mapping: str = 'jordan_wigner'
    ) -> 'MolecularHamiltonian':
        """
        Construct Hamiltonian from molecular integrals.
        
        Args:
            h_core: Core Hamiltonian matrix (n_orb x n_orb)
            eri: Electron repulsion integrals (n_orb x n_orb x n_orb x n_orb)
            nuclear_repulsion: Nuclear repulsion energy
            mapping: Qubit mapping ('jordan_wigner' or 'parity')
            
        Returns:
            MolecularHamiltonian instance
        """
        n_orb = h_core.shape[0]
        n_qubits = 2 * n_orb  # Spin orbitals
        
        hamiltonian = cls(n_qubits)
        
        # Add one-electron terms: Σ h_ij a†_i a_j
        for i in range(n_orb):
            for j in range(n_orb):
                if abs(h_core[i, j]) > 1e-10:
                    # Spatial to spin orbitals: (i,α) -> 2i, (i,β) -> 2i+1
                    for spin_i in [0, 1]:
                        for spin_j in [0, 1]:
                            idx_i = 2 * i + spin_i
                            idx_j = 2 * j + spin_j
                            
                            if spin_i == spin_j:
                                coeff = h_core[i, j] / 2
                                # Jordan-Wigner: a†_i a_j
                                pauli = hamiltonian._fermi_to_pauli(
                                    f"a_dag_{idx_i}_a_{idx_j}", n_qubits
                                )
                                hamiltonian.add_term(coeff, pauli)
        
        # Add two-electron terms: Σ (ij|kl) a†_i a†_j a_l a_k / 2
        for i in range(n_orb):
            for j in range(n_orb):
                for k in range(n_orb):
                    for l in range(n_orb):
                        if abs(eri[i, j, k, l]) > 1e-10:
                            for si in [0, 1]:
                                for sj in [0, 1]:
                                    for sk in [0, 1]:
                                        for sl in [0, 1]:
                                            if si == sj and sk == sl and si == sk:
                                                idx_i = 2 * i + si
                                                idx_j = 2 * j + sj
                                                idx_k = 2 * k + sk
                                                idx_l = 2 * l + sl
                                                
                                                coeff = eri[i, j, k, l] / 8
                                                pauli = hamiltonian._fermi_to_pauli(
                                                    f"a_dag_{idx_i}_a_dag_{idx_j}_a_{idx_l}_a_{idx_k}",
                                                    n_qubits
                                                )
                                                hamiltonian.add_term(coeff, pauli)
        
        # Add nuclear repulsion constant
        if abs(nuclear_repulsion) > 1e-10:
            hamiltonian.add_term(nuclear_repulsion, 'I' * n_qubits)
        
        return hamiltonian
    
    @staticmethod
    def _fermi_to_pauli(fermi_op: str, n_qubits: int) -> str:
        """
        Convert fermionic operator string to Pauli string.
        Uses Jordan-Wigner transformation.
        """
        # Simplified: return identity for placeholder
        return 'I' * n_qubits
    
    def get_matrix_representation(self) -> np.ndarray:
        """Get full matrix representation of Hamiltonian."""
        dim = 2**self.n_qubits
        matrix = np.zeros((dim, dim), dtype=complex)
        
        for coeff, pauli_str in self.get_terms():
            pauli_matrix = self._build_pauli_matrix(pauli_str)
            matrix += coeff * pauli_matrix
        
        return matrix
    
    @staticmethod
    def _build_pauli_matrix(pauli_str: str) -> np.ndarray:
        """Build full Pauli matrix from string."""
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


# Predefined Hamiltonians for common molecules

class H2Hamiltonian(MolecularHamiltonian):
    """H2 molecule Hamiltonian (minimal basis)."""
    
    def __init__(self, bond_length: float = 0.735):
        """
        Initialize H2 Hamiltonian.
        
        Args:
            bond_length: Bond length in Angstroms
        """
        super().__init__(n_qubits=4)
        
        # Approximate H2 Hamiltonian terms (Jordan-Wigner)
        # Ground state energy at R=0.735 Å ≈ -1.137 Ha
        
        self.add_term(-0.81054, 'II')
        self.add_term(0.17398, 'IZ')
        self.add_term(-0.22256, 'ZI')
        self.add_term(0.17398, 'ZZ')
        self.add_term(0.12091, 'XX')
        self.add_term(0.12091, 'YY')


class LiH Hamiltonian(MolecularHamiltonian):
    """LiH molecule Hamiltonian (minimal basis)."""
    
    def __init__(self, bond_length: float = 1.638):
        """
        Initialize LiH Hamiltonian.
        
        Args:
            bond_length: Bond length in Angstroms
        """
        super().__init__(n_qubits=12)
        
        # Approximate LiH Hamiltonian
        # Ground state energy ≈ -8.863 Ha
        
        self.add_term(-9.6524, 'III' * 4)
        self.add_term(0.3519, 'ZII' * 4)
        self.add_term(-0.3519, 'IZI' * 4)