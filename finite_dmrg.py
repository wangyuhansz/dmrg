""" Infinite and finite DMRG algorithm for the 1D Heisenberg model. """

import copy
import numpy as np
from numpy import kron

I = np.eye(2)
Sz = np.array([[0.5, 0], [0, -0.5]])
Sp = np.array([[0, 1], [0, 0]])
Sm = np.array([[0, 0], [1, 0]])

Exact_E = -0.4431471805599453  # -log(2) + 0.25


class Block:
    def __init__(self, blockH, blockI, blockSz, blockSp, blockSm, 
                 blockM=2, blockL=1, blockE=-0.75, blockDk=None):
        self.H = blockH
        self.I = blockI
        self.Sz = blockSz
        self.Sp = blockSp
        self.Sm = blockSm

        self.M = blockM  # Number of states kept in the block
        self.L = blockL  # Number of sites in the block

        self.E = blockE  # Initial energy of the block

        self.Dk = blockDk  # Eigenvalues of the reduced density matrix

        self.history = {
            self.L: {
                'H': blockH,
                'I': blockI,
                'Sz': blockSz,
                'Sp': blockSp,
                'Sm': blockSm,
                'M': 2
            }
        }
    
    def save_state(self):
        self.history[self.L] = {
            'H': self.H,
            'I': self.I,
            'Sz': self.Sz,
            'Sp': self.Sp,
            'Sm': self.Sm,
            'M': self.M
        }
    
    def get_history(self, L):
        return self.history.get(L, None)


def construct_block_site_Hamiltonian(block: Block):
    """ Construct the block-site Hamiltonian from the block. """
    block.L += 1  # Add a site to the block
    block.M *= 2  # Double the Hilbert space of the block

    block.H = kron(block.H, I)
    block.H += (J / 2 * (kron(block.Sp, Sm) + kron(block.Sm, Sp)) + 
                Jz * kron(block.Sz, Sz) - h * kron(block.I, Sz))

    block.Sz = kron(block.I, Sz)
    block.Sp = kron(block.I, Sp)
    block.Sm = kron(block.I, Sm)
    block.I = kron(block.I, I)

    block.save_state()

    return block


def construct_superblock_Hamiltonian(blockL: Block, blockR: Block=None):
    """ Construct the superblock Hamiltonian from the left and right blocks. 
        If the right block is not provided, the superblock Hamiltonian will
        be constructed from the left block and its "reflection".
    """
    blockR = blockL if blockR is None else blockR

    superblockH = kron(blockL.H, blockR.I) + kron(blockL.I, blockR.H)
    superblockH += (J / 2 * (kron(blockL.Sp, blockR.Sm) + 
                             kron(blockL.Sm, blockR.Sp)) + 
                    Jz * kron(blockL.Sz, blockR.Sz))
    
    return superblockH


def diagonalize_sort(matrix, sort=1):
    """ Diagonalize the matrix and sort the eigenvalues and eigenvectors. 
    Args:
        matrix: The matrix to be diagonalized.
        sort: The order of the eigenvalues and eigenvectors. 
            1: from small to large
           -1: from large to small
    """
    D, V = np.linalg.eigh(matrix)
    indices = np.argsort(sort * D)
    D, V = D[indices], V[:, indices]
    return D, V


def construct_reduced_density_matrix(Psis: np.ndarray, L, R=None):
    R = L if R is None else R
    Psi0 = Psis[:, 0]
    PsiM = Psi0.reshape(L, R)
    rho_reduced = np.dot(PsiM, PsiM.conj().T)
    return rho_reduced


def truncate(block: Block, D, V):
    keep = min(block.M, M)

    block.M = keep
    block.Dk = np.sum(D[:keep])
    O = V[:, :keep]

    block.H = O.T @ block.H @ O
    block.I = O.T @ block.I @ O
    block.Sz = O.T @ block.Sz @ O
    block.Sp = O.T @ block.Sp @ O
    block.Sm = O.T @ block.Sm @ O

    return block


def print_results(algorithm: str, i, block: Block, Es):
    E0 = Es[0]

    if algorithm == 'Infinite DMRG':
        Eperband = (E0 - block.E) / 2
        print(f'{algorithm}',
              f'Iteration {i}: Energy = {E0:.9f}, '
              f'Energy per band = {Eperband:.9f}, '
              f'Error = {Exact_E - Eperband:.5e}, '
              f'Truncation error = {1 - block.Dk:.5e}, '
              f'Number of states kept = {block.M}')
    elif algorithm == 'Finite DMRG':
        E_diff = abs(E0 - block.E)
        print(f'{algorithm}',
              f'Iteration {i}: Energy = {E0:.9f}, '
              f'Error = {E_diff:.5e}, '
              f'Truncation error = {1 - block.Dk:.5e}, '
              f'Number of states kept = {block.M}')
    
    block.E = E0
    return block


def infinite_dmrg() -> Block:
    """ Perform infinite DMRG on the 1D Heisenberg model."""
    block = Block(blockH=-h * Sz, blockI=I, blockSz=Sz, blockSp=Sp, blockSm=Sm)

    for i in range(L - 1):
        block = construct_block_site_Hamiltonian(block)
        superblockH = construct_superblock_Hamiltonian(block)
        Es, Psis = diagonalize_sort(superblockH, sort=1)
        rho_reduced = construct_reduced_density_matrix(Psis, L=block.M)
        D, V = diagonalize_sort(rho_reduced, sort=-1)
        block = truncate(block, D, V)
        block = print_results('Infinite DMRG', i, block, Es)

    return block


def finite_dmrg(block: Block):
    """ Perform finite DMRG on the result block from the infinite DMRG. """
    def shrink_block_Hamiltonian(block: Block):
        """ Shrink the block by removing the last site. Use the history to 
            restore the previous state of the block.
        """
        block.L -= 1
        state_dict = block.get_history(block.L)
        block.H = state_dict['H']
        block.I = state_dict['I']
        block.Sz = state_dict['Sz']
        block.Sp = state_dict['Sp']
        block.Sm = state_dict['Sm']
        block.M = state_dict['M']
        return block
    
    def direction_reverser(blockL: Block, blockR: Block):
        """ Reverse the direction of the block if the number of states kept 
            in the right block is less than the desired number of states.
        """
        if 2 ** blockR.L < M:
            print('Finite DMRG direction reversed.')
            return blockR, blockL
        else:
            return blockL, blockR
    
    blockL, blockR = copy.deepcopy(block), copy.deepcopy(block)

    for i in range(N):
        blockL = construct_block_site_Hamiltonian(blockL)
        blockR = shrink_block_Hamiltonian(blockR)
        superblockH = construct_superblock_Hamiltonian(blockL, blockR)
        Es, Psis = diagonalize_sort(superblockH, sort=1)
        rho_reduced = construct_reduced_density_matrix(Psis, blockL.M, blockR.M)
        D, V = diagonalize_sort(rho_reduced, sort=-1)
        blockL = truncate(blockL, D, V)
        blockL = print_results('Finite DMRG', i, blockL, Es)
        blockL, blockR = direction_reverser(blockL, blockR)


if __name__ == '__main__':
    J = 1
    Jz = 1
    h = 0
    L = 40  # Length of half of the chain
    M = 12  # Number of states kept
    N = 200  # Iterations of the finite DMRG

    block = infinite_dmrg()
    finite_dmrg(block)
