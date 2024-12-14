""" Infinite DMRG algorithm for the 1D Heisenberg model. """

import numpy as np
from numpy import kron

I = np.eye(2)
Sz = np.array([[0.5, 0], [0, -0.5]])
Sp = np.array([[0, 1], [0, 0]])
Sm = np.array([[0, 0], [1, 0]])

E = -0.75
Exact_E = -0.4431471805599453  # -log(2) + 0.25


class Block:
    def __init__(self, blockH, blockI, blockSz, blockSp, blockSm):
        self.H = blockH
        self.I = blockI
        self.Sz = blockSz
        self.Sp = blockSp
        self.Sm = blockSm


def construct_block_site_Hamiltonian(block: Block):
    block.H = kron(block.H, I)
    block.H += (J / 2 * (kron(block.Sp, Sm) + kron(block.Sm, Sp)) + 
                Jz * kron(block.Sz, Sz) - h * kron(block.I, Sz))

    block.Sz = kron(block.I, Sz)
    block.Sp = kron(block.I, Sp)
    block.Sm = kron(block.I, Sm)
    block.I = kron(block.I, I)

    return block


def construct_superblock_Hamiltonian(block: Block):
    superblockH = kron(block.H, block.I) + kron(block.I, block.H)
    superblockH += (J / 2 * (kron(block.Sp, block.Sm) + kron(block.Sm, block.Sp)) + 
                    Jz * kron(block.Sz, block.Sz))
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


def construct_density_matrix(Psis):
    Psi0 = Psis[:, 0]
    Dim = int(np.sqrt(np.size(Psi0)))
    PsiM = Psi0.reshape(Dim, Dim)
    rho = np.dot(PsiM.T, PsiM)
    return rho


def truncate(block: Block, D, V, m):
    keep = int(min(np.size(D), m))
    O = V[:, :keep]
    Dk = D[:keep]

    block.H = np.dot(O.T, np.dot(block.H, O))
    block.I = np.dot(O.T, np.dot(block.I, O))
    block.Sz = np.dot(O.T, np.dot(block.Sz, O))
    block.Sp = np.dot(O.T, np.dot(block.Sp, O))
    block.Sm = np.dot(O.T, np.dot(block.Sm, O))

    return block, Dk


def print_results(i, Dk, Es, lastE):
    E0 = Es[0]
    Eperband = (E0 - lastE) / 2
    print(f'Iteration {i}: Energy = {E0:.10f}, Energy per band = {Eperband:.10f}, '
          f'Error = {Exact_E - Eperband:.10f}, Truncation error = {1 - np.sum(Dk):.10f}, '
          f'Number of states kept = {len(Dk)}')
    return E0


def infinite_dmrg():
    block = Block(blockH=-h * Sz, blockI=I, blockSz=Sz, blockSp=Sp, blockSm=Sm)
    lastE = E
    for i in range(L - 1):
        block = construct_block_site_Hamiltonian(block)
        superblockH = construct_superblock_Hamiltonian(block)
        Es, Psis = diagonalize_sort(superblockH, sort=1)
        rho = construct_density_matrix(Psis)
        D, V = diagonalize_sort(rho, sort=-1)
        block, Dk = truncate(block, D, V, m)
        lastE = print_results(i, Dk, Es, lastE)


if __name__ == '__main__':
    J = 1
    Jz = 1
    h = 0
    L = 40
    m = 12

    infinite_dmrg()
