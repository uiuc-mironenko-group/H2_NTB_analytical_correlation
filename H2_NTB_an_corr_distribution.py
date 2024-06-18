# Authors: Leung, L.; Mironenko, A. V.
# June 18, 2024 version

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import fsolve, minimize
from scipy.special import exp1 as E1
from pyscf import gto, dft
import matplotlib.pyplot as plt

S_mat = np.zeros([2,2])
D_mat = np.zeros([2,2])

def get_abba(R,Z):
    '''Compute (ab|ba) integral'''
    # McQuarrie, D.A. Quantum Chemistry (2nd ed.). Table 10.8, page 544
    w = Z*R                                              
    gamma = np.euler_gamma                                 
    S = np.exp(-w) * (1 + w + w**2/3)                      
    S_prime = np.exp(w) * (1 - w + w**2/3)                  
    abba = Z/5 * (-np.exp(-2*w)*(-25/8 + 23*w/4 + 3*w**2 + w**3/3) \
        + 6/w *(S**2 * (gamma + np.log(w)) - S_prime**2*E1(4*w) + 2*S*S_prime*E1(2*w)))
    return abba

def get_S_matrix(R,Z):
    '''Calculate overlap matrix off-diagonal elements'''
    S_ab = ((R**2*Z**2)/3 + R*Z + 1)*np.exp(-R*Z)
    S_mat[0,1] = S_ab
    S_mat[1,0] = S_ab

def get_D_matrix(R,Z):
    '''Calculate Huckel matrix'''
    D_mat[0,0] = -(Z**2)/2
    D_mat[1,1] = -(Z**2)/2
    Dij = -1/2 * get_abba(R,Z)
    D_mat[0,1] = Dij
    D_mat[1,0] = Dij

def get_e_ortho():
    '''Calculate orthogonalization (Pauli repulsion) energy'''
    e_orthogonal = - (S_mat[0,1] * D_mat[1,0] + S_mat[1,0] * D_mat[0,1])
    return e_orthogonal

def get_e_hybridization():
    '''Calculate hybridization (resonance) energy'''
    D_eigenvalues, D_eigenvectors = eigh(D_mat)
    eps = D_eigenvalues[0]
    e_AO = D_eigenvalues[1]
    e_hyb = (2 * eps) - (D_mat[0,0] + D_mat[1,1])
    return e_hyb

def get_e_electrostatics(R,Z):
    '''Calculate electrostatic energy'''
    e_NN = 1/R
    e_Ne = Z * np.exp(-2*Z*R) * (1 + (1/(Z*R))) - (1/R)
    e_ee = (1/R) - (Z * np.exp(-2*Z*R) * ((1/(Z*R)) + (11/8) + (3*Z*R/4) + (((Z*R)**2)/6)))
    e_es = e_NN + (2 * e_Ne) + e_ee
    return e_es

def get_e_c(R,Z): 
    '''Calculate dynamic correlation energy'''
    D_eigenvalues, D_eigenvectors = eigh(D_mat)
    e_BO = D_eigenvalues[0]
    e_AO = D_eigenvalues[1]
    abba = -0.5*get_abba(R,Z)
    CI_matrix = np.array([[e_BO, abba[0]],[abba[0], e_AO]])
    CI_eigenvalues, CI_eigenvectors = eigh(CI_matrix)
    e_c = CI_eigenvalues[0]-e_BO   
    return e_c

def atomion(Z,R):
    '''Solve the atomion equation to obtain self-consistent atomions'''
    get_S_matrix(R,Z)
    get_D_matrix(R,Z)
    e_Ne = Z * np.exp(-2*Z*R) * (1 + (1/(Z*R))) - (1/R)
    e_ee = 1/R - Z * np.exp(-2*Z*R) * ((1/(Z*R)) + (11/8) + (3*Z*R/4) + (((Z*R)**2)/6))  
    Daba = -1/2 * get_abba(R,Z)/np.sqrt(2)
    residual = Z**2 - Z + (e_Ne + e_ee) + Daba - S_mat[0,1] * D_mat[1,0]
    return residual

def get_singleval(R):
    '''
    Calculates H2 bond formation energy in Hartrees for a single interatomic distance
    
    Args
        R: interatomic distance in Bohr. Data type: float
    '''
    Z_val = fsolve(atomion, x0=1, args=(R))
    e_ortho = get_e_ortho()
    e_hyb = get_e_hybridization()
    e_es = get_e_electrostatics(R, Z_val)
    e_c = get_e_c(R, Z_val)

    e_tot = e_ortho + e_hyb + e_es + e_c
    return e_tot

def plot_PES(Rs):
    '''Plots H2 bond formation energy as a function of interatomic distance
    
    Args
        Rs: interatomic distances in Angstrom. Data type: ndarray

    Example
        >> rs = np.arange(start=0.3, stop=2., step=0.01)
        >> plot(rs)
    '''
    Es = np.zeros_like(Rs)

    # Convert interatomic distances from Angstrom to Bohr
    Rs_bohr = Rs*1.88973
    
    for i, R in enumerate(Rs_bohr):
        Es[i] = get_singleval(R)

    # Convert energies from Hartree to eV
    Es *= 27.2114

    plt.figure(figsize=(6,6), dpi=600)
    plt.hlines(0, Rs[0], Rs[-1])
    plt.plot(Rs, Es, linewidth=4, color='blue')
    plt.xlim((Rs[0], Rs[-1]))
    plt.xlabel('Interatomic distance (Å)')
    plt.ylabel('Bond formation energy (eV)')
    plt.show()

rs = np.arange(start=0.5, stop=3., step=0.01)
# rs = np.asarray([0.743])
plot_PES(rs)