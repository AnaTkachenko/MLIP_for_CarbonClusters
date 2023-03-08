#EXAMPLE OF SCRIPT FOR CALCULATIONS WITH ANI_1ccx MLIP FOR C3 STOICHIOMETRY

from ase.md.langevin import Langevin
from ase.optimize import BFGS, LBFGS
from ase import units, Atoms
from ase.units import Hartree
import torchani
import time
from ase.optimize.basin import BasinHopping
from ase.optimize.minimahopping import MinimaHopping
import numpy as np
import time
import pickle

def build_A(geometry): #Function that creates an adjacency matrix from the CK generated structure
    Threshold_distance = 2.0
    N = len(geometry)
    A = np.zeros((N,N))
    for i1, atom1 in enumerate(geometry[:-1]):
        for i2, atom2 in enumerate(geometry[i1+1:]):
            if np.sqrt((atom1[0]-atom2[0])**2 + (atom1[1]-atom2[1])**2 + (atom1[2]-atom2[2])**2) < Threshold_distance:
                A[i1][i2+i1+1] = 1
                A[i2+i1+1][i1] = 1
    return A

def conn(geometry): #Function that checks if the structure is connected using the adjacency matrix
	A = build_A(geometry)
	D = np.zeros((len(geometry),len(geometry)))
	for d,row in enumerate(A):
		D[d][d] = sum(row)
	L = D - A
	e,v = np.linalg.eigh(L)
	return e[1] > 1e-6

def not_too_close(geometry): #Function that checks if any two atoms in the geometry are not too close to each other
    Threshold_distance = 0.8
    for i1, atom1 in enumerate(geometry[:-1]):
        for i2, atom2 in enumerate(geometry[i1+1:]):
            if np.sqrt((atom1[0]-atom2[0])**2 + (atom1[1]-atom2[1])**2 + (atom1[2]-atom2[2])**2) < Threshold_distance:
                return False
    return True


def ML_run(size): #Function that runs geometry optimization with MLIP for the given size
    num_files = 5*2**size
    calculator = torchani.models.ANI1ccx().ase()
    e_geom_list = []
    times = []
    for i in range(num_files):
        print("---------- ", i)
        with open('struct'+str(i)+".com", 'r') as f:
            data = f.read()
        data = data.split('\n')[8:-2]
        for i in range(len(data)):
            data[i] = data[i][1:].split()
            data[i] = list(map(float,data[i]))
        atom_name = 'C'+str(size)
        atom = Atoms(atom_name, positions=data)
        atom.set_calculator(calculator)
        opt = BFGS(atom)
        st = time.perf_counter()
        opt.run(fmax=1e-5, steps=200)
        fin=time.perf_counter() - st
        times.append(fin)
        e_geom_list.append([atom.positions, atom.get_potential_energy()/Hartree, opt.get_number_of_steps(), fin])

    e_geom_list = sorted(e_geom_list, key=lambda x: x[1])
    e_geom_list_conn = [i for i in e_geom_list if conn(i[0])] #The list of connected structures
    e_geom_list_conn_en = [i for d, i in enumerate(e_geom_list_conn) if i[1]-e_geom_list_conn[d-1][1]>1e-4 or d == 0] #The list of connected structures with distinct energies
    e_geom_list_conn_en_not_too_close = [i for i in e_geom_list_conn_en if not_too_close(i[0])] #The list of connected structures with distinct energies and without collapsed atoms
    with open(f'C{size}.pkl', 'wb') as f:
        pickle.dump([times, e_geom_list, e_geom_list_conn, e_geom_list_conn_en, e_geom_list_conn_en_not_too_close], f)

ML_run(3) #ML_run function call for C3 stoichiometry