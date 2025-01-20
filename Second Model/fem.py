#%%

import dolfinx
from mpi4py import MPI
import numpy as np

import ufl
from dolfinx import fem, io, mesh, plot
# from dolfinx.fem.petsc import LinearProblem
from ufl import ds, dx, grad, inner

import pyvista
try:
    pyvista.start_xvfb()
except OSError:
    pass

#%%

msh = dolfinx.mesh.create_unit_interval(
    comm = MPI.COMM_WORLD,
    nx = 100
)
V = fem.functionspace(msh, ("Lagrange", 1))

# print nodes
# print(V.tabulate_dof_coordinates())

#%%

