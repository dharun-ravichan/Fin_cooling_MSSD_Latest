import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import dolfinx
import ufl
import numpy as np
import matplotlib.pyplot as plt
from src.mesh_gen import generate_fin_mesh
from src.solver import solve_transient_fin

print("Executing baseline fine mesh physical simulation (lc=0.025)...")
mesh, cell_tags, facet_tags = generate_fin_mesh(L=2.0, H=0.5, lc=0.025)
T_final, t_hist, q_hist = solve_transient_fin(
    mesh,
    facet_tags,
    h=5.0,
    t_end=5000.0,
    dt=5
)

# Extract spatial fields for 2D contour generation
plt.figure(figsize=(10, 3), dpi=300)
import dolfinx.plot
topology, cell_types, x_vtk = dolfinx.plot.vtk_mesh(mesh, mesh.topology.dim)
num_cells = mesh.topology.index_map(mesh.topology.dim).size_local
triangles = topology.reshape(num_cells, 4)[:, 1:4]

cntr = plt.tripcolor(x_vtk[:, 0], x_vtk[:, 1], triangles, T_final.x.array, cmap='jet')
plt.colorbar(cntr, label="Temperature (°C)")
plt.title("Exercise 4: Steady-State Temperature Distribution Field (t = 10s)", fontsize=11, fontweight='bold')
plt.xlabel("X Coordinate (m)")
plt.ylabel("Y Coordinate (m)")
plt.axis('equal')
plt.savefig("report_temperature_contour.png", bbox_inches='tight')
plt.close()
print("Minimum temperature:", np.min(T_final.x.array))
print("Maximum temperature:", np.max(T_final.x.array))
