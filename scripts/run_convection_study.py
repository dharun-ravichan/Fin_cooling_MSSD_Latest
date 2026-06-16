#Convection comparision between natural and forced convection
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import dolfinx
import dolfinx.fem
from dolfinx.fem.petsc import LinearProblem 
import ufl
from mpi4py import MPI  
import numpy as np
import matplotlib.pyplot as plt
from src.mesh_gen import generate_fin_mesh

def execute_convection_run(h_val, label):
    print(f"Simulating Convection Variant: {label} (h = {h_val})...")
    mesh, cell_tags, facet_tags = generate_fin_mesh(L=2.0, H=0.5, lc=0.025)
    V = dolfinx.fem.functionspace(mesh, ("P", 1))
    
    T_curr = dolfinx.fem.Function(V)
    T_curr.interpolate(lambda x: np.full(x.shape[1], 20.0))
    
    k = dolfinx.fem.Constant(mesh, 400.0)
    rho_cp = dolfinx.fem.Constant(mesh, 3.5e6)
    h_param = dolfinx.fem.Constant(mesh, h_val)
    T_inf = dolfinx.fem.Constant(mesh, 20.0)
    dt, t_end = 0.1, 10.0
    
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags)
    dx = ufl.Measure("dx", domain=mesh)
    
    a = (rho_cp * u / dt) * v * dx + k * ufl.dot(ufl.grad(u), ufl.grad(v)) * dx + h_param * u * v * ds(2)
    L = (rho_cp * T_curr / dt) * v * dx + h_param * T_inf * v * ds(2)
    
    bc_base = dolfinx.fem.dirichletbc(dolfinx.fem.Constant(mesh, 100.0), 
                                      dolfinx.fem.locate_dofs_topological(V, mesh.topology.dim - 1, facet_tags.find(1)), V)
    
    problem = LinearProblem(
        a, L, bcs=[bc_base], 
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        petsc_options_prefix=f"fin_study_{int(h_val)}_"
    )
    
    time_steps = []
    q_out_history = []
    
    t = 0.0
    while t < t_end:
        t += dt
        
        res = problem.solve()
        T_curr.x.array[:] = res.x.array[:]
        
        # Calculate designated Quantity of Interest (Surface Dissipation Flux)
        loss_form = dolfinx.fem.form(h_val * (T_curr - 20.0) * ds(2))
        
        q_out = mesh.comm.allreduce(dolfinx.fem.assemble_scalar(loss_form), op=MPI.SUM)
        
        time_steps.append(t)
        q_out_history.append(q_out)
        
    return time_steps, q_out_history

t_steps, q_natural = execute_convection_run(10.0, "Natural Ambient Air")
_, q_forced = execute_convection_run(100.0, "Forced Fan Exhaust")

print("\n" + "="*60)
print("REQUIRED STUDY PERFORMANCE METRICS (t = 10.0s)")
print("="*60)
print(f"Variant 1 (Natural Air, h=10)  Total Dissipation Rate (QoI): {q_natural[-1]:.3f} W/m")
print(f"Variant 2 (Forced Fan,  h=100) Total Dissipation Rate (QoI): {q_forced[-1]:.3f} W/m")
print("="*60)

plt.figure(figsize=(8, 4.5), dpi=200)
plt.plot(t_steps, q_natural, color='orange', linewidth=2.5, label="Variant 1: Natural Convection ($h=10$)")
plt.plot(t_steps, q_forced, color='teal', linewidth=2.5, label="Variant 2: Forced Fan Cooling ($h=100$)")
plt.title("Required Comparative Project Study: Impact of Airflow Environment", fontsize=11, fontweight='bold')
plt.xlabel("Time (seconds)", fontsize=10)
plt.ylabel("Quantity of Interest: Integrated Heat Dissipated $Q_{out}$ (W/m)", fontsize=10)
plt.grid(True, linestyle='--', alpha=0.4)
plt.legend()
plt.savefig("convection_study_comparison.png", bbox_inches='tight')
plt.close()
print("\nSUCCESS: Comparative plot generated and saved as 'convection_study_comparison.png'")
