import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

from src.mesh_gen import generate_fin_mesh
from src.verification import solve_transient_fin_mms



# Mesh Refinement Levels

lc_values = [0.1, 0.05, 0.025, 0.0125]

h_values = []
l2_errors = []
h1_errors = []

print("\n")
print("=" * 90)
print("TRANSIENT MMS VERIFICATION STUDY")
print("=" * 90)
print(
    f"{'lc':<12}"
    f"{'h':<15}"
    f"{'L2 Error':<20}"
    f"{'H1 Error':<20}"
)
print("-" * 90)



# Solve on each mesh
for lc in lc_values:

    mesh, cell_tags, facet_tags = generate_fin_mesh(
        L=2.0,
        H=0.5,
        lc=lc
    )

    num_cells = mesh.topology.index_map(
        mesh.topology.dim
    ).size_global

    h = 1.0 / np.sqrt(num_cells)

    l2_err, h1_err = solve_transient_fin_mms(
        mesh,
        facet_tags,
        t_end=0.2,
        dt=0.00025
    )

    h_values.append(h)
    l2_errors.append(l2_err)
    h1_errors.append(h1_err)

    print(
        f"{lc:<12.5f}"
        f"{h:<15.6e}"
        f"{l2_err:<20.6e}"
        f"{h1_err:<20.6e}"
    )



# Convergence Rates
l2_rates = []
h1_rates = []

for i in range(len(h_values) - 1):

    l2_rate = (
        np.log(l2_errors[i + 1] / l2_errors[i])
        /
        np.log(h_values[i + 1] / h_values[i])
    )

    h1_rate = (
        np.log(h1_errors[i + 1] / h1_errors[i])
        /
        np.log(h_values[i + 1] / h_values[i])
    )

    l2_rates.append(l2_rate)
    h1_rates.append(h1_rate)


print("\n")
print("=" * 90)
print("EXPERIMENTAL CONVERGENCE RATES")
print("=" * 90)

for i in range(len(l2_rates)):

    print(
        f"Refinement {i+1}: "
        f"L2 Rate = {l2_rates[i]:.3f}    "
        f"H1 Rate = {h1_rates[i]:.3f}"
    )


print("\n")
print("=" * 90)
print("EXPECTED THEORETICAL RATES FOR P1 ELEMENTS")
print("=" * 90)
print("L2  Error : O(h^2)")
print("H1  Error : O(h^1)")
print("=" * 90)

import json, pathlib

pathlib.Path("results").mkdir(exist_ok=True)

data = {
    "h_values":   h_values,
    "l2_errors":  l2_errors,
    "h1_errors":  h1_errors,
    "l2_rates":   l2_rates,
    "h1_rates":   h1_rates,
}

with open("results/convergence_data.json", "w") as f:
    json.dump(data, f, indent=2)

print("\nSaved to results/convergence_data.json")
