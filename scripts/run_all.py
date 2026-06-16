#Comprehensive run file that generates figures and comparisions from different runs
import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

import numpy as np
import matplotlib.pyplot as plt

import dolfinx
import dolfinx.plot

from src.mesh_gen import generate_fin_mesh
from src.solver import solve_transient_fin
from src.verification import solve_transient_fin_mms



# OUTPUT FOLDER
FIG_DIR = "../report_figures"
os.makedirs(FIG_DIR, exist_ok=True)

print("=" * 80)
print("GENERATING REPORT FIGURES")
print("=" * 80)


# FIGURE 1
# MESH
print("\n[1/6] Generating mesh figure...")

mesh, cell_tags, facet_tags = generate_fin_mesh(
    L=2.0,
    H=0.5,
    lc=0.05
)

topology, cell_types, x = dolfinx.plot.vtk_mesh(
    mesh,
    mesh.topology.dim
)

num_cells = mesh.topology.index_map(
    mesh.topology.dim
).size_local

triangles = topology.reshape(
    num_cells,
    4
)[:, 1:4]

plt.figure(figsize=(12, 4))

plt.triplot(
    x[:, 0],
    x[:, 1],
    triangles,
    linewidth=0.25
)

plt.axis("equal")

plt.title(
    "Finite Element Mesh and Boundary Tags",
    pad=20
)

# Tag 1 — Base (left edge, Dirichlet)
plt.annotate(
    "Tag 1: Base\n(Dirichlet, T = 100°C)",
    xy=(0, 0.25),
    xytext=(-0.55, 0.25),
    fontsize=8,
    arrowprops=dict(arrowstyle="->")
)

# Tag 2 — Convective (top edge)
plt.annotate(
    "Tag 2: Convective\n(Robin)",
    xy=(1.0, 0.5),
    xytext=(0.75, 0.72),
    fontsize=8,
    arrowprops=dict(arrowstyle="->")
)

# Tag 2 — Convective (bottom edge)
plt.annotate(
    "Tag 2: Convective\n(Robin)",
    xy=(1.0, 0.0),
    xytext=(0.75, -0.28),
    fontsize=8,
    arrowprops=dict(arrowstyle="->")
)

# Tag 3 — Adiabatic_Tip (right edge)
plt.annotate(
    "Tag 3: Adiabatic_Tip\n(Zero flux)",
    xy=(2.0, 0.25),
    xytext=(2.1, 0.25),
    fontsize=8,
    arrowprops=dict(arrowstyle="->")
)

plt.tight_layout()

plt.savefig(
    f"{FIG_DIR}/mesh.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved mesh.png")



# FIGURE 2 & 3
# TEMPERATURE CONTOURs
print("\n[2/6] Generating temperature contours...")

mesh, cell_tags, facet_tags = generate_fin_mesh(
    L=2.0,
    H=0.5,
    lc=0.025
)

cases = [
    (5.0, "natural"),
    (250.0, "forced")
]

for h_coeff, label in cases:

    T_final, _, _ = solve_transient_fin(
        mesh,
        facet_tags,
        h=h_coeff,
        t_end=3600,
        dt=1
    )

    topology, _, coords = dolfinx.plot.vtk_mesh(
        mesh,
        mesh.topology.dim
    )

    num_cells = mesh.topology.index_map(
        mesh.topology.dim
    ).size_local

    triangles = topology.reshape(
        num_cells,
        4
    )[:, 1:4]

    plt.figure(figsize=(10, 3))

    contour = plt.tripcolor(
        coords[:, 0],
        coords[:, 1],
        triangles,
        T_final.x.array,
        shading="gouraud"
    )

    plt.colorbar(
        contour,
        label="Temperature (°C)"
    )

    plt.axis("equal")

    plt.title(
        f"{label.capitalize()} Convection Temperature Field "
        f"(h = {h_coeff} W/m²K, t = 3600 s)"
    )

    plt.savefig(
        f"{FIG_DIR}/temperature_{label}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved temperature_{label}.png")



# FIGURE 4
# QOI COMPARISON
==========================================================

print("\n[3/6] Computing QoI comparison...")

mesh, cell_tags, facet_tags = generate_fin_mesh(
    L=2.0,
    H=0.5,
    lc=0.025
)

_, t_nat, q_nat = solve_transient_fin(
    mesh,
    facet_tags,
    h=5.0,
    t_end=3600,
    dt=1
)

_, t_forced, q_forced = solve_transient_fin(
    mesh,
    facet_tags,
    h=250.0,
    t_end=3600,
    dt=1
)

E_nat = np.trapezoid(q_nat, t_nat)
E_forced = np.trapezoid(q_forced, t_forced)

print()
print(f"Natural Convection Energy : {E_nat:.3f} J/m")
print(f"Forced  Convection Energy : {E_forced:.3f} J/m")
print(f"Improvement Factor        : {E_forced/E_nat:.3f}")

with open(f"{FIG_DIR}/qoi_summary.txt", "w") as f:
    f.write(f"Natural Convection Energy : {E_nat:.6f} J/m\n")
    f.write(f"Forced Convection Energy  : {E_forced:.6f} J/m\n")
    f.write(f"Improvement Factor        : {E_forced/E_nat:.6f}\n")

plt.figure(figsize=(8, 5))

plt.plot(
    t_nat,
    q_nat,
    linewidth=2,
    label="Natural Convection (h = 5 W/m²K)"
)

plt.plot(
    t_forced,
    q_forced,
    linewidth=2,
    label="Forced Convection (h = 250 W/m²K)"
)

plt.xlabel("Time (s)")
plt.ylabel("Heat Dissipation Rate (W/m)")

plt.grid(True)
plt.legend()

plt.title("Heat Dissipation Rate Comparison")

plt.savefig(
    f"{FIG_DIR}/qoi_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved qoi_comparison.png")



# FIGURE 5 & 6
# L2 AND H1 CONVERGENCE WITH REFERENCE SLOPES

print("\n[4/6] Running MMS convergence study...")

lc_values = [0.1, 0.05, 0.025, 0.0125]

hvals = []
l2vals = []
h1vals = []

for lc in lc_values:

    mesh, cell_tags, facet_tags = generate_fin_mesh(
        L=2.0,
        H=0.5,
        lc=lc
    )

    ncells = mesh.topology.index_map(
        mesh.topology.dim
    ).size_global

    h = 1.0 / np.sqrt(ncells)

    L2, H1 = solve_transient_fin_mms(
        mesh,
        facet_tags,
        t_end=0.2,
        dt=0.00025
    )

    hvals.append(h)
    l2vals.append(L2)
    h1vals.append(H1)

    print(
        f"lc={lc:.5f}  h={h:.6e}  "
        f"L2={L2:.6e}  H1={H1:.6e}"
    )

hvals = np.array(hvals)
l2vals = np.array(l2vals)
h1vals = np.array(h1vals)


# L2 convergence plot 
plt.figure(figsize=(6, 5))

plt.loglog(hvals, l2vals, "o-", linewidth=2, label="$L^2$ error")

# Reference line anchored to the first data point
ref_l2 = l2vals[0] * (hvals / hvals[0]) ** 2
plt.loglog(hvals, ref_l2, "--", color="gray", linewidth=1, label=r"$O(h^2)$ reference")

plt.xlabel("Mesh Size $h$")
plt.ylabel("$L^2$ Error")
plt.title("$L^2$ Convergence")
plt.legend()
plt.grid(True, which="both")

plt.savefig(
    f"{FIG_DIR}/l2_convergence.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved l2_convergence.png")


# H1 convergence plot (expected slope: O(h^1))
plt.figure(figsize=(6, 5))

plt.loglog(hvals, h1vals, "o-", linewidth=2, label="$H^1$ seminorm")

# Reference line anchored to the first data point
ref_h1 = h1vals[0] * (hvals / hvals[0]) ** 1
plt.loglog(hvals, ref_h1, "--", color="gray", linewidth=1, label=r"$O(h)$ reference")

plt.xlabel("Mesh Size $h$")
plt.ylabel("$H^1$ Seminorm Error")
plt.title("$H^1$ Convergence")
plt.legend()
plt.grid(True, which="both")

plt.savefig(
    f"{FIG_DIR}/h1_convergence.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved h1_convergence.png")


print("\n" + "=" * 80)
print("ALL FIGURES GENERATED")
print("=" * 80)

print("\nGenerated files:")
print("  mesh.png")
print("  temperature_natural.png")
print("  temperature_forced.png")
print("  qoi_comparison.png")
print("  l2_convergence.png")
print("  h1_convergence.png")
print("  qoi_summary.txt")
