#Generates a schematic of the physical setup, later used in the report
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 4))


# Fin body
rect = plt.Rectangle(
    (0, 0),
    2.0,
    0.5,
    fill=False,
    linewidth=3
)

ax.add_patch(rect)

# Tag 1 — Base (left edge, Dirichlet)
ax.plot(
    [0, 0],
    [0, 0.5],
    color="tab:red",
    linewidth=5
)

ax.text(
    -0.25,
    0.25,
    "$\\Gamma_{\\mathrm{Base}}$ (Tag 1)\nDirichlet: $T = 100^\\circ$C",
    ha="right",
    va="center",
    fontsize=10,
    color="tab:red"
)


# Tag 2 — Convective (top and bottom edges)

# Top convection arrows (pointing into fin from above)
for x in [0.2, 0.6, 1.0, 1.4, 1.8]:
    ax.arrow(
        x,
        0.72,
        0,
        -0.14,
        head_width=0.03,
        length_includes_head=True,
        color="tab:blue"
    )

# Bottom convection arrows (pointing into fin from below)
for x in [0.2, 0.6, 1.0, 1.4, 1.8]:
    ax.arrow(
        x,
        -0.22,
        0,
        0.14,
        head_width=0.03,
        length_includes_head=True,
        color="tab:blue"
    )

ax.text(
    1.0,
    0.92,
    "$\\Gamma_{\\mathrm{Convective}}$ (Tag 2)\n"
    "$-k\\nabla T \\cdot n = h(T - T_\\infty),\\quad T_\\infty = 20^\\circ$C",
    ha="center",
    va="bottom",
    fontsize=10,
    color="tab:blue"
)

# Tag 3 — Adiabatic_Tip (right edge)
ax.plot(
    [2.0, 2.0],
    [0, 0.5],
    color="tab:green",
    linewidth=5
)

ax.text(
    2.25,
    0.25,
    "$\\Gamma_{\\mathrm{Adiabatic\\_Tip}}$ (Tag 3)\n$-k\\nabla T \\cdot n = 0$",
    ha="left",
    va="center",
    fontsize=10,
    color="tab:green"
)


# Fin interior label
ax.text(
    1.0,
    0.25,
    "Copper Fin\n$k = 400$ W/(m$\\,$K)"
    "\n$\\rho c_p = 3.5 \\times 10^6$ J/(m$^3$K)",
    ha="center",
    va="center",
    fontsize=11
)


# Axes
ax.set_xlim(-1.1, 3.2)
ax.set_ylim(-0.5, 1.1)

ax.set_aspect("equal")
ax.axis("off")

plt.tight_layout()

plt.savefig(
    "physical_setup.png",
    dpi=300,
    bbox_inches="tight"
)

print("Saved physical_setup.png")
