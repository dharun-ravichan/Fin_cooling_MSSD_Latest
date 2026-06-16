import dolfinx
import ufl
import numpy as np

from mpi4py import MPI
from dolfinx.fem.petsc import LinearProblem


def solve_transient_fin(
    mesh,
    facet_tags,
    h=8.0,
    t_end=10.0,
    dt=0.1
):
    """
    Solves transient heat conduction in a 2D copper cooling fin.

    Parameters:
    mesh : dolfinx.mesh.Mesh
        Computational mesh.

    facet_tags : dolfinx.mesh.MeshTags
        Boundary markers.

    h : float
        Convective heat transfer coefficient.

    t_end : float
        Final simulation time.

    dt : float
        Time step.

    Returns:
    T_curr : dolfinx.fem.Function
        Final temperature field.

    time_history : list
        Simulation times.

    q_history : list
        Total heat dissipation history.
    """

    # Function Space
    V = dolfinx.fem.functionspace(mesh, ("P", 1))

    # Material Properties
    k = dolfinx.fem.Constant(mesh, 400.0)
    rho_cp = dolfinx.fem.Constant(mesh, 3.5e6)

    # Thermal Parameters
    T_base = 100.0
    T_inf = 20.0

    h_conv = dolfinx.fem.Constant(mesh, h)

    # Initial Condition
    T_curr = dolfinx.fem.Function(V)
    T_curr.interpolate(
        lambda x: np.full(x.shape[1], T_inf)
    )


    # Measures
    dx = ufl.Measure("dx", domain=mesh)

    ds = ufl.Measure(
        "ds",
        domain=mesh,
        subdomain_data=facet_tags
    )


    # Variational Problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = (
        (rho_cp / dt) * u * v * dx
        + k * ufl.dot(ufl.grad(u), ufl.grad(v)) * dx
        + h_conv * u * v * ds(2)
    )

    L = (
        (rho_cp / dt) * T_curr * v * dx
        + h_conv * T_inf * v * ds(2)
    )


    # Dirichlet BC (Base)
    base_facets = facet_tags.find(1)

    base_dofs = dolfinx.fem.locate_dofs_topological(
        V,
        mesh.topology.dim - 1,
        base_facets
    )

    bc = dolfinx.fem.dirichletbc(
        dolfinx.fem.Constant(mesh, T_base),
        base_dofs,
        V
    )


    # Linear Problem
    problem = LinearProblem(
        a,
        L,
        bcs=[bc],
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu"
        },
        petsc_options_prefix="fin_"
    )


    # Time Integration
    t = 0.0

    time_history = []
    q_history = []

    while t < t_end:

        t += dt

        T_new = problem.solve()

        T_curr.x.array[:] = T_new.x.array[:]

        q_form = dolfinx.fem.form(
            h_conv * (T_curr - T_inf) * ds(2)
        )

        q_out = mesh.comm.allreduce(
            dolfinx.fem.assemble_scalar(q_form),
            op=MPI.SUM
        )

        time_history.append(t)
        q_history.append(q_out)

    return T_curr, time_history, q_history

