import dolfinx
import ufl
import numpy as np

from mpi4py import MPI
from dolfinx.fem.petsc import LinearProblem


def solve_transient_fin_mms(
    mesh,
    facet_tags,
    t_end=0.2,
    dt=0.01,
    return_error=True
):
    """
    Transient MMS verification problem.

    Exact solution:

    T_exact(x,y,t)
    =
    20 + 80*sin(pi*x/L)*sin(pi*y/H)*exp(-t)

    Returns:
        L2 error norm
        H1 seminorm error
    """

    V = dolfinx.fem.functionspace(mesh, ("P", 1))

    Lx = 2.0
    Hy = 0.5

    k = 400.0
    rho_cp = 3.5e6

    k_c = dolfinx.fem.Constant(mesh, k)
    rho_c = dolfinx.fem.Constant(mesh, rho_cp)

    x = ufl.SpatialCoordinate(mesh)

    # Exact MMS solution

    t_symbol = dolfinx.fem.Constant(mesh, 0.0)

    A = (
    80.0
    * ufl.sin(np.pi * x[0] / Lx)
    * ufl.sin(np.pi * x[1] / Hy)
    )

    t_symbol = dolfinx.fem.Constant(mesh, 0.0)

    T_exact = 20.0 + A * ufl.exp(-t_symbol)

    dTdt = -A * ufl.exp(-t_symbol)
    

    # Manufactured source term
    lapT = ufl.div(ufl.grad(T_exact))

    f_mms = rho_c * dTdt - k_c * lapT
    
    # Initial condition
    T_n = dolfinx.fem.Function(V)

    def initial_condition(xvals):
        return (
            20.0
            + 80.0
            * np.sin(np.pi * xvals[0] / Lx)
            * np.sin(np.pi * xvals[1] / Hy)
        )

    T_n.interpolate(initial_condition)

    # Variational form
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    dx = ufl.dx

    a = (
        rho_c / dt * u * v * dx
        + k_c * ufl.dot(ufl.grad(u), ufl.grad(v)) * dx
    )

    Lform = (
        rho_c / dt * T_n * v * dx
        + f_mms * v * dx
    )


    # Homogeneous Dirichlet boundaries
    def boundary_all(x):
        return (
            np.isclose(x[0], 0.0)
            | np.isclose(x[0], Lx)
            | np.isclose(x[1], 0.0)
            | np.isclose(x[1], Hy)
        )

    boundary_dofs = dolfinx.fem.locate_dofs_geometrical(
        V,
        boundary_all
    )

    bc_func = dolfinx.fem.Function(V)
    bc_func.interpolate(
        lambda xx: (
            20.0
            + 80.0
            * np.sin(np.pi * xx[0] / Lx)
            * np.sin(np.pi * xx[1] / Hy)
            * np.exp(-float(t_symbol.value))
        )
    )

    bc = dolfinx.fem.dirichletbc(
        bc_func,
        boundary_dofs
    )

    # Linear solver
    problem = LinearProblem(
        a,
        Lform,
        bcs=[bc],
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu"
        },
        petsc_options_prefix="mms_"
    )

    # Time stepping
    t = 0.0

    while t < t_end:

        t += dt

        t_symbol.value = t

        bc_func.interpolate(
            lambda xx: (
                20.0
                + 80.0
                * np.sin(np.pi * xx[0] / Lx)
                * np.sin(np.pi * xx[1] / Hy)
                * np.exp(-t)
            )
        )

        T_sol = problem.solve()

        T_n.x.array[:] = T_sol.x.array[:]

  
    # Error computation
    T_exact_final = (
        20.0
        + 80.0
        * ufl.sin(np.pi * x[0] / Lx)
        * ufl.sin(np.pi * x[1] / Hy)
        * ufl.exp(-t_end)
    )

    error_L2 = dolfinx.fem.form(
        (T_sol - T_exact_final) ** 2 * dx
    )

    L2 = np.sqrt(
        mesh.comm.allreduce(
            dolfinx.fem.assemble_scalar(error_L2),
            op=MPI.SUM
        )
    )

    error_H1 = dolfinx.fem.form(
        ufl.dot(
            ufl.grad(T_sol - T_exact_final),
            ufl.grad(T_sol - T_exact_final)
        ) * dx
    )

    H1 = np.sqrt(
        mesh.comm.allreduce(
            dolfinx.fem.assemble_scalar(error_H1),
            op=MPI.SUM
        )
    )

    if return_error:
        return L2, H1

    return T_sol
