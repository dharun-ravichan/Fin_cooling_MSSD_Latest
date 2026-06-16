import gmsh
import dolfinx.io.gmsh
from mpi4py import MPI

def generate_fin_mesh(L=2.0, H=0.5, lc=0.05):
    
    gmsh.initialize()
    gmsh.model.add("cooling_fin")

    # create points
    p1 = gmsh.model.occ.addPoint(0, 0, 0, lc)
    p2 = gmsh.model.occ.addPoint(L, 0, 0, lc)
    p3 = gmsh.model.occ.addPoint(L, H, 0, lc)
    p4 = gmsh.model.occ.addPoint(0, H, 0, lc)

    # Create lines
    l_bottom = gmsh.model.occ.addLine(p1, p2)
    l_tip = gmsh.model.occ.addLine(p2, p3)
    l_top = gmsh.model.occ.addLine(p3, p4)
    l_base = gmsh.model.occ.addLine(p4, p1)

    # Create surface
    loop = gmsh.model.occ.addCurveLoop([l_bottom, l_tip, l_top, l_base])
    surface = gmsh.model.occ.addPlaneSurface([loop])
    
    gmsh.model.occ.synchronize()

    # Define Physical Boundaries
    gmsh.model.addPhysicalGroup(
    1,
    [l_base],
    tag=1,
    name="Base"
    )

    gmsh.model.addPhysicalGroup(
    1,
    [l_top, l_bottom],
    tag=2,
    name="Convective"
    )
    
    gmsh.model.addPhysicalGroup(
    1,
    [l_tip],
    tag=3,
    name="Adiabatic_Tip"
    )
    
    # Define Physical Surface 
    gmsh.model.addPhysicalGroup(2, [surface], tag=10, name="Fin_Domain")

    # Generate Mesh
    gmsh.model.mesh.generate(2)
    
    # Capture the MeshData object
    mesh_data = dolfinx.io.gmsh.model_to_mesh(
        comm=MPI.COMM_WORLD,
        model=gmsh.model,
        gdim=2,
        rank=0
    )
    
    # Extract attributes
    mesh = mesh_data.mesh
    cell_tags = mesh_data.cell_tags
    facet_tags = mesh_data.facet_tags
    
    gmsh.finalize()
    return mesh, cell_tags, facet_tags
