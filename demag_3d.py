import numpy as np
from skfem.helpers import *
import skfem as fem


def main():
    """Main function to solve the demagnetization potential problem."""
    
    # Load mesh from GMSH file
    mesh = fem.Mesh.load('notebooks/files/cube_with_air.msh')
    
    # function spaces
    V = fem.Basis(mesh, fem.ElementTetP1())
    VDG = fem.Basis(mesh, fem.ElementTetP0())

    # sample indicator
    sample_arr = np.zeros(mesh.nelements)
    sample_arr[mesh.subdomains["magnetic"]] = 1.0
    sample = VDG.interpolate(sample_arr)
    
    # magnetization
    m = np.array([0.0, 0.0, 1.0])

    # Define bilinear form: ∫ ∇φ · ∇v dΩ (Laplacian)
    @fem.BilinearForm
    def a(u, v, _):
        return dot(grad(u), grad(v))

    # Define linear form: ∫ m · ∇v dΩ (magnetization source)
    @fem.LinearForm
    def L(v, _):
        return sample * dot(m, grad(v))

    # assembly
    A = a.assemble(V)
    b = L.assemble(V)

    # Solve
    D = V.get_dofs()
    u = fem.solve(*fem.condense(A, b, D=D))

    # Save to VTU format
    mesh.save("result.vtu", point_data={'u': u})


    # experiments

    # (1) compute H by projection
    u_field = V.interpolate(u)
    VV = fem.Basis(mesh, fem.ElementVector(fem.ElementTetP1()))

    @fem.BilinearForm
    def a(u, v, w):
        return dot(u, v)

    @fem.LinearForm
    def L(v, w):
        return dot(grad(u_field), v)

    A = a.assemble(VV)
    b = L.assemble(VV)

    H = fem.solve(A, b)

    mesh.save("result.vtu", point_data={'H': H.reshape(-1, 3)})

    # (2) use predefined projec method of sciket fem
    H = VV.project(grad(u_field))

    mesh.save("result.vtu", point_data={'H': H.reshape(-1, 3)})

if __name__ == "__main__":
    main()
