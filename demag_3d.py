"""
Demagnetization Potential in 3D with scikit-fem

Solves the demagnetization potential problem on a 3D mesh loaded from GMSH:
-Δφ = ∇·m in Ω
φ = 0 on ∂Ω

where:
- φ is the demagnetization potential
- m is a piecewise constant magnetization vector function:
  * m = (0, 0, 1) in the magnetic domain (cube)
  * m = (0, 0, 0) in the air region
- The source term ∇·m creates jump discontinuities at material interfaces
- This approach naturally restricts the magnetization to the magnetic domain

This models the magnetic field outside a uniformly magnetized cube.
"""

import numpy as np
from skfem import Mesh, Basis, ElementTetP1, asm
from skfem.helpers import *
from skfem import BilinearForm, LinearForm
from skfem.utils import condense


def main():
    """Main function to solve the demagnetization potential problem."""
    
    # Load mesh from GMSH file
    m = Mesh.load('notebooks/files/cube_with_air.msh')
    e = ElementTetP1()
    basis = Basis(m, e)

    print(f"Mesh has {m.p.shape[1]} nodes and {m.t.shape[1]} tetrahedra")
    print(f"Mesh bounding box:")
    print(f"  x: [{m.p[0].min():.3f}, {m.p[0].max():.3f}]")
    print(f"  y: [{m.p[1].min():.3f}, {m.p[1].max():.3f}]")
    print(f"  z: [{m.p[2].min():.3f}, {m.p[2].max():.3f}]")

    # Check available subdomains
    if hasattr(m, 'subdomains'):
        print(f"Available subdomains: {list(m.subdomains.keys())}")
    else:
        print("No subdomain information found")

    # Define bilinear form: ∫ ∇φ · ∇v dΩ (Laplacian)
    @BilinearForm
    def laplace_form(u, v, _):
        return ddot(u.grad, v.grad)

    # Define magnetization as a piecewise constant function
    def magnetization_function(x, y, z):
        """
        Piecewise constant magnetization vector function.
        Returns m = (0, 0, 1) inside the cube [-0.5, 0.5]³
        Returns m = (0, 0, 0) outside (in air)
        """
        # Check if point is inside the cube [-0.5, 0.5]³
        inside_cube = (np.abs(x) <= 0.5) & (np.abs(y) <= 0.5) & (np.abs(z) <= 0.5)
        
        # Initialize magnetization arrays
        mx = np.zeros_like(x)
        my = np.zeros_like(x) 
        mz = np.zeros_like(x)
        
        # Set magnetization in magnetic domain
        mz[inside_cube] = 1.0
        
        return mx, my, mz

    # Define linear form: ∫ m · ∇v dΩ (magnetization source)
    @LinearForm
    def magnetization_form(v, w):
        # Get coordinates at quadrature points
        x, y, z = w.x[0], w.x[1], w.x[2]
        
        # Evaluate piecewise magnetization function
        mx, my, mz = magnetization_function(x, y, z)
        
        # m · ∇v = mx * ∂v/∂x + my * ∂v/∂y + mz * ∂v/∂z
        return mx * v.grad[0] + my * v.grad[1] + mz * v.grad[2]

    # Assemble stiffness matrix over entire domain
    A = asm(laplace_form, basis)

    # Assemble load vector over entire domain
    # The piecewise magnetization function automatically restricts the source
    b = asm(magnetization_form, basis)
    print("Using piecewise magnetization function (automatically domain-restricted)")

    print(f"System size: {A.shape[0]} x {A.shape[1]}")
    print(f"Load vector norm: {np.linalg.norm(b):.3e}")

    # Find boundary nodes (all external boundaries)
    boundary = m.boundary_nodes()
    print(f"Found {len(boundary)} boundary nodes")

    # Use condense method for symmetric BC application
    interior = basis.complement_dofs(boundary)
    A_int, b_int, *_ = condense(A, b, I=interior)

    print(f"Reduced system size: {A_int.shape[0]} x {A_int.shape[1]}")

    # Solve reduced system
    phi_int = np.linalg.solve(A_int.toarray(), b_int)

    # Reconstruct full solution
    phi = np.zeros(basis.N)
    phi[interior] = phi_int
    phi[boundary] = 0.0  # Homogeneous Dirichlet BC

    print(f"Solution computed with {len(phi)} degrees of freedom")
    print(f"Solution range: [{phi.min():.6f}, {phi.max():.6f}]")
    
    # Verify magnetization restriction by checking source term distribution
    x, y, z = m.p[0], m.p[1], m.p[2]
    mx, my, mz = magnetization_function(x, y, z)
    mag_nodes = np.sum(mz > 0)
    print(f"Magnetization active at {mag_nodes} nodes out of {len(phi)} total")

    # Export solution to VTU file
    output_filename = 'demag_potential_3d.vtu'

    # Create a dictionary with the solution data
    point_data = {
        'demagnetization_potential': phi
    }

    # Save to VTU format
    m.save(output_filename, point_data=point_data)

    print(f"Solution exported to {output_filename}")
    print(f"File contains:")
    print(f"  - Mesh with {m.p.shape[1]} nodes and {m.t.shape[1]} tetrahedra")
    print(f"  - Demagnetization potential field as point data")
    print(f"  - Solution range: [{phi.min():.6f}, {phi.max():.6f}]")
    print(f"\nTo visualize:")
    print(f"  - Open {output_filename} in ParaView")
    print(f"  - Apply 'Clip' or 'Slice' filters to see internal structure")
    print(f"  - Use 'Calculator' filter to compute magnetic field: -grad(demagnetization_potential)")
    print(f"  - Color by 'demagnetization_potential' to see field distribution")


if __name__ == "__main__":
    main()
