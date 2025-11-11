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
from skfem import Mesh, Basis, ElementTetP1, ElementTetDG0, asm
from skfem.helpers import *
from skfem import BilinearForm, LinearForm
from skfem.utils import condense


def main():
    """Main function to solve the demagnetization potential problem."""
    
    # Load mesh from GMSH file
    m = Mesh.load('notebooks/files/cube_with_air.msh')
    e = ElementTetP1()
    basis = Basis(m, e)
    
    # Create DG0 basis for piecewise constant magnetization
    e_dg0 = ElementTetDG0()
    basis_dg0 = Basis(m, e_dg0)

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

    # Create piecewise constant magnetization function using DG0 elements
    # Full magnetization vector m = (mx, my, mz)
    magnetization_x = np.zeros(basis_dg0.N)
    magnetization_y = np.zeros(basis_dg0.N)
    magnetization_z = np.zeros(basis_dg0.N)
    
    # Set magnetization values based on subdomain information
    if hasattr(m, 'subdomains') and 'magnetic' in m.subdomains:
        # Get elements in magnetic subdomain
        magnetic_elements = m.subdomains['magnetic']
        # Set magnetization vector m = (0, 0, 1) in magnetic domain
        magnetization_x[magnetic_elements] = 0.0
        magnetization_y[magnetic_elements] = 0.0
        magnetization_z[magnetic_elements] = 1.0
        print(f"Set magnetization m=(0,0,1) in {len(magnetic_elements)} magnetic elements")
    elif hasattr(m, 'subdomains') and '1' in m.subdomains:
        # Fallback to numeric subdomain tag
        magnetic_elements = m.subdomains['1']
        magnetization_x[magnetic_elements] = 0.0
        magnetization_y[magnetic_elements] = 0.0
        magnetization_z[magnetic_elements] = 1.0
        print(f"Set magnetization m=(0,0,1) in {len(magnetic_elements)} magnetic elements (subdomain '1')")
    else:
        print("Warning: No magnetic subdomain found, magnetization will be zero everywhere")
    
    # Define linear form: ∫ m · ∇v dΩ (magnetization source)
    @LinearForm
    def magnetization_form(v, w):
        # Interpolate DG0 magnetization components to quadrature points
        mx = basis_dg0.interpolate(magnetization_x)(w)
        my = basis_dg0.interpolate(magnetization_y)(w)
        mz = basis_dg0.interpolate(magnetization_z)(w)
        # m · ∇v = mx * ∂v/∂x + my * ∂v/∂y + mz * ∂v/∂z
        return mx * v.grad[0] + my * v.grad[1] + mz * v.grad[2]

    # Assemble stiffness matrix over entire domain
    A = asm(laplace_form, basis)

    # Assemble load vector over full domain using piecewise constant magnetization
    b = asm(magnetization_form, basis)
    print("Using DG0 piecewise constant magnetization over full domain")

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
    
    # Verify magnetization restriction by checking load vector and DG0 function
    nonzero_entries = np.sum(np.abs(b) > 1e-12)
    nonzero_mag_x = np.sum(np.abs(magnetization_x) > 1e-12)
    nonzero_mag_y = np.sum(np.abs(magnetization_y) > 1e-12)
    nonzero_mag_z = np.sum(np.abs(magnetization_z) > 1e-12)
    print(f"Load vector has {nonzero_entries} non-zero entries out of {len(b)} total")
    print(f"Magnetization active in {nonzero_mag_z} elements out of {len(magnetization_z)} total")
    print(f"Magnetization components: mx={nonzero_mag_x}, my={nonzero_mag_y}, mz={nonzero_mag_z}")

    # Export solution to VTU file
    output_filename = 'demag_potential_3d.vtu'

    # Create a dictionary with the solution data
    point_data = {
        'demagnetization_potential': phi
    }
    
    # Also save the magnetization as cell data for visualization
    cell_data = {
        'magnetization_x': magnetization_x,
        'magnetization_y': magnetization_y,
        'magnetization_z': magnetization_z
    }

    # Save to VTU format
    m.save(output_filename, point_data=point_data, cell_data=cell_data)

    print(f"Solution exported to {output_filename}")
    print(f"File contains:")
    print(f"  - Mesh with {m.p.shape[1]} nodes and {m.t.shape[1]} tetrahedra")
    print(f"  - Demagnetization potential field as point data")
    print(f"  - Full magnetization vector as cell data")
    print(f"  - Solution range: [{phi.min():.6f}, {phi.max():.6f}]")
    print(f"\nTo visualize:")
    print(f"  - Open {output_filename} in ParaView")
    print(f"  - Apply 'Clip' or 'Slice' filters to see internal structure")
    print(f"  - Use 'Calculator' filter to compute magnetic field: -grad(demagnetization_potential)")
    print(f"  - Color by 'demagnetization_potential' to see field distribution")


if __name__ == "__main__":
    main()
