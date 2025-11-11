"""
Demagnetization Potential in 3D with scikit-fem

Solves the demagnetization potential problem on a 3D mesh loaded from GMSH:
-Δφ = ∇·m in Ω
φ = 0 on ∂Ω

where:
- φ is the demagnetization potential
- m = (0, 0, 1) is the magnetization vector (uniform in z-direction)
- The source term ∇·m is only active in the magnetic domain (region 1)
- The air region (region 2) has no magnetization

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

    # Define linear form: ∫ m · ∇v dΩ (magnetization source)
    # Only active in magnetic region (domain 1)
    @LinearForm
    def magnetization_form(v, w):
        # Magnetization vector m = (0, 0, 1)
        m = np.array([0.0, 0.0, 1.0])
        # m · ∇v = m_z * ∂v/∂z
        return m[2] * v.grad[2]  # Only z-component is non-zero

    # Assemble stiffness matrix over entire domain
    A = asm(laplace_form, basis)

    # Assemble load vector only over magnetic region (domain 1)
    try:
        # Try to use subdomain if available
        if hasattr(m, 'subdomains') and '1' in m.subdomains:
            # Create basis restricted to magnetic domain
            basis_mag = basis.with_element(e).with_subdomain('1')
            b = asm(magnetization_form, basis_mag)
            print("Using subdomain-restricted assembly")
        else:
            # Fallback: assemble over entire domain
            # (In practice, you'd need to identify magnetic elements)
            b = asm(magnetization_form, basis)
            print("Using full domain assembly (fallback)")
    except Exception as e:
        print(f"Subdomain assembly failed: {e}")
        # Simple fallback: assemble over entire domain
        b = asm(magnetization_form, basis)
        print("Using full domain assembly")

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
