#!/usr/bin/env python3
"""
2D Heat Equation Solver using scikit-fem

Solves the steady-state heat equation (Laplace equation) on a unit square:
    -Δu = 0,  (x,y) ∈ (0,1)²

with Dirichlet boundary conditions:
    - u(x,0) = 1 (bottom edge: hot)
    - u(x,1) = 0 (top edge: cold)
    - u(0,y) = 0 (left edge: cold)
    - u(1,y) = 0 (right edge: cold)

Uses scikit-fem's condense() method for clean boundary condition handling.
"""

import numpy as np
import matplotlib.pyplot as plt
import skfem as fem
from skfem.helpers import *


def solve_heat_2d(n=20, plot=True):
    """
    Solve the 2D heat equation on a unit square.
    
    Parameters:
    -----------
    n : int
        Number of subdivisions per direction (default: 20)
    plot : bool
        Whether to create visualization plots (default: True)
        
    Returns:
    --------
    mesh : skfem.MeshTri
        The triangular mesh
    u : numpy.ndarray
        Solution vector
    """
    
    # Step 1: Create mesh and function space
    print(f"Creating {n}x{n} triangular mesh...")
    mesh = fem.MeshTri.init_tensor(np.linspace(0, 1, n+1), np.linspace(0, 1, n+1))
    V = fem.Basis(mesh, fem.ElementTriP1())
    
    print(f"Mesh has {mesh.p.shape[1]} nodes and {mesh.t.shape[1]} triangles")
    
    # Step 2: Define variational forms
    @fem.BilinearForm
    def bilinear_form(u, v, _):
        """Bilinear form: ∫ ∇u · ∇v dx dy"""
        return dot(grad(u), grad(v))
    
    @fem.LinearForm  
    def linear_form(v, _):
        """Linear form: ∫ 0 * v dx dy (no source term)"""
        return 0.0 * v
    
    # Step 3: Assemble system
    print("Assembling stiffness matrix and load vector...")
    A = bilinear_form.assemble(V)
    b = linear_form.assemble(V)
    
    print(f"System size: {A.shape[0]} x {A.shape[1]}")
    
    # Step 4: Apply boundary conditions using condense()
    print("Applying boundary conditions...")
    
    # Get node coordinates
    x, y = mesh.p[0], mesh.p[1]
    
    # Find boundary nodes
    tol = 1e-12
    bottom = np.where(np.abs(y) < tol)[0]          # y = 0 (hot)
    top = np.where(np.abs(y - 1.0) < tol)[0]       # y = 1 (cold)
    left = np.where(np.abs(x) < tol)[0]            # x = 0 (cold)
    right = np.where(np.abs(x - 1.0) < tol)[0]     # x = 1 (cold)
    
    print(f"Boundary nodes: bottom={len(bottom)}, top={len(top)}, left={len(left)}, right={len(right)}")
    
    # Combine all boundary nodes and their values
    all_boundary_nodes = np.concatenate([bottom, top, left, right])
    all_boundary_values = np.concatenate([
        np.ones(len(bottom)),      # bottom = 1.0 (hot)
        np.zeros(len(top)),        # top = 0.0 (cold)
        np.zeros(len(left)),       # left = 0.0 (cold)
        np.zeros(len(right))       # right = 0.0 (cold)
    ])
    
    # Remove duplicates (corner nodes appear in multiple boundary arrays)
    unique_nodes, unique_indices = np.unique(all_boundary_nodes, return_index=True)
    unique_values = all_boundary_values[unique_indices]

    # Apply boundary conditions using manual matrix modification
    # Convert to dense matrix for easier manipulation
    A_dense = A.toarray()
    
    # Apply boundary conditions: A[i,:] = 0; A[i,i] = 1; b[i] = value
    for i in unique_nodes:
        A_dense[i, :] = 0
        A_dense[i, i] = 1
        b[i] = unique_values[np.where(unique_nodes == i)[0][0]]
    
    # Solve the system
    print("Solving linear system...")
    u = np.linalg.solve(A_dense, b)
    
    print(f"Solution computed with {len(u)} degrees of freedom")
    print(f"Solution range: [{u.min():.3f}, {u.max():.3f}]")
    
    # Step 6: Visualization
    if plot:
        create_plots(mesh, u)
    
    return mesh, u


def create_plots(mesh, u):
    """Create visualization plots for the solution."""
    
    x, y = mesh.p[0], mesh.p[1]
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Contour plot
    ax1 = axes[0]
    levels = np.linspace(0, 1, 11)
    cs = ax1.tricontourf(x, y, u, levels=levels, cmap='hot')
    ax1.tricontour(x, y, u, levels=levels, colors='black', alpha=0.3, linewidths=0.5)
    plt.colorbar(cs, ax=ax1, label='Temperature')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Temperature Distribution')
    ax1.set_aspect('equal')
    
    # 3D surface plot
    ax2 = fig.add_subplot(122, projection='3d')
    surf = ax2.plot_trisurf(x, y, u, cmap='hot', alpha=0.8)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('Temperature')
    ax2.set_title('3D Temperature Surface')
    plt.colorbar(surf, ax=ax2, shrink=0.5)
    
    plt.tight_layout()
    plt.show()
    
    # Temperature profile along centerline (x=0.5)
    plt.figure(figsize=(8, 4))
    center_idx = np.where(np.abs(x - 0.5) < 0.02)[0]  # nodes near x=0.5
    center_y = y[center_idx]
    center_u = u[center_idx]
    sort_idx = np.argsort(center_y)
    
    plt.plot(center_y[sort_idx], center_u[sort_idx], 'o-', linewidth=2, markersize=6)
    plt.xlabel('y (height)')
    plt.ylabel('Temperature')
    plt.title('Temperature Profile along Centerline (x=0.5)')
    plt.grid(True, alpha=0.3)
    plt.show()


def main():
    """Main function to run the heat equation solver."""
    
    print("2D Heat Equation Solver")
    print("=" * 40)
    
    # Solve with default parameters
    mesh, u = solve_heat_2d(n=20, plot=True)
    
    print("\nSolution completed successfully!")
    print("\nKey features of the solution:")
    print("- Temperature is 1.0 at the bottom edge (hot)")
    print("- Temperature is 0.0 on other edges (cold)")
    print("- Heat diffuses smoothly from bottom to top")
    print("- Solution is symmetric about the vertical centerline")
    
    # Optional: solve with different mesh sizes for comparison
    print("\nMesh refinement study:")
    for n in [10, 20, 40]:
        print(f"\nSolving with {n}x{n} mesh...")
        mesh_n, u_n = solve_heat_2d(n=n, plot=False)
        print(f"  Max temperature: {u_n.max():.6f}")
        print(f"  Min temperature: {u_n.min():.6f}")


if __name__ == "__main__":
    main()
