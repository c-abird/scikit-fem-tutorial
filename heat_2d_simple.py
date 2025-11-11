#!/usr/bin/env python3
"""
2D Heat Equation Solver with Mixed Boundary Conditions

Comprehensive implementation demonstrating:
1. Dirichlet boundary conditions (prescribed temperature)
2. Neumann boundary conditions (prescribed heat flux)
3. Mixed boundary condition problems
4. Various heat source configurations

Based on the scikit-fem tutorial notebook for 2D heat equation.
"""

import numpy as np
import matplotlib.pyplot as plt
import skfem as fem
from skfem.helpers import *


def solve_heat_2d_dirichlet(n=20, plot=True):
    """
    Solve 2D heat equation with pure Dirichlet boundary conditions.
    
    Problem: -Δu = 0 on unit square with:
    - u(x,0) = 1 (bottom: hot)
    - u(x,1) = 0 (top: cold)  
    - u(0,y) = 0 (left: cold)
    - u(1,y) = 0 (right: cold)
    """
    print(f"\n=== Dirichlet Problem (n={n}) ===")
    
    # Create mesh and function space
    mesh = fem.MeshTri.init_tensor(np.linspace(0, 1, n+1), np.linspace(0, 1, n+1))
    V = fem.Basis(mesh, fem.ElementTriP1())
    
    print(f"Mesh: {mesh.p.shape[1]} nodes, {mesh.t.shape[1]} triangles")
    
    # Define variational forms
    @fem.BilinearForm
    def a(u, v, _):
        return dot(grad(u), grad(v))
    
    @fem.LinearForm  
    def L(v, _):
        return 0.0 * v  # No source term
    
    # Assemble system
    A = a.assemble(V)
    b = L.assemble(V)
    
    # Apply Dirichlet boundary conditions
    x, y = mesh.p[0], mesh.p[1]
    tol = 1e-12
    
    bottom = np.where(np.abs(y) < tol)[0]
    top = np.where(np.abs(y - 1.0) < tol)[0]
    left = np.where(np.abs(x) < tol)[0]
    right = np.where(np.abs(x - 1.0) < tol)[0]
    
    # Manual boundary condition application
    A_dense = A.toarray()
    
    # Bottom edge: u = 1 (hot)
    for i in bottom:
        A_dense[i, :] = 0
        A_dense[i, i] = 1
        b[i] = 1.0
    
    # Other edges: u = 0 (cold)
    for boundary_nodes in [top, left, right]:
        for i in boundary_nodes:
            A_dense[i, :] = 0
            A_dense[i, i] = 1
            b[i] = 0.0
    
    # Solve
    u = np.linalg.solve(A_dense, b)
    
    print(f"Solution range: [{u.min():.3f}, {u.max():.3f}]")
    
    if plot:
        plot_solution(mesh, u, "Dirichlet BC: Hot Bottom, Cold Sides")
    
    return mesh, u


def solve_heat_2d_mixed(n=20, plot=True):
    """
    Solve 2D heat equation with mixed boundary conditions.
    
    Problem: -Δu = 0 on unit square with:
    - u(0,y) = 1 (left: Dirichlet, hot)
    - ∂u/∂n = -2 on right edge (Neumann, heat flux out)
    - ∂u/∂n = 0 on top/bottom (Neumann, insulated)
    """
    print(f"\n=== Mixed BC Problem (n={n}) ===")
    
    # Create mesh and function space
    mesh = fem.MeshTri.init_tensor(np.linspace(0, 1, n+1), np.linspace(0, 1, n+1))
    V = fem.Basis(mesh, fem.ElementTriP1())
    
    # Define variational forms
    @fem.BilinearForm
    def a(u, v, _):
        return dot(grad(u), grad(v))
    
    @fem.LinearForm  
    def L(v, _):
        return 0.0 * v  # No volume source
    
    # Assemble system
    A = a.assemble(V)
    b = L.assemble(V)
    
    # Add Neumann boundary conditions (heat flux on right edge)
    # Find boundary elements on right edge (x = 1)
    x, y = mesh.p[0], mesh.p[1]
    tol = 1e-12
    
    # Get boundary mesh
    boundary_mesh = mesh.boundary()
    
    # Find right edge elements
    right_elements = []
    for i, elem in enumerate(boundary_mesh.t.T):
        edge_coords = boundary_mesh.p[:, elem]
        if np.all(np.abs(edge_coords[0] - 1.0) < tol):
            right_elements.append(i)
    
    if right_elements:
        # Create boundary basis for right edge
        right_mesh = fem.MeshLine(
            boundary_mesh.p[:, boundary_mesh.t[:, right_elements].flatten()],
            np.arange(len(boundary_mesh.t[:, right_elements].flatten())).reshape(-1, 2)
        )
        
        # Simplified approach: add flux contribution directly
        # For right edge with outward normal (1,0), flux = -∂u/∂x = -2
        # This contributes ∫ g*v ds = ∫ (-2)*v ds to the load vector
        
        # Find nodes on right edge
        right_nodes = np.where(np.abs(x - 1.0) < tol)[0]
        
        # Add flux contribution (simplified integration)
        edge_length = 1.0 / (n)  # approximate edge length
        for i in right_nodes:
            b[i] += -2.0 * edge_length / 2  # flux * edge_length / 2 (linear elements)
    
    # Apply Dirichlet boundary condition on left edge
    left = np.where(np.abs(x) < tol)[0]
    
    A_dense = A.toarray()
    
    # Left edge: u = 1 (Dirichlet)
    for i in left:
        A_dense[i, :] = 0
        A_dense[i, i] = 1
        b[i] = 1.0
    
    # Solve
    u = np.linalg.solve(A_dense, b)
    
    print(f"Solution range: [{u.min():.3f}, {u.max():.3f}]")
    
    if plot:
        plot_solution(mesh, u, "Mixed BC: Dirichlet Left, Neumann Right")
    
    return mesh, u


def solve_heat_2d_source(n=20, source_type="uniform", plot=True):
    """
    Solve 2D heat equation with heat source.
    
    Problem: -Δu = f on unit square with homogeneous Dirichlet BC.
    """
    print(f"\n=== Heat Source Problem: {source_type} (n={n}) ===")
    
    # Create mesh and function space
    mesh = fem.MeshTri.init_tensor(np.linspace(0, 1, n+1), np.linspace(0, 1, n+1))
    V = fem.Basis(mesh, fem.ElementTriP1())
    
    # Define variational forms
    @fem.BilinearForm
    def a(u, v, _):
        return dot(grad(u), grad(v))
    
    # Different source terms
    if source_type == "uniform":
        @fem.LinearForm  
        def L(v, _):
            return 1.0 * v  # Uniform heat generation
    elif source_type == "localized":
        @fem.LinearForm  
        def L(v, w):
            # Heat source in center square [0.4, 0.6] × [0.4, 0.6]
            mask = ((w.x[0] > 0.4) & (w.x[0] < 0.6) & 
                   (w.x[1] > 0.4) & (w.x[1] < 0.6))
            return 10.0 * v * mask
    elif source_type == "sinusoidal":
        @fem.LinearForm  
        def L(v, w):
            return np.sin(np.pi * w.x[0]) * np.sin(np.pi * w.x[1]) * v
    else:
        @fem.LinearForm  
        def L(v, _):
            return 0.0 * v
    
    # Assemble system
    A = a.assemble(V)
    b = L.assemble(V)
    
    # Apply homogeneous Dirichlet BC on all boundaries
    x, y = mesh.p[0], mesh.p[1]
    tol = 1e-12
    
    boundary_nodes = np.unique(np.concatenate([
        np.where(np.abs(y) < tol)[0],          # bottom
        np.where(np.abs(y - 1.0) < tol)[0],    # top
        np.where(np.abs(x) < tol)[0],          # left
        np.where(np.abs(x - 1.0) < tol)[0]     # right
    ]))
    
    A_dense = A.toarray()
    
    for i in boundary_nodes:
        A_dense[i, :] = 0
        A_dense[i, i] = 1
        b[i] = 0.0
    
    # Solve
    u = np.linalg.solve(A_dense, b)
    
    print(f"Solution range: [{u.min():.3f}, {u.max():.3f}]")
    
    if plot:
        plot_solution(mesh, u, f"Heat Source: {source_type}")
    
    return mesh, u


def plot_solution(mesh, u, title="Temperature Distribution"):
    """Create visualization plots for the solution."""
    
    x, y = mesh.p[0], mesh.p[1]
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Contour plot
    ax1 = axes[0]
    levels = np.linspace(u.min(), u.max(), 11)
    cs = ax1.tricontourf(x, y, u, levels=levels, cmap='hot')
    ax1.tricontour(x, y, u, levels=levels, colors='black', alpha=0.3, linewidths=0.5)
    plt.colorbar(cs, ax=ax1, label='Temperature')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title(title)
    ax1.set_aspect('equal')
    
    # 3D surface plot
    ax2 = fig.add_subplot(122, projection='3d')
    surf = ax2.plot_trisurf(x, y, u, cmap='hot', alpha=0.8)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('Temperature')
    ax2.set_title('3D Surface')
    plt.colorbar(surf, ax=ax2, shrink=0.5)
    
    plt.tight_layout()
    plt.show()


def convergence_study():
    """Study mesh convergence for different problems."""
    
    print("\n" + "="*50)
    print("MESH CONVERGENCE STUDY")
    print("="*50)
    
    n_values = [10, 20, 40]
    
    print("\n1. Dirichlet Problem Convergence:")
    for n in n_values:
        mesh, u = solve_heat_2d_dirichlet(n=n, plot=False)
        print(f"  n={n:2d}: max={u.max():.6f}, min={u.min():.6f}, nodes={len(u)}")
    
    print("\n2. Mixed BC Problem Convergence:")
    for n in n_values:
        mesh, u = solve_heat_2d_mixed(n=n, plot=False)
        print(f"  n={n:2d}: max={u.max():.6f}, min={u.min():.6f}, nodes={len(u)}")


def main():
    """Main function demonstrating various 2D heat equation problems."""
    
    print("2D Heat Equation Solver - Comprehensive Examples")
    print("="*60)
    
    # 1. Standard Dirichlet problem
    print("\n1. DIRICHLET BOUNDARY CONDITIONS")
    mesh1, u1 = solve_heat_2d_dirichlet(n=20, plot=True)
    
    # 2. Mixed boundary conditions
    print("\n2. MIXED BOUNDARY CONDITIONS")
    mesh2, u2 = solve_heat_2d_mixed(n=20, plot=True)
    
    # 3. Heat source problems
    print("\n3. HEAT SOURCE PROBLEMS")
    
    # Uniform source
    mesh3a, u3a = solve_heat_2d_source(n=20, source_type="uniform", plot=True)
    
    # Localized source
    mesh3b, u3b = solve_heat_2d_source(n=20, source_type="localized", plot=True)
    
    # Sinusoidal source
    mesh3c, u3c = solve_heat_2d_source(n=20, source_type="sinusoidal", plot=True)
    
    # 4. Convergence study
    convergence_study()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("This script demonstrates:")
    print("• Dirichlet boundary conditions (prescribed temperature)")
    print("• Neumann boundary conditions (prescribed heat flux)")
    print("• Mixed boundary condition problems")
    print("• Various heat source configurations")
    print("• Mesh convergence studies")
    print("\nKey insights:")
    print("• Neumann conditions modify the load vector, not the matrix")
    print("• Mixed problems combine essential and natural boundary conditions")
    print("• Heat sources create non-zero right-hand sides")
    print("• Mesh refinement improves solution accuracy")


if __name__ == "__main__":
    main()
