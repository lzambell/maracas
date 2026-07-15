import numpy as np
from numba import njit, prange



@njit(nopython = True)
def build_rho_node(rho_node, rho, dim):
    Nx_p1, Ny_p1, Nz_p1 = rho_node.shape
    Nx, Ny, Nz = rho.shape

    # Clear arrays
    for i in range(Nx_p1):
        for j in range(Ny_p1):
            for k in range(Nz_p1):
                rho_node[i,j,k] = 0.0

    # Accumulate contributions from surrounding cells
    if dim == 1:
        for i in range(Nx):
            rho_node[i,0,0] += rho[i,0,0]
            rho_node[i+1,0,0] += rho[i,0,0]

    elif dim == 2:
        for i in range(Nx):
            for j in range(Ny):
                r = rho[i,j,0]
                rho_node[i,  j,  0] += r
                rho_node[i+1,j,  0] += r
                rho_node[i,  j+1,0] += r
                rho_node[i+1,j+1,0] += r

    else:  # dim == 3
        for i in range(Nx):
            for j in range(Ny):
                for k in range(Nz):
                    r = rho[i,j,k]
                    rho_node[i,  j,  k] += r
                    rho_node[i+1,j,  k] += r
                    rho_node[i,  j+1,k] += r
                    rho_node[i+1,j+1,k] += r
                    rho_node[i,  j,  k+1] += r
                    rho_node[i+1,j,  k+1] += r
                    rho_node[i,  j+1,k+1] += r
                    rho_node[i+1,j+1,k+1] += r

    return



@njit(nopython = True)
def jacobi_step(phi_new, phi_old, rho_node, dx, eps, dim):
    """ rho_node lives on the physical grid, hence the -1 indices """
    
    Nx, Ny, Nz = phi_old.shape
    c = dx*dx/eps

    if dim == 1:
        for i in range(1, Nx-1):
            phi_new[i,0,0] = 0.5 * (phi_old[i-1,0,0] +
                                    phi_old[i+1,0,0] +
                                    c*rho_node[i-1,0,0])

    elif dim == 2:
        for i in range(1, Nx-1):
            for j in range(1, Ny-1):            
                phi_new[i,j,0] = 0.25 * (phi_old[i-1,j,0] +
                                         phi_old[i+1,j,0] +
                                         phi_old[i,j-1,0] +
                                         phi_old[i,j+1,0] +
                                         c*rho_node[i-1, j-1,0])

            
        
    else:  # 3D
        for i in range(1, Nx-1):
            for j in range(1, Ny-1):
                for k in range(1, Nz-1):
                    phi_new[i,j,k] = (phi_old[i-1,j,k] +
                                      phi_old[i+1,j,k] +
                                      phi_old[i,j-1,k] +
                                      phi_old[i,j+1,k] +
                                      phi_old[i,j,k-1] +
                                      phi_old[i,j,k+1] +
                                      c*rho_node[i-1,j-1,k-1]) / 6.0



@njit(nopython = True)
def poisson_residual(phi, rho_node, dx, eps, dim):
    """
    Compute max residual of discretized Poisson equation:
        r = Laplacian(phi) + rho/eps

    Boundary values (ghost cells) are assumed already set in `phi`.
    rho_node lives on the physical nodes: (Nx+1, Ny+1, Nz+1).
    """
    
    Nx, Ny, Nz = phi.shape
    inv_dx2 = 1.0 / (dx * dx)

    max_res = 0.0

    if dim == 1:
        # Physical region in 1D: i = 1 .. Nx-2
        for i in range(1, Nx-1):
            lap = (phi[i-1,0,0] - 2*phi[i,0,0] + phi[i+1,0,0]) * inv_dx2
            r = lap + rho_node[i-1,0,0] / eps
            r_abs = abs(r)
            if r_abs > max_res:
                max_res = r_abs

    elif dim == 2:
        # Physical region: i = 1..Nx-2, j = 1..Ny-2
        for i in range(1, Nx-1):
            for j in range(1, Ny-1):
                lap = (
                    (phi[i-1,j,0] - 2*phi[i,j,0] + phi[i+1,j,0]) +
                    (phi[i,j-1,0] - 2*phi[i,j,0] + phi[i,j+1,0])
                ) * inv_dx2

                r = lap + rho_node[i-1, j-1, 0] / eps
                r_abs = abs(r)
                                
                if r_abs > max_res:
                    max_res = r_abs
    
    else:  # dim == 3
        for i in range(1, Nx-1):
            for j in range(1, Ny-1):
                for k in range(1, Nz-1):
                    lap = (
                        (phi[i-1,j,k] - 2*phi[i,j,k] + phi[i+1,j,k]) +
                        (phi[i,j-1,k] - 2*phi[i,j,k] + phi[i,j+1,k]) +
                        (phi[i,j,k-1] - 2*phi[i,j,k] + phi[i,j,k+1])
                    ) * inv_dx2

                    r = lap + rho_node[i-1, j-1, k-1] / eps
                    r_abs = abs(r)
                    if r_abs > max_res:
                        max_res = r_abs

    return max_res

                    
def poisson_solve(param):

    dim = param.geo.dim
    dx  = param.geo.dx
    eps = param.epsilon

    # Build rho on node faces 
    rho_node = np.zeros_like(param.Ex) 
    build_rho_node(rho_node, param.rho, dim)

    # Compute weights on nodes (same shape as rho_node)
    
    weights = np.zeros_like(rho_node)
    if dim == 1:
        weights[:-1,0,0] += 1
        weights[1:, 0,0] += 1
    elif dim == 2:
        weights[:-1,:-1,0] += 1
        weights[1:, :-1,0] += 1
        weights[:-1,1:, 0] += 1
        weights[1:, 1:, 0] += 1
    else:
        weights[:-1,:-1,:-1] += 1
        weights[1:, :-1,:-1] += 1
        weights[:-1,1:, :-1] += 1
        weights[1:, 1:, :-1] += 1
        weights[:-1,:-1,1:] += 1
        weights[1:, :-1,1:] += 1
        weights[:-1,1:, 1:] += 1
        weights[1:, 1:, 1:] += 1

    rho_node /= weights

    # Jacobi iteration
    phi_old = param.phi
    phi_new = phi_old.copy()

    #(re) set Dirichlet BC on previous phi (safety precaution)
    param.set_boundary_conditions_with_ghost(phi_old)


    jacobi_step(phi_new, phi_old, rho_node, dx, eps, dim)

    #set Dirichlet BC on new phi 
    param.set_boundary_conditions_with_ghost(phi_new)

    # convergence metric
    res = poisson_residual(phi_new, rho_node, dx, eps, dim)
    residuals = [res]
    ipoisson = 1
    while(res > param.conv_poisson):
        
        phi_old = phi_new.copy()
        #phi_new = phi_old.copy()
        
        jacobi_step(phi_new, phi_old, rho_node, dx, eps, dim)

        #set Dirichlet BC on new phi 
        param.set_boundary_conditions_with_ghost(phi_new)

        # convergence metric
        res = poisson_residual(phi_new, rho_node, dx, eps, dim)
        residuals.append(res)

        #if(ipoisson%500 == 0):
        #    print('   poisson solving iteration ', ipoisson, ' res=', res)
        ipoisson += 1
    
    param.phi = phi_new
    return residuals, ipoisson



    
def compute_field_fdm(param):
    
    if(param.geo.dim == 1):
        Ex = np.zeros_like(param.Ex)    
        Ex = - (param.phi[2:,:,:] - param.phi[:-2,:,:]) / (2 * param.geo.dx)
        param.Ex = Ex
        
    if(param.geo.dim == 2):
        Ex = np.zeros_like(param.Ey)
        Ey = np.zeros_like(param.Ez)
    
        Ex = - (param.phi[2:,1:-1,:] - param.phi[:-2,1:-1,:]) / (2 * param.geo.dx)        
        Ey = - (param.phi[1:-1,2:,:] - param.phi[1:-1,:-2,:]) / (2 * param.geo.dy)

        param.Ex = Ex
        param.Ey = Ey


                
    if(param.geo.dim == 3):
        Ex = np.zeros_like(param.Ex)
        Ey = np.zeros_like(param.Ey)
        Ez = np.zeros_like(param.Ez)
        
        Ex = - (param.phi[2:,1:-1,1:-1] - param.phi[:-2,1:-1,1:-1]) / (2 * param.geo.dx)
        Ey = - (param.phi[1:-1,2:,1:-1] - param.phi[1:-1,:-2,1:-1]) / (2 * param.geo.dy)
        Ez = - (param.phi[1:-1,1:-1,2:] - param.phi[1:-1,1:-1,:-2]) / (2 * param.geo.dz)

        param.Ex = Ex
        param.Ey = Ey
        param.Ez = Ez

