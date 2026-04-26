#import plotting as plot

from numba import njit, prange
import numpy as np

@njit(parallel=True)
def divergence_upwind_3d(rho, vx, vy, vz, dx, dy, dz):
    """
    inputs
    rho : (Nx, Ny, Nz)
    vx  : (Nx+1, Ny, Nz)     velocity at x-faces
    vy  : (Nx, Ny+1, Nz)     velocity at y-faces
    vz  : (Nx, Ny, Nz+1)     velocity at z-faces
    """

    Nx, Ny, Nz = rho.shape
    div = np.zeros((Nx, Ny, Nz), dtype=rho.dtype)

    
    # X-FLUX: F_x[i] = flux crossing face between cell i-1 and i    
    # flux_x has shape (Nx+1, Ny, Nz)
    flux_x = np.zeros((Nx+1, Ny, Nz), dtype=rho.dtype)

    # Internal faces: i = 1 ... Nx-1
    for i in prange(1, Nx):
        for j in range(Ny):
            for k in range(Nz):
                v = vx[i, j, k]
                if v > 0:   # flow from left
                    flux_x[i, j, k] = v * rho[i-1, j, k]
                else:       # flow from right
                    flux_x[i, j, k] = v * rho[i, j, k]

    # Boundary x = 0 (i = 0) — only outflow allowed
    for j in prange(Ny):
        for k in range(Nz):
            v = vx[0, j, k]
            if v < 0:      
                flux_x[0, j, k] = v * rho[0, j, k] # flowing out of domain
            else:
                flux_x[0, j, k] = 0.0 #nothing enters from outside

    # Boundary x = Nx (i = Nx)
    for j in prange(Ny):
        for k in range(Nz):
            v = vx[Nx, j, k]
            if v > 0:      
                flux_x[Nx, j, k] = v * rho[Nx-1, j, k] # flowing out of domain
            else:
                flux_x[Nx, j, k] = 0.0 # no incoming flux

    # Add contribution
    for i in prange(Nx):
        for j in range(Ny):
            for k in range(Nz):
                div[i,j,k] += (flux_x[i+1,j,k] - flux_x[i,j,k]) / dx

    
    # Y-FLUX: F_y[j] between cell j-1 and j    
    if(Ny > 1):
        flux_y = np.zeros((Nx, Ny+1, Nz), dtype=rho.dtype)

        # interior case (ybin from 1 to Ny-1)
        for i in prange(Nx):
            for j in range(1, Ny):
                for k in range(Nz):
                    v = vy[i, j, k]
                    if v > 0:
                        flux_y[i, j, k] = v * rho[i, j-1, k] #bin gets ions from prev cell
                    else:
                        flux_y[i, j, k] = v * rho[i, j, k]   #bin looses ions to next cell

        # boundary y = 0
        for i in prange(Nx):
            for k in range(Nz):
                v = vy[i, 0, k]
                if v < 0:
                    flux_y[i, 0, k] = v * rho[i, 0, k] #ions lost to FC
                else:
                    flux_y[i, 0, k] = 0.0 #no ions coming from outside

                
                

                    
        # boundary y = Ny
        for i in prange(Nx):
            for k in range(Nz):
                v = vy[i, Ny, k]
                if v > 0:
                    flux_y[i, Ny, k] = v * rho[i, Ny-1, k]
                else:
                    flux_y[i, Ny, k] = 0.0

        for i in prange(Nx):
            for j in range(Ny):
                for k in range(Nz):
                    div[i,j,k] += (flux_y[i,j+1,k] - flux_y[i,j,k]) / dy




    # Z-FLUX: F_z[k] between cell k-1 and k
    flux_z = np.zeros((Nx, Ny, Nz+1), dtype=rho.dtype)
    if(Nz>1):
        for i in prange(Nx):
            for j in range(Ny):
                for k in range(1, Nz):
                    v = vz[i, j, k]
                    if v > 0:
                        flux_z[i, j, k] = v * rho[i, j, k-1]
                    else:
                        flux_z[i, j, k] = v * rho[i, j, k]

        # boundary z = 0
        for i in prange(Nx):
            for j in range(Ny):
                v = vz[i, j, 0]
                if v < 0:
                    flux_z[i, j, 0] = v * rho[i, j, 0]
                else:
                    flux_z[i, j, 0] = 0.0

        # boundary z = Nz
        for i in prange(Nx):
            for j in range(Ny):
                v = vz[i, j, Nz]
                if v > 0:
                    flux_z[i, j, Nz] = v * rho[i, j, Nz-1]
                else:
                    flux_z[i, j, Nz] = 0.0

        for i in prange(Nx):
            for j in range(Ny):
                for k in range(Nz):
                    div[i,j,k] += (flux_z[i,j,k+1] - flux_z[i,j,k]) / dz

    return div

    
def transport_charge(param):

    vx_fc = mean_velocities_at_faces(0, param.geo.dim, param.mu * param.Ex, param.flow_x)
    vy_fc = mean_velocities_at_faces(1, param.geo.dim, param.mu * param.Ey, param.flow_y)
    vz_fc = mean_velocities_at_faces(2, param.geo.dim, param.mu * param.Ez, param.flow_z)


    """ compute the flux divergence """
    div = divergence_upwind_3d(param.rho, vx_fc, vy_fc, vz_fc, param.geo.dx, param.geo.dy, param.geo.dz)

    """ move charges """
    param.rho += -param.dt*div
    
    """ add source term """
    param.rho += param.dt*param.S

    """ safety fix, no negative ion concentration """
    param.rho = np.maximum(param.rho, 0)

    #return div

def mean_velocities_at_faces(axis, dim, attraction, flow):
    """
    input v = attraction + flow 
    v:  velocity at faces (Nx+1, Ny+1, Nz+1)

    returns: (depending on the axis chosen)
        vx: (Nx+1, Ny, Nz)
        vy: (Nx, Ny+1, Nz)
        vz: (Nx, Ny, Nz+1)
    """

    v = attraction + flow


    if(dim == 1):
        v_face = v
        
    if(dim == 2):
        if(axis==0):
            v_face = 0.5 * (
                v[:, :-1, :] +
                v[:, 1:,  :] 
            )
        elif(axis==1):
            v_face = 0.5 * (
                v[:-1, :, :] +
                v[1:,  :, :]
            )
        elif(axis==2):
            v_face = 0.5 * (
                v[:-1, :-1, :] +
                v[1:,  :-1, :]                
            )

    if(dim == 3):
        if(axis==0):
            v_face = 0.25 * (
                v[:, :-1, :-1] +
                v[:, 1:,  :-1] +
                v[:, :-1, 1:] +
                v[:, 1:,  1:]
            )
        elif(axis==1):
            v_face = 0.25 * (
                v[:-1, :, :-1] +
                v[1:,  :, :-1] +
                v[:-1, :, 1:] +
                v[1:,  :, 1:]
            )
        elif(axis==2):
            v_face = 0.25 * (
                v[:-1, :-1, :] +
                v[1:,  :-1, :] +
                v[:-1, 1:,  :] +
                v[1:,  1:,  :]
            )

    return v_face
