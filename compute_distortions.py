import numba as nb
import numpy as np
import time



@nb.njit(cache=True, fastmath=True)
def mobility(E):
    """
    LAr electron mobility parameterization.
    Input E must be in V/cm (not kV/cm).
    Output mobility in cm^2/(V·s).
    """
    # convert to kV/cm
    E_kV = abs(E) * 1e-3

    # constants
    T = 89.0
    T0 = 89.0

    a0 = 551.6
    a1 = 7158.3
    a2 = 4440.43
    a3 = 4.29
    a4 = 43.63
    a5 = 0.2053

    num = (a0
           + a1 * E_kV
           + a2 * (E_kV ** 1.5)
           + a3 * (E_kV ** 2.5))

    den = (1.0
           + (a1 / a0) * E_kV
           + a4 * (E_kV ** 2)
           + a5 * (E_kV ** 3))

    mu = num / den

    # temperature correction
    mu = mu * (T / T0) ** (-1.5)

    return mu              # cm^2/(V·s)




@nb.njit(cache=True, fastmath=True)
def interp_bilinear(x, y, Ex, Ey, x_grid, y_grid):
    """
    Bilinear interpolation for Ex, Ey on a uniform grid.
    """
    Nx = x_grid.size
    Ny = y_grid.size

    # Find indices
    # assumes monotonic grid
    i = int((x - x_grid[0]) / (x_grid[1] - x_grid[0]))
    j = int((y - y_grid[0]) / (y_grid[1] - y_grid[0]))

    # Clamp indices
    if i < 0: i = 0
    if i > Nx - 2: i = Nx - 2
    if j < 0: j = 0
    if j > Ny - 2: j = Ny - 2

    # Fractions
    x1 = x_grid[i]
    x2 = x_grid[i+1]
    y1 = y_grid[j]
    y2 = y_grid[j+1]

    tx = (x - x1) / (x2 - x1)
    ty = (y - y1) / (y2 - y1)

    # Bilinear interpolation
    Ex_val = (
        (1 - tx) * (1 - ty) * Ex[i, j] +
        tx       * (1 - ty) * Ex[i+1, j] +
        (1 - tx) * ty       * Ex[i, j+1] +
        tx       * ty       * Ex[i+1, j+1]
    )

    Ey_val = (
        (1 - tx) * (1 - ty) * Ey[i, j] +
        tx       * (1 - ty) * Ey[i+1, j] +
        (1 - tx) * ty       * Ey[i, j+1] +
        tx       * ty       * Ey[i+1, j+1]
    )

    return float(Ex_val), float(Ey_val)



@nb.njit(cache=True, fastmath=True)
def drift_path_2d(x0, y0,
                  Ex, Ey, x_grid, y_grid,
                  xmin, xmax, dx,
                  ymin, ymax, dy,
                  ds, E0):
    
    max_steps = 20000
    E0 *= 1e-2
    
    traj = np.zeros((max_steps, 2))

    
    drift_time = 0.0
    drift_len  = 0.0
    
    xs = xmin + x0*dx
    ys = ymin + y0*dy

    traj[0,0] = xs
    traj[0,1] = ys

    x, y = xs, ys
    step = 0

    while True:
        step += 1
        if step >= max_steps:
            print('reached max_steps limit')
            break

        # interpolate E-field
        ex, ey = interp_bilinear(x, y, Ex, Ey, x_grid, y_grid)

        ex *= 1e-2 #in V/cm
        ey *= 1e-2 #in V/cm
        
        # magnitude
        Etot = np.sqrt(ex*ex + ey*ey)

        # velocity
        mu = mobility(Etot)
        v  = mu * Etot     # cm/sec
        v *= 1e-2 #m/sec
        
        # drift direction
        ux = ex / Etot
        uy = ey / Etot

        # update time + length
        drift_time += ds / v
        drift_len  += ds

        # update position (electrons drift opposite direction)
        x_new = x - ds * ux
        y_new = y - ds * uy

        #if(step%10 == 0):
        traj[step,0] = x_new
        traj[step,1] = y_new

        # boundaries
        if (x_new < xmin) or (x_new > xmax) or (y_new < ymin) or (y_new > ymax):
            break

        x = x_new
        y = y_new

    reco_x = xmin + drift_time * (mobility(E0) * E0) * 1e-2 
    reco_y = y

    #return traj[:step+1], np.array([x0, y0, reco_x, reco_y, drift_time, drift_len])
    return traj[:step+1], np.array([xs-reco_x, ys-reco_y, drift_time, drift_len])






def compute_regular_distortions(param):

    if(param.geo.dim == 2):
        start_positions = [[ix,iy] for iy in range(param.geo.Ny_path) for ix in range(param.geo.Nx_path)]
        
    N = len(start_positions)
    print('--> ',N, ' points to track')

    trajectories = []
    t0 = time.time()
    i = 0
    for pos in start_positions:
        
        traj, res = drift_path_2d(
            pos[0], pos[1],
            param.Ex[:,:,0], param.Ey[:,:,0],
            param.x_field, param.y_field,
            param.geo.xmin, param.geo.xmax, param.geo.dx_path,
            param.geo.ymin, param.geo.ymax, param.geo.dy_path,
            param.geo.ds_path, param.E0
        )
        
        param.delta_x[pos[0], pos[1], 0] = res[0]
        param.delta_y[pos[0], pos[1], 0] = res[1]
        #param.delta_z[pos[0], pos[1], 0] = res[0]
        #if(i%500==0):
        #    trajectories.append(traj)
        if(pos[0]==param.geo.Nx_path-1 or pos[1] == 0 or pos[1]==param.geo.Ny_path-1):
            trajectories.append(traj)
        if(i%10000==0):
            print(i, 'at',pos)
            xs = param.geo.xmin + pos[0]*param.geo.dx_path
            ys = param.geo.ymin + pos[1]*param.geo.dy_path
            print('corresponding to ', xs, ys)
            print('delta : ', res[0], res[1])
            print('drift time ', res[2])
            print('drift length', res[3])
            print('nstep: ', len(traj))
        #print(pos, '-->', res)
        i+=1
    print('distortions took ', time.time()-t0)

    return trajectories[::10]
