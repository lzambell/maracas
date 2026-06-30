import numba as nb
import numpy as np
import time
import tables as tab
from scipy.interpolate import RegularGridInterpolator

def compute_backward_distortions(param):

    if(param.geo.dim == 2):
        return compute_backward_distortions_2D(param)

    elif(param.geo.dim == 3):
        return compute_backward_distortions_3D(param)
    else:
        print('oops backward distortions for', param.geo.dim, 'D geometry is not implemented yet!')


def compute_backward_distortions_2D(param):
    
    x = np.linspace(param.geo.xmin+param.geo.dx/2, param.geo.xmax-param.geo.dx/2, param.geo.Nx_path)
    y = np.linspace(param.geo.ymin+param.geo.dy/2, param.geo.ymax-param.geo.dy/2, param.geo.Ny_path)


    
    distortions = np.stack((param.forward_delta_x, param.forward_delta_y), axis=-1)
    
    fwd_interp = RegularGridInterpolator(
        (x, y),
        distortions,
        bounds_error=False,
        fill_value=None
    )

    n_converged = 0
    n_iter = []
    residual = []
    
    n_tot = param.geo.Nx_path*param.geo.Ny_path
    
    t0 = time.time()
    for i, xr in enumerate(x):        
        print(f"[inverse map] x-slice {i+1}/{len(x)} (at x_reco = {xr:.4f})")

        t1 = time.time()
        
        for j, yr in enumerate(y):
            p_reco = np.array([xr, yr], dtype=float)
                
            if(k==0):
                p = p_reco.copy()
            else:
                p = p_true
                
            converged = False
            last_step_norm = np.inf
            
            for it in range(param.backward_max_iter):
                p_eval = p
                d = fwd_interp(p_eval)[0]

                p_new = p_reco - d
                
                
                last_step_norm = np.linalg.norm(p_new - p)

                p = p_new

                if last_step_norm < param.backward_norm_tol:
                    converged = True
                    n_converged += 1
                    break

            p_true = p
            d_inv = p_reco - p_true
            param.backward_delta_x[i,j,0] = d_inv[0]
            param.backward_delta_y[i,j,0] = d_inv[1]
            

                
            n_iter.append(it)
            residual.append(last_step_norm)
                
        print(f' .... took {time.time()-t1:.3f}s')

    print(' --- DONE ---')
    print(f'took {time.time()-t0:.3f}s')
    print(f'Nb of converged: {n_converged} / {n_tot} = {100.*n_converged/n_tot:.3f}')

    return n_iter, residual


        
def compute_backward_distortions_3D(param):
    
    x = np.linspace(param.geo.xmin+param.geo.dx/2, param.geo.xmax-param.geo.dx/2, param.geo.Nx_path)
    y = np.linspace(param.geo.ymin+param.geo.dy/2, param.geo.ymax-param.geo.dy/2, param.geo.Ny_path)
    z = np.linspace(param.geo.zmin+param.geo.dz/2, param.geo.zmax-param.geo.dz/2, param.geo.Nz_path)

    
    distortions = np.stack((param.forward_delta_x, param.forward_delta_y, param.forward_delta_z), axis=-1)
    
    fwd_interp = RegularGridInterpolator(
        (x, y, z),
        distortions,
        bounds_error=False,
        fill_value=None
    )

    n_converged = 0
    n_iter = []
    residual = []
    
    n_tot = param.geo.Nx_path*param.geo.Ny_path*param.geo.Nz_path
    
    t0 = time.time()
    for i, xr in enumerate(x):        
        print(f"[inverse map] x-slice {i+1}/{len(x)} (at x_reco = {xr:.4f})")
        t1 = time.time()
        
        for j, yr in enumerate(y):
            for k, zr in enumerate(z):
                p_reco = np.array([xr, yr, zr], dtype=float)
                
                if(k==0):
                    p = p_reco.copy()
                elif k == 1:
                    p = p_true_prev.copy()
                else:
                    p = 2*p_true_prev - p_true_prev2
                    
                converged = False
                last_step_norm = np.inf

                for it in range(param.backward_max_iter):
                    p_eval = p
                    d = fwd_interp(p_eval)[0]

                    """ search for the true point """
                    p_new = p_reco + d

                    #if clamp_each_iter:
                    #p_new = self._clamp_point(p_new)

                    last_step_norm = np.linalg.norm(p_new - p)
                    
                    p = p_new
                    
                    #d = fwd_interp(p)[0]
                    #last_step_norm = np.linalg.norm(p_new - p_reco)


                    if last_step_norm < param.backward_norm_tol:
                        converged = True
                        n_converged += 1
                        break

                p_true = p
                d = fwd_interp(p_true)[0]

                res = np.linalg.norm(p_true - d - p_reco)
                
                d_inv = p_reco - p_true
                param.backward_delta_x[i,j,k] = d_inv[0]
                param.backward_delta_y[i,j,k] = d_inv[1]
                param.backward_delta_z[i,j,k] = d_inv[2]


                if k == 0:
                    p_true_prev = p_true.copy()
                else:
                    p_true_prev2 = p_true_prev.copy()
                    p_true_prev  = p_true.copy()                

                n_iter.append(it)
                residual.append(res)
                
        print(f' .... took {time.time()-t1:.3f}s')

    print(' --- DONE ---')
    print(f'took {time.time()-t0:.3f}s')
    print(f'Nb of converged: {n_converged} / {n_tot} = {100.*n_converged/n_tot:.3f}')

    print('residuals stat: ')
    residual = np.asarray(residual)    
    print("min ", np.min(residual), " max ", np.max(residual))
    print('mean ', np.mean(residual), " std ", np.std(residual))
    
    return n_iter, residual


