import numpy as np
from scipy.interpolate import RegularGridInterpolator

import matplotlib.pyplot as plt

def validate_maps(param, frac=0.02):
    rng = np.random.default_rng(None)

    
    n_tot = param.geo.Nx_path*param.geo.Ny_path*param.geo.Nz_path
    ntest = int(n_tot*frac)

    x = np.linspace(param.geo.xmin+param.geo.dx/2, param.geo.xmax-param.geo.dx/2, param.geo.Nx_path)
    y = np.linspace(param.geo.ymin+param.geo.dy/2, param.geo.ymax-param.geo.dy/2, param.geo.Ny_path)
    z = np.linspace(param.geo.zmin+param.geo.dz/2, param.geo.zmax-param.geo.dz/2, param.geo.Nz_path)

    
    fwd_distortions = np.stack((param.forward_delta_x, param.forward_delta_y, param.forward_delta_z), axis=-1)
    
    fwd_interp = RegularGridInterpolator(
        (x, y, z),
        fwd_distortions,
        bounds_error=False,
        fill_value=None
    )
    bkd_distortions = np.stack((param.backward_delta_x, param.backward_delta_y, param.backward_delta_z), axis=-1)
    
    bkd_interp = RegularGridInterpolator(
        (x, y, z),
        bkd_distortions,
        bounds_error=False,
        fill_value=None
    )


    residuals = []


    print('TEST: ')
    p_true = np.array([
        rng.uniform(param.geo.xmin, param.geo.xmax),
        rng.uniform(param.geo.ymin, param.geo.ymax),
        rng.uniform(param.geo.zmin, param.geo.zmax),
    ])

    print('TRUE at ', p_true)

    p_reco = p_true - fwd_interp(p_true)[0]
    print('RECO at ', p_reco)

    p_true_recov = p_reco - bkd_interp(p_reco)[0]
    print('BACK TO TRUE ', p_true_recov)
    print('--> ', p_true_recov - p_true)

    d = p_true_recov - p_true
    norm = np.linalg.norm(d)
    print('norm == ', norm)

    
    for _ in range(ntest):
        p_true = np.array([
            rng.uniform(param.geo.xmin, param.geo.xmax),
            rng.uniform(param.geo.ymin, param.geo.ymax),
            rng.uniform(param.geo.zmin, param.geo.zmax),
        ])

        p_reco = p_true - fwd_interp(p_true)[0]

            
        p_true_recov = p_reco - bkd_interp(p_reco)[0]
        
        residuals.append(p_true_recov - p_true)

    residuals = np.asarray(residuals)
    norms = np.linalg.norm(residuals, axis=1)
    
    print(norms.shape)
    print("\nmean_dx", residuals[:, 0].mean(),
          "\nmean_dy", residuals[:, 1].mean(),
          "\nmean_dz", residuals[:, 2].mean(),
          "\nstd_dx", residuals[:, 0].std(),
          "\nstd_dy", residuals[:, 1].std(),
          "\nstd_dz", residuals[:, 2].std(),
          "\nmean_norm", norms.mean(),
          "\nmax_norm", norms.max(),
          "\nrms_norm", np.sqrt(np.mean(norms**2)))

    fig=plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(norms.ravel(), bins = 100, range = [0, norms.max()], histtype='stepfilled', fc='None', edgecolor='k')
    ax.set_xlabel(r'$p_{true} - F^{-1}(F(p_{true}))$ [m]')
    plt.show()

    
