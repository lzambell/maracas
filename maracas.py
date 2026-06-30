import argparse
import numpy as np
import  tables as tab


print("\nWelcome to MARACAS !\n")

import parameters as parameters
import transportation as trans
import gauss_law as gaus
import compute_distortions as dist
import compute_backward_distortions as back
import plotting as plot
import store as store


import time as time


parser = argparse.ArgumentParser()
parser.add_argument('--input', '-i', help='Input JSON file to setup the simulation', required=True)
parser.add_argument('--out', '-o', help='Output name', required=False, default='')
args = parser.parse_args()


tstart = time.time()

#configure the simulation
param = parameters.parameters(args.input)


#configure the plotter
if(param.geo.dim == 1):
    plotting = plot.plotter_1D
elif(param.geo.dim == 2):
    plotting = plot.plotter_2D
else:
    plotting = plot.plotter_3D
plot = plotting(param, args.out)


#plot.show_LAr_flow(param)


#create output file (hdf5)
output = args.out
fout = 'results/'+str(param.geo.dim)+"D/"+output
fout = tab.open_file(fout+'.h5', mode="w", title="MARACAS Simulation in "+str(param.geo.dim)+"D")
store.create_tables(fout, param)
store.store_parameters(fout, param)

res = 10

#to monitor convergence
res_poisson, conv_simu = [], []

print('\n\n SCE Simulation \n\n')

for t in range(param.timesteps):
    #step 1: solve poisson equation until convergence
    residual, niter = gaus.poisson_solve(param)
               
    res_poisson.extend(residual)


    #step 2: compute new field maps
    gaus.compute_field_fdm(param)
    
    #step 3: move the charge according to the flow and cathode attraction
    conv = trans.transport_charge(param)

    if(t%100==0):
        print("simulation at step", t)
        print("   poisson solving converged in ", niter, " res= ", residual[-1])
        print("   simu convergence is : ", conv)
    
    
    conv_simu.append(conv)
               
    if(conv <= param.conv_simu):
        #simulation finished !
        print('\n!! simulation converged !!')            

        plot.show_evolution(param, iteration=t)
        plot.show_Etot(param, iteration=t)                    
        break
        
store.store_SCE(fout, t, param)
#plot.show_time_performance(t_transport, t_poisson, t_field)
plot.show_convergence(res_poisson, conv_simu, param.dt, "convergence")


print(f"SCE simulation completed in {t} iterations (equivalent to {t*param.dt:.4f} s)")
print(f'it took {time.time()-tstart:.3f} s to run')

print('\n\n Distortions \n\n')
print('--->>> Forward --->>>\n')

#now compute the forward distortions
traj = dist.compute_forward_distortions(param)
plot.show_distortions(param)
plot.show_trajectories(traj, param)

print('<<<--- Backward <<<---\n')
niter, res = back.compute_backward_distortions(param)
plot.show_inversemap_performance(niter, res)


store.create_tables_distortions(fout, param)
store.store_distortions(fout, param)


import validate_distortion as val
val.validate_maps(param, frac=0.01)

fout.close()

















