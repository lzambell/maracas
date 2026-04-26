import argparse
import numpy as np
import  tables as tab


print("\nWelcome to MARACAS !\n")

import parameters as parameters
import transportation as trans
import gauss_law as gaus
import compute_distortions as dist
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
plot = plot.plotter_2D(param, args.out)

plot.show_LAr_flow(param)



#create output file (hdf5)
output = args.out
fout = 'results/'+output+"_"+str(param.geo.dim)+"D"
fout = tab.open_file(fout+'.h5', mode="w", title="MARACAS Simulation in "+str(param.geo.dim)+"D")
store.create_tables(fout, param)
store.store_parameters(fout, param)

res = 10

#to monitor the simulation speed and convergence
t_transport, t_poisson, t_field, conv = [], [], [], []

print('\n\n SCE Simulation \n\n')

for t in range(param.timesteps):
    #step 1: move the charge according to the flow and cathode attraction
    t0 = time.time()
    trans.transport_charge(param)


    #step 2: solve poisson equation
    t1 = time.time()    
    res = gaus.poisson_solve(param)

    #step 3: compute new field maps
    t2 = time.time()
    gaus.compute_field_fdm(param)


    t3 = time.time()
    
    if(t>0):
        t_transport.append((t1-t0)*1e3)
        t_poisson.append((t2-t1)*1e3)
        t_field.append((t3-t2)*1e3)
        
    conv.append(res)
    

    #if(res <= param.conv):
    if(( t>0 and t%10000==0) or res <= param.conv):
        print('itreation ',t, " convergence is at: ", res)

        plot.show_evolution(param, iteration=t)

        plot.show_velocity(param)
        plot.show_projection_along_y(param, "rho")
        plot.show_projection_along_y(param, "Ex")
        plot.show_projection_along_y(param, "Ey")
        plot.show_projection_along_y(param, "phi")
        

        if(res <= param.conv):
            #simulation finished !
            print('simulation converged!')            
            break
        
store.store_SCE(fout, t, param)
plot.show_time_performance(t_transport, t_poisson, t_field)
plot.show_convergence(conv)


print(f"SCE simulation completed in {t} iterations (equivalent to {t**param.dt:.4f} s)")
print(f'it took {time.time()-tstart:.3f} s to run')

print('\n\n Distortions \n\n')

#now compute the distortions
traj = dist.compute_regular_distortions(param)
plot.show_distortions(param)
plot.show_trajectories(traj, param)


store.create_tables_distortions(fout, param)
store.store_distortions(fout, param)

fout.close()

















