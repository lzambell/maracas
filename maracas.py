import argparse
import numpy as np
import  tables as tab
import time as time
print("\nWelcome to MARACAS !\n")

import parameters as parameters
import transportation as trans
import gauss_law as gaus
import compute_distortions as dist
import plotting as plot


parser = argparse.ArgumentParser()
parser.add_argument('--input', '-i', help='Input JSON file to setup the simulation', required=True)
parser.add_argument('--out', '-o', help='Output name', required=False, default='')
args = parser.parse_args()


tstart = time.time()
param = parameters.parameters(args.input)
plot = plot.plotter_2D(param, args.out)
#plot.show_flow(param)
#plot.show_evolution(param)
plot.show_velocity(param)
#exit()
import store as store

output = args.out
fout = 'results/'+output+"_"+str(param.geo.dim)+"D"
fout = tab.open_file(fout+'.h5', mode="w", title="MARACAS Simulation in "+str(param.geo.dim)+"D")
store.create_tables(fout, param)
store.store_parameters(fout, param)

res = 10


t_transport, t_poisson, t_field, conv = [], [], [], []



#plot.show_phi(geo, param)
#plot.show_rho(param, mag=1)
#plot.show_Ex_Ey(param)


for t in range(param.timesteps):
    t0 = time.time()
    trans.transport_charge(param)
    t1 = time.time()
    res = gaus.poisson_solve(param)
    t2 = time.time()
    gaus.compute_field_fdm(param)
    t3 = time.time()
    
    if(t>0):
        t_transport.append((t1-t0)*1e3)
        t_poisson.append((t2-t1)*1e3)
        t_field.append((t3-t2)*1e3)
        
    conv.append(res)
    
    #if(False):#( t>0 and t%50000==0) or res <= param.conv):
    #if(res <= param.conv):
    if(( t>0 and t%10000==0) or res <= param.conv):
        print(t, ' now tot:', np.sum(param.rho), " res= ", res)
        print(f'transport: {(t1-t0)*1e3:.3f} ms, poisson {(t2-t1)*1e3:.3f}ms, field {(t3-t2)*1e3:.3f}ms')
        #plot.show_phi(param)
        #plot.show_convergence(conv)
        #plot.show_rho(param, mag=1)
        #plot.show_rho_projy(param)
        #plot.show_rho_projx(param)
        #plot.show_vel_proj(param)

        #plot.show_Ex_Ey_proj(param)
        #plot.show_phi_proj(param)
        #plot.show_Ex_Ey(param)
        plot.show_evolution(param)
        plot.show_velocity(param)
        if(res <= param.conv):
            print('CONVERGED ! at ', t, "=", t*param.dt, " s")
            #plot.show_rho(geo, param, mag=1)
            #plot.show_rho_proj(param)

            #plot.show_Ex_proj(param)
            #plot.show_Ex_Ey(geo, param)
            break
store.store_SCE(fout, t, param)
#plot.show_time_perf(t_transport, t_poisson, t_field)
#plot.show_convergence(conv)

    #print('l', param.rho[4,5,0], 'r', param.rho[6,5,0])
    #print('u', param.rho[5,6,0], 'd', param.rho[5,4,0])
#div = trans.transport_charge(param)
#param.rho = div
#plot.show_rho(geo, param)


print("Simulation Complete!")
print('Took ', t, ' iterations <=> ', t**param.dt, " s")
print(f'TOOK {time.time()-tstart:.3f} s')

print('\n\n Distortions \n\n')

store.create_tables_distortions(fout, param)
traj = dist.compute_regular_distortions(param)
plot.show_distortions(param)
plot.show_trajectories(traj, param)
store.store_distortions(fout, param)

fout.close()
exit()
















