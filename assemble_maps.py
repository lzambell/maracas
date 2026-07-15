import tables as tab
import argparse
import numpy as np
import store as store


class geo:
    def __init__(self, param_a, param_b):
        self.make_assert(param_a, param_b, 'dim')
        
        self.dim = param_a['dim'][0]

        self.Nx = param_a['Nx'][0] + param_b['Nx'][0]

        self.Nx_a = param_a['Nx'][0]
        self.Nx_b = param_b['Nx'][0]
        
        self.make_assert(param_a, param_b, 'Ny')
        self.make_assert(param_a, param_b, 'Nz')

        self.Ny = param_a['Ny'][0]
        self.Nz = param_a['Nz'][0]
        
        self.Lx = param_a['Lx'][0] + param_b['Lx'][0]
        
        self.make_assert(param_a, param_b, 'Ly')
        self.make_assert(param_a, param_b, 'Lz')
        
        self.Ly = param_a['Ly'][0]
        self.Lz = param_a['Lz'][0]

        self.make_assert(param_a, param_b, 'dx')
        self.make_assert(param_a, param_b, 'dy')
        self.make_assert(param_a, param_b, 'dz')
        
        self.dx = param_a['dx'][0]
        self.dy = param_a['dy'][0]
        self.dz = param_a['dz'][0]
        
        self.xmin = min(param_a['xmin'][0], param_b['xmin'][0])
        self.xmax = max(param_a['xmax'][0], param_b['xmax'][0])
        
        self.make_assert(param_a, param_b,'ymin')
        self.make_assert(param_a, param_b,'ymax')
        self.make_assert(param_a, param_b,'zmin')
        self.make_assert(param_a, param_b,'zmax')

        self.ymin = param_a['ymin'][0]
        self.ymax = param_a['ymax'][0]
        self.zmin = param_a['zmin'][0]
        self.zmax = param_a['zmax'][0]

        
        self.Nx_path = param_a['Nx_path'][0] + param_b['Nx_path'][0]

        self.Nx_path_a = param_a['Nx_path'][0]
        self.Nx_path_b = param_b['Nx_path'][0]
        
        self.make_assert(param_a, param_b, 'Ny_path')
        self.make_assert(param_a, param_b, 'Nz_path')

        self.Ny_path = param_a['Ny_path'][0]
        self.Nz_path = param_a['Nz_path'][0]


        
        self.make_assert(param_a, param_b, 'dx_path')
        self.make_assert(param_a, param_b, 'dy_path')
        self.make_assert(param_a, param_b, 'dz_path')
        
        self.dx_path = param_a['dx_path'][0]
        self.dy_path = param_a['dy_path'][0]
        self.dz_path = param_a['dz_path'][0]

        self.make_assert(param_a, param_b,'coll_ymin')
        self.make_assert(param_a, param_b,'coll_ymax')
        self.make_assert(param_a, param_b,'coll_zmin')
        self.make_assert(param_a, param_b,'coll_zmax')

        self.coll_xmin = min(param_a['coll_xmin'][0], param_b['coll_xmin'][0])
        self.coll_xmax = max(param_a['coll_xmax'][0], param_b['coll_xmax'][0])
        self.coll_ymin = param_a['coll_ymin'][0]
        self.coll_ymax = param_a['coll_ymax'][0]
        self.coll_zmin = param_a['coll_zmin'][0]
        self.coll_zmax = param_a['coll_zmax'][0]


        
    def make_assert(self, param_a, param_b, val):
            assert param_a[val][0] == param_b[val][0]

class parameters:
    def __init__(self, f_a, f_b):
        self.f_a = tab.open_file(f_a, 'r')
        self.f_b = tab.open_file(f_b, 'r')

        self.check_order()
        self.read_param()

        self.build_maps()
        
        self.f_a.close()
        self.f_b.close()

    def check_order(self):
        param_a = self.f_a.root.parameters.read()
        param_b = self.f_b.root.parameters.read()
        if(param_a['xmin'][0] > param_b['xmin'][0]):
            print('inverting files')
            self.f_a, self.f_b = self.f_b, self.f_a
            
    def read_param(self):
            
        param_a = self.f_a.root.parameters.read()
        param_b = self.f_b.root.parameters.read()
        print(param_a['xmin'][0] , " and ",  param_b['xmin'][0])
        
        self.geo = geo(param_a, param_b)
        
        self.make_assert(param_a, param_b,'dt')
        self.make_assert(param_a, param_b,'mu')
        self.make_assert(param_a, param_b,'D')
        self.make_assert(param_a, param_b,'S')
        self.make_assert(param_a, param_b,'E0')
        self.make_assert(param_a, param_b,'alpha')
        self.make_assert(param_a, param_b,'rho0')
        self.make_assert(param_a, param_b,'T')
        self.dt    = param_a['dt'][0]
        self.mu    = param_a['mu'][0]
        self.D     = param_a['D'][0]
        self.S     = param_a['S'][0]
        self.E0    = param_a['E0'][0]
        self.alpha = param_a['alpha'][0]
        self.rho0  = param_a['rho0'][0]
        self.T     = param_a['T'][0]
        
    def make_assert(self, param_a, param_b, val):
        assert param_a[val][0] == param_b[val][0]


    def build_maps(self):
        if(self.geo.dim == 1):
            self.build_maps_1D()
        elif(self.geo.dim == 2):
            self.build_maps_2D()
        elif(self.geo.dim == 3):
            self.build_maps_3D()
        else:
            print('dimension ', self.geo.dim, 'is not possible ?')
            exit()
            
    def build_maps_1D(self):
        self.rho = np.zeros((self.geo.Nx))
        self.Ex  = np.zeros((self.geo.Nx+1))
        self.flow_x = np.zeros((self.geo.Nx+1))

    def build_maps_2D(self):
        self.rho = np.zeros((self.geo.Nx,   self.geo.Ny))
        self.Ex  = np.zeros((self.geo.Nx+1, self.geo.Ny+1))
        self.Ey  = np.zeros((self.geo.Nx+1, self.geo.Ny+1))

        self.flow_x = np.zeros((self.geo.Nx+1, self.geo.Ny+1))
        self.flow_y = np.zeros((self.geo.Nx+1, self.geo.Ny+1))
        

    def build_maps_3D(self):
        self.rho = np.zeros((self.geo.Nx+1,   self.geo.Ny,   self.geo.Nz))
        self.Ex  = np.zeros((self.geo.Nx+2, self.geo.Ny+1, self.geo.Nz+1))
        self.Ey  = np.zeros((self.geo.Nx+2, self.geo.Ny+1, self.geo.Nz+1))
        self.Ez  = np.zeros((self.geo.Nx+2, self.geo.Ny+1, self.geo.Nz+1))

        self.flow_x = np.zeros((self.geo.Nx+2, self.geo.Ny+1, self.geo.Nz+1))
        self.flow_y = np.zeros((self.geo.Nx+2, self.geo.Ny+1, self.geo.Nz+1))
        self.flow_z = np.zeros((self.geo.Nx+2, self.geo.Ny+1, self.geo.Nz+1))

        self.forward_delta_x = np.zeros((self.geo.Nx_path, self.geo.Ny_path, self.geo.Nz_path))
        self.forward_delta_y = np.zeros((self.geo.Nx_path, self.geo.Ny_path, self.geo.Nz_path))
        self.forward_delta_z = np.zeros((self.geo.Nx_path, self.geo.Ny_path, self.geo.Nz_path))

        self.backward_delta_x = np.zeros((self.geo.Nx_path, self.geo.Ny_path, self.geo.Nz_path))
        self.backward_delta_y = np.zeros((self.geo.Nx_path, self.geo.Ny_path, self.geo.Nz_path))
        self.backward_delta_z = np.zeros((self.geo.Nx_path, self.geo.Ny_path, self.geo.Nz_path))

        
        maps_a = self.f_a.root.SCE_3D.read()
        maps_b = self.f_b.root.SCE_3D.read()
        
        self.rho[0:self.geo.Nx_a, :,:] = maps_a['rho'][0]
        self.rho[self.geo.Nx_a+1:, :,:]  = maps_b['rho'][0]

        self.Ex[0:self.geo.Nx_a+1, :,:] = maps_a['Ex'][0]
        self.Ex[self.geo.Nx_a+1:, :,:]  = maps_b['Ex'][0]


        #print(self.Ex[:,10,10])
        
        self.Ey[0:self.geo.Nx_a+1, :,:] = maps_a['Ey'][0]
        self.Ey[self.geo.Nx_a+1:, :,:]  = maps_b['Ey'][0]

        self.Ez[0:self.geo.Nx_a+1, :,:] = maps_a['Ez'][0]
        self.Ez[self.geo.Nx_a+1:, :,:]  = maps_b['Ez'][0]

        self.flow_x[0:self.geo.Nx_a+1, :,:] = maps_a['flow_x'][0]
        self.flow_x[self.geo.Nx_a+1:, :,:]  = maps_b['flow_x'][0]

        self.flow_y[0:self.geo.Nx_a+1, :,:] = maps_a['flow_y'][0]
        self.flow_y[self.geo.Nx_a+1:, :,:]  = maps_b['flow_y'][0]

        self.flow_z[0:self.geo.Nx_a+1, :,:] = maps_a['flow_z'][0]
        self.flow_z[self.geo.Nx_a+1:, :,:]  = maps_b['flow_z'][0]

        
        dist_a = self.f_a.root.dist_3D.read()
        dist_b = self.f_b.root.dist_3D.read()

        self.forward_delta_x[0:self.geo.Nx_path_a, :,:] = dist_a['forward_delta_x'][0]
        self.forward_delta_x[self.geo.Nx_path_a:, :,:]  = dist_b['forward_delta_x'][0]

        self.forward_delta_y[0:self.geo.Nx_path_a, :,:] = dist_a['forward_delta_y'][0]
        self.forward_delta_y[self.geo.Nx_path_a:, :,:]  = dist_b['forward_delta_y'][0]

        self.forward_delta_z[0:self.geo.Nx_path_a, :,:] = dist_a['forward_delta_z'][0]
        self.forward_delta_z[self.geo.Nx_path_a:, :,:]  = dist_b['forward_delta_z'][0]

        
        self.backward_delta_x[0:self.geo.Nx_path_a, :,:] = dist_a['backward_delta_x'][0]
        self.backward_delta_x[self.geo.Nx_path_a:, :,:]  = dist_b['backward_delta_x'][0]

        self.backward_delta_y[0:self.geo.Nx_path_a, :,:] = dist_a['backward_delta_y'][0]
        self.backward_delta_y[self.geo.Nx_path_a:, :,:]  = dist_b['backward_delta_y'][0]

        self.backward_delta_z[0:self.geo.Nx_path_a, :,:] = dist_a['backward_delta_z'][0]
        self.backward_delta_z[self.geo.Nx_path_a:, :,:]  = dist_b['backward_delta_z'][0]

        
parser = argparse.ArgumentParser()
parser.add_argument('--inputs', '-i', help='Input HDF5 files to merge', required=True, nargs="+")
parser.add_argument('--out', '-o', help='Output name', required=True)
args = parser.parse_args()

output = args.out
fout = 'results/'+output

inputs = args.inputs
print(inputs)


param = parameters(inputs[0], inputs[1])

print(param.geo.Nx, param.geo.Ny, param.geo.Nz)
print(param.geo.Lx, param.geo.Ly, param.geo.Lz)
print(param.geo.dx, param.geo.dy, param.geo.dz)
print(param.geo.xmin, param.geo.ymin, param.geo.zmin)
print(param.geo.xmax, param.geo.ymax, param.geo.zmax)
print(param.dt, param.mu, param.alpha)


fout = tab.open_file(fout+'.h5', mode="w", title="MARACAS Simulation in "+str(param.geo.dim)+"D")
store.create_tables_merge(fout, param)
store.store_parameters(fout, param)
store.store_SCE(fout, 0, param)

store.create_tables_distortions(fout, param)
store.store_distortions(fout, param)
fout.close()
