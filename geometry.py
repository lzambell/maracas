import jsonc as json
import numpy as np



class geometry:
    def __init__(self, param):
        self.param = param

    def build_geometry(self):
        self.dim = self.param['dimension']
        self.dx =  self.param['dx']
        self.dy =  self.param['dy']
        self.dz =  self.param['dz']
        self.drift_dir = self.param["direction"]
        

        """ simulation delimiter defined by volume  """        
        xx, yy, zz = self.simulation_box(self.param['volume']['box'])            


        """ physical boundaries of the simulation """
        self.xmin, self.xmax = xx[0], xx[1]
        self.ymin, self.ymax = yy[0], yy[1]
        self.zmin, self.zmax = zz[0], zz[1]

        """ field boundaries of the simulation """
        self.xmin_field, self.xmax_field = xx[0]-self.dx/2, xx[1]+self.dx/2
        self.ymin_field, self.ymax_field = yy[0]-self.dy/2, yy[1]+self.dy/2
        self.zmin_field, self.zmax_field = zz[0]-self.dz/2, zz[1]+self.dy/2

        """ ghost boundaries of the simulation """
        self.xmin_ghost, self.xmax_ghost = xx[0]-3*self.dx/2, xx[1]+3*self.dx/2
        self.ymin_ghost, self.ymax_ghost = yy[0]-3*self.dy/2, yy[1]+3*self.dy/2
        self.zmin_ghost, self.zmax_ghost = zz[0]-3*self.dz/2, zz[1]+3*self.dy/2


        
        self.Lx = xx[2]
        self.Ly = yy[2]
        self.Lz = zz[2]

        self.Nx = int(np.floor(self.Lx / self.dx + 1e-9))
        self.Ny = 1
        self.Nz = 1
        
        if(self.dim > 1):
            self.Ny = int(np.floor(self.Ly / self.dy + 1e-9))
            assert self.dx == self.dy, ("binning should be the same in x and y (2D case)")
        if(self.dim > 2):
            self.Nz = int(np.floor(self.Lz / self.dz + 1e-9))
            assert self.dx == self.dz, ("binning should be the same in x, y and z (3D case)")
            
        print('\n---- General Geometry ----')
        print('Simulation in ',self.dim,' dimensions')
        print('Lengthes: Lx:', self.Lx, 'm, Ly:',self.Ly, 'm, Lz:',self.Lz,'m')
        print('Nbins:    Nx:', self.Nx, ', Ny:',self.Ny, ', Nz',self.Nz)
        print('along x from ', self.xmin, 'm to', self.xmax,'m')
        print('along y from ', self.ymin, 'm to',  self.ymax,'m')
        print('along z from ', self.zmin,  'm to', self.zmax,'m')
        print('------------------\n')

        print('-- Geometry for the Field Maps --')
        print('along x from ', self.xmin_field, 'm to', self.xmax_field,'m')
        print('along y from ', self.ymin_field, 'm to', self.ymax_field,'m')
        print('along z from ', self.zmin_field, 'm to', self.zmax_field,'m')

        print('-- Geometry for the Potential Map (with ghost cells) --')
        print('along x from ', self.xmin_ghost, 'm to', self.xmax_ghost,'m')
        print('along y from ', self.ymin_ghost, 'm to', self.ymax_ghost,'m')
        print('along z from ', self.zmin_ghost, 'm to', self.zmax_ghost,'m')


        """ dictionary of 'potential':'x/y/z indices' for constant boundary conditions"""
        self.boundaries = {}

        self.anode_idx = []
        self.cathode_idx = []

        self.anode_xpos = []
        self.cathode_xpos = []
        
        n_anode_planes = len(self.param['anode_plane']['plane'])
        if(n_anode_planes > 1):
            print('Having more than one anode plane in the geometry is a feature not yet implemented, sorry')
            exit()
            
        for i in range(n_anode_planes):
            bound = self.boundary_plane(self.param['anode_plane']['plane'][i])#, "anode_"+str(i))

            potential = self.param['anode_plane']['potential'][i]
            bound['V'] = potential
            
            self.anode_idx.append(bound['x'][0])
            self.anode_xpos.append(self.param['anode_plane']['plane'][i][0])
                
            self.boundaries["anode_"+str(i)] = bound
            if(i==0):
                anode_xpos = self.param['anode_plane']['plane'][i][0]
                anode_V = potential
                



        n_cathode_planes = len(self.param['cathode_plane']['plane'])
        if(n_cathode_planes > 1):
            print('Only one cathode plane can be simulated, please update!')
            exit()
        for i in range(n_cathode_planes):
            bound = self.boundary_plane(self.param['cathode_plane']['plane'][i])

            potential = self.param['cathode_plane']['potential'][i]
            bound['V'] = potential
            self.cathode_idx.append(bound['x'][0])
            self.cathode_xpos.append(self.param['cathode_plane']['plane'][i][0])
            if(i==0):
                cathode_xpos = self.param['cathode_plane']['plane'][i][0]
                cathode_V = potential

            self.boundaries["cathode_"+str(i)] = bound




       
       
        bound = self.boundary_plane(self.param['field_cage']['box'])
        FC_potential = self.param['field_cage']['gradient']
        bound['gradient'] = FC_potential
        self.boundaries["field_cage"] = bound


        
        #print('-->> boundaries')
        #print(self.boundaries)


        self.E0 = (anode_V-cathode_V)/(np.fabs(anode_xpos-cathode_xpos))
        self.L_drift = np.fabs(anode_xpos-cathode_xpos)
        
    def build_distortions(self, dist_param):
        self.ds_path = dist_param['ds']
        self.dx_path = dist_param['dx']
        self.dy_path = dist_param['dy']
        self.dz_path = dist_param['dz']

        self.Nx_path = int(np.floor(self.Lx / self.dx_path + 1e-9))
        self.Ny_path = 1
        self.Nz_path = 1
        
        if(self.dim > 1):
            self.Ny_path = int(np.floor(self.Ly / self.dy_path + 1e-9))

        if(self.dim > 2):
            self.Nz_path = int(np.floor(self.Lz / self.dz_path + 1e-9))


        print('-- Geometry of the Distortion Maps --')
        print('Nx:', self.Nx_path, ' with steps of ', self.dx_path, 'm')
        print('Ny:', self.Ny_path, ' with steps of ', self.dy_path, 'm')
        print('Nz:', self.Nz_path, ' with steps of ', self.dz_path, 'm')
        print('\n')
        
    def boundary_plane(self, plane_param):

        x, y, z, dx, dy, dz = plane_param        
        bound = {'x':(0, 1), 'y':(0, 1), 'z':(0, 1)}
        bound = {'x_ghost':(0, 1), 'y_ghost':(0, 1), 'z_ghost':(0, 1)}
        
        if(x is not None and dx is not None):
            xmin, xmax = min(x, x+dx), max(x, x+dx)
            if(xmin < self.xmin or xmax > self.xmax):
                print('There is a geometry problem along x, please check and try again')
                exit()
            i0 = int((xmin - self.xmin_ghost)/self.dx)
            i1 = int((xmax - self.xmin_ghost)/self.dx)

            bound['x_ghost'] = (i0, i1)
            i0 = int((xmin - self.xmin)/self.dx)
            i1 = int((xmax - self.xmin)/self.dx)

            bound['x'] = (i0, i1)


        if(y is not None and dy is not None):
            ymin, ymax = min(y, y+dy), max(y, y+dy)
            if(ymin < self.ymin or ymax > self.ymax):
                print('There is a geometry problem along y, please check and try again')
                exit()
            i0 = int((ymin - self.ymin_ghost)/self.dy)
            i1 = int((ymax - self.ymin_ghost)/self.dy)

            bound['y_ghost'] = (i0, i1)
            i0 = int((ymin - self.ymin)/self.dy)
            i1 = int((ymax - self.ymin)/self.dy)

            bound['y'] = (i0, i1)

        if(z is not None and dz is not None):
            zmin, zmax = min(z, z+dz), max(z, z+dz)
            if(zmin < self.zmin or zmax > self.zmax):
                print('There is a geometry problem again z, please check and try again')
                exit()
            i0 = int((zmin - self.zmin_ghost)/self.dz)
            i1 = int((zmax - self.zmin_ghost)/self.dz)

            bound['z_ghost'] = (i0, i1)
            i0 = int((zmin - self.zmin)/self.dz)
            i1 = int((zmax - self.zmin)/self.dz)

            bound['z'] = (i0, i1)

        return bound
    
    def simulation_box(self, box_param):
        x, y, z, dx, dy, dz = box_param
        
        xmin, xmax = min(x, x+dx), max(x, x+dx)    
        ymin, ymax = 0,0
        zmin, zmax = 0,0
        
        if(self.dim > 1):
            ymin, ymax = min(y, y+dy), max(y, y+dy)
        
        if(self.dim > 2):
            zmin, zmax = min(z, z+dz), max(z, z+dz)


        Lx = xmax-xmin
        Ly = ymax-ymin
        Lz = zmax-zmin
        
        return (xmin, xmax, Lx), (ymin, ymax, Ly), (zmin, zmax, Lz)

