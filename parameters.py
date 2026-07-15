import numpy as np
import jsonc as json
import lar_flow as lar_flow
import geometry as geometry

class parameters:
    def __init__(self, f_in):
        self.f_in = f_in
        self.geo_param = self.read_input('geometry')
        self.geo = geometry.geometry(self.geo_param)
        self.geo.build_geometry()

        
        self.E0 = self.geo.E0
        
        self.min_potential =  99999999
        self.max_potential = -99999999
        
        self.physics_param = self.read_input('physics')
        self.set_physics_quantities()

        
        self.simu_param = self.read_input('simulation')
        self.set_simulation_quantities()

        self.distortion_param = self.read_input('distortions')
        self.geo.build_distortions(self.distortion_param)
        

        print('--------')
        print('Default drift field  = ', self.E0*1e-2, ' V/cm')
        print('Velocity of the ions = ', self.mu * self.E0, ' m/s')
        print(f'alpha = {self.alpha:.4f}')        
        print('--------')
        
        """ rho (Nx, Ny, Nz), density at the grid center """
        self.rho = np.zeros((self.geo.Nx, self.geo.Ny, self.geo.Nz), dtype=np.float64)

        try:
            rho_param = self.read_input("charge_density_input")
            self.rho = np.load(rho_param['file'])
        except KeyError:    
            """ initialisation of charge density array  """
            rho_start = self.set_initial_density(val_anode=0, val_cathode=self.rho0*self.alpha**2)
            rho_2d    = np.repeat([rho_start], self.geo.Ny, axis=0).T
            self.rho = np.repeat(rho_2d[:, :, None], repeats = self.geo.Nz, axis=2)
            



        
        """ EField & flow are (Nx+1, Ny+1, Nz+1) such that anode(s), cathode, FC are at the FACES of the bins """

        if(self.geo.dim == 1):
            nx, ny, nz = self.geo.Nx, self.geo.Ny, 1
            Nx, Ny, Nz = self.geo.Nx+1, 1, 1
            Npx, Npy, Npz = self.geo.Nx_path, 1, 1
            
            self.x = np.linspace(self.geo.xmin+self.geo.dx/2, self.geo.xmax-self.geo.dx/2, nx)
            self.x_field = np.linspace(self.geo.xmin_field+self.geo.dx/2, self.geo.xmax_field-self.geo.dx/2, nx+1)
            self.x_ghost = np.linspace(self.geo.xmin_ghost+self.geo.dx/2, self.geo.xmax_ghost-self.geo.dx/2, nx+3)

            self.y_field = []
            self.z_field = []
            
        elif(self.geo.dim == 2):

            nx, ny, nz = self.geo.Nx, self.geo.Ny, 1
            Nx, Ny, Nz = self.geo.Nx+1, self.geo.Ny+1, 1
            Npx, Npy, Npz = self.geo.Nx_path, self.geo.Ny_path, 1
            
            self.x = np.linspace(self.geo.xmin+self.geo.dx/2, self.geo.xmax-self.geo.dx/2, nx)
            self.x_field = np.linspace(self.geo.xmin_field+self.geo.dx/2, self.geo.xmax_field-self.geo.dx/2, nx+1)
            self.x_ghost = np.linspace(self.geo.xmin_ghost+self.geo.dx/2, self.geo.xmax_ghost-self.geo.dx/2, nx+3)
            
            self.y = np.linspace(self.geo.ymin, self.geo.ymax, ny)
            self.y_field = np.linspace(self.geo.ymin_field+self.geo.dy/2, self.geo.ymax_field-self.geo.dy/2, ny+1)
            self.y_ghost = np.linspace(self.geo.ymin_ghost+self.geo.dy/2, self.geo.ymax_ghost-self.geo.dy/2, ny+3)

            self.z_field = []

        else:
            nx, ny, nz = self.geo.Nx, self.geo.Ny, self.geo.Nz
            Nx, Ny, Nz = self.geo.Nx+1, self.geo.Ny+1, self.geo.Nz+1
            Npx, Npy, Npz = self.geo.Nx_path, self.geo.Ny_path, self.geo.Nz_path
            
            self.x = np.linspace(self.geo.xmin+self.geo.dx/2, self.geo.xmax-self.geo.dx/2, nx)
            self.x_field = np.linspace(self.geo.xmin_field+self.geo.dx/2, self.geo.xmax_field-self.geo.dx/2, nx+1)
            self.x_ghost = np.linspace(self.geo.xmin_ghost+self.geo.dx/2, self.geo.xmax_ghost-self.geo.dx/2, nx+3)
            
            self.y = np.linspace(self.geo.ymin, self.geo.ymax, ny)
            self.y_field = np.linspace(self.geo.ymin_field+self.geo.dy/2, self.geo.ymax_field-self.geo.dy/2, ny+1)
            self.y_ghost = np.linspace(self.geo.ymin_ghost+self.geo.dy/2, self.geo.ymax_ghost-self.geo.dy/2, ny+3)
            
            self.z = np.linspace(self.geo.zmin, self.geo.zmax, nz)
            self.z_field = np.linspace(self.geo.zmin_field+self.geo.dz/2, self.geo.zmax_field-self.geo.dz/2, nz+1)
            self.z_ghost = np.linspace(self.geo.zmin_ghost+self.geo.dz/2, self.geo.zmax_ghost-self.geo.dz/2, nz+3)

            

        self.Ex = np.zeros((Nx, Ny, Nz), dtype=np.float64)
        self.Ey = np.zeros((Nx, Ny, Nz), dtype=np.float64)
        self.Ez = np.zeros((Nx, Ny, Nz), dtype=np.float64)

        self.forward_delta_x = np.zeros((Npx, Npy, Npz), dtype=np.float64)
        self.forward_delta_y = np.zeros((Npx, Npy, Npz), dtype=np.float64)
        self.forward_delta_z = np.zeros((Npx, Npy, Npz), dtype=np.float64)

        self.backward_delta_x = np.zeros((Npx, Npy, Npz), dtype=np.float64)
        self.backward_delta_y = np.zeros((Npx, Npy, Npz), dtype=np.float64)
        self.backward_delta_z = np.zeros((Npx, Npy, Npz), dtype=np.float64)


        self.dist_param = self.read_input('distortions')
        self.backward_max_iter = self.dist_param['max_iter']
        self.backward_norm_tol = self.dist_param['norm_tol']
        
        lflow = lar_flow.flow(self.read_input('lar_flow'), (Nx, Ny, Nz), self.x_field, self.y_field, self.z_field)
        self.flow_x = lflow.flow_x
        self.flow_y = lflow.flow_y
        self.flow_z = lflow.flow_z

        
        """ Phi is (Nx+3, Ny+3, Nz+3) defined like E, but with one ghost cell layer all around """
        if(self.geo.dim == 1):
            Nx, Ny, Nz = self.geo.Nx+3, 1, 1
        elif(self.geo.dim == 2):
            Nx, Ny, Nz = self.geo.Nx+3, self.geo.Ny+3, 1
        else:
            Nx, Ny, Nz = self.geo.Nx+3, self.geo.Ny+3, self.geo.Nz+3
        
        self.phi = np.zeros((Nx, Ny, Nz), dtype=np.float64)


        
        self.boundary_conditions = {}
        self.boundary_conditions['field_cage'] = self.extract_FC_BC()
        self.boundary_conditions['anode'] = self.extract_plane_BC("anode")
        self.boundary_conditions['cathode'] = self.extract_plane_BC("cathode")
        #print('BC are: ')
        #print(self.boundary_conditions)


        
        self.set_initial_potential()
        self.set_boundary_conditions_with_ghost(self.phi)


    def read_input(self, thing):
        with open(self.f_in,'r') as f:
            return json.load(f)[thing]

    def set_physics_quantities(self):
        self.kB = 1.380649e-23 #J/K
        self.q  = 1.602e-19 #C
        self.epsilon = 1.504 * 8.854e-12 #F/m #FOR LAR!!!


        self.mu = float(self.physics_param['mu'])
        self.T = float(self.physics_param['T'])
        self.S = float(self.physics_param['S'])

        try:
            self.D = float(self.physics_param['D'])
        except KeyError:
            self.D = self.mu * self.kB * self.T / self.q


        self.alpha = (self.geo.L_drift/self.E0)*np.sqrt(self.S/(self.epsilon*self.mu))
        self.rho0 = self.epsilon*self.E0/self.geo.Lx


    def set_simulation_quantities(self):
        self.dt = self.simu_param['dt']
        self.conv_poisson = self.simu_param['conv_poisson']
        self.conv_simu = self.simu_param['conv_simu']
        self.timesteps = self.simu_param['timesteps']
        self.static_simulation = self.simu_param['static']
        if(self.static_simulation == True):
            print(' !! The simulation will be static in time !!\n')



    def extract_FC_BC(self):
        
        
        def set_V(x, i0, i1, V0, V1):
            x0 = self.geo.xmin_ghost + i0*self.geo.dx
            x1 = self.geo.xmin_ghost + i1*self.geo.dx
            if x0 <= x <= x1:
                return V0 + (V1 - V0) * (x - x0) / (x1 - x0)
            else:
                return 0
        
        for plane, bc in self.geo.boundaries.items():
            if("field_cage" not in plane):
                continue

            for name, val in bc.items():
                if(name == "gradient"):
                    if(self.geo.drift_forward):
                        V0, V1 = float(val[0]), float(val[1])
                    else:
                        V1, V0 = float(val[0]), float(val[1])
                
                    if(V0 > self.max_potential):self.max_potential = V0
                    if(V0 < self.min_potential):self.min_potential = V0
                    if(V1 > self.max_potential):self.max_potential = V1
                    if(V1 < self.min_potential):self.min_potential = V1
                    
                elif(name == "x_ghost"):
                    x0, x1 = val
                elif(name == "y_ghost"):
                    y0, y1 = val
                elif(name == "z_ghost"):
                    z0, z1 = val


            the_plane = {}
            the_plane['index'] = (x0, x1, y0, y1, z0, z1)
            nx = x1-x0+1
            gradient = np.zeros(nx)

            
            for i in range(nx):
                x = self.geo.xmin_ghost + (i+x0)*self.geo.dx
                gradient[i] = set_V(x,x0,x1,V0,V1)

            the_plane['gradient'] = gradient


        return the_plane


    def extract_plane_BC(self, name):


        for plane, bc in self.geo.boundaries.items():
            if(name not in plane):
                continue

            for name, val in bc.items():
                if(name == "V"):
                    V = float(val)
                elif(name == "x_ghost"):
                    x0, x1 = val
                elif(name == "y_ghost"):
                    y0, y1 = val
                elif(name == "z_ghost"):
                    z0, z1 = val


            the_plane = {}
            the_plane['index'] = (x0, x1, y0, y1, z0, z1)

            the_plane['potential'] = V



        if V < self.min_potential:
            self.min_potential = V 
        if V > self.max_potential:
            self.max_potential = V 

        return the_plane

    def set_initial_potential(self):
        if(self.geo.dim == 1):
            x0, x1, _,_,_,_ = self.boundary_conditions['field_cage']['index']
            self.phi[x0:x1+1, 0, 0] = self.boundary_conditions['field_cage']['gradient']
            
        elif(self.geo.dim == 2):
            x0, x1, y0, y1, _,_ = self.boundary_conditions['field_cage']['index']
            ny = y1-y0+1
            phi_2d = np.repeat([self.boundary_conditions['field_cage']['gradient']] , ny, axis=0).T
            self.phi[x0:x1+1, y0:y1+1,:] = np.repeat(phi_2d[:,:,None], repeats=self.geo.Nz, axis=2)

            
        elif(self.geo.dim == 3):
            x0, x1, y0, y1, z0, z1 = self.boundary_conditions['field_cage']['index']
            ny = y1-y0+1
            phi_2d = np.repeat([self.boundary_conditions['field_cage']['gradient']] , ny, axis=0).T
            nz = z1-z0+1
            self.phi[x0:x1+1, y0:y1+1, z0:z1+1] = np.repeat(phi_2d[:,:,None], repeats=nz, axis=2)
            
    def set_initial_density(self, val_anode, val_cathode):

        planes = []
        
        for plane, bc in self.geo.boundaries.items():
            if('anode' in plane):
                plane_value = val_anode
                #print("at anode: ", plane_value)
            elif('cathode' in plane):
                plane_value = val_cathode
                #print("at cathode: ", plane_value)
            else:
                continue
            
            for name, val in bc.items():
                if(name == "x"):
                    x0, x1 = val
                    #print('->', x0, x1)                    
                else:
                    continue

            if(x0 == x1 and x0 >= 0):
                planes.append((x0,  plane_value))

            
            
            #planes.append((x0, plane_value))
            
        planes.sort(key=lambda p: p[0]) #sort along x index


        density = np.linspace(planes[0][1], planes[1][1], self.geo.Nx, endpoint=True)
        return density
        



    def set_boundary_conditions_with_ghost(self, X):
        
        if(self.geo.dim == 1):
             x0, x1, _, _, _, _ = self.boundary_conditions['field_cage']['index']
             if(self.geo.drift_forward):
                 X[ 0, :, :]   = 2*self.boundary_conditions['anode']['potential'] - X[ 2, :, :]
                 X[-1, :, :]   = 2*self.boundary_conditions['cathode']['potential']- X[-3, :, :]
             else:
                 X[ 0, :, :]   = 2*self.boundary_conditions['cathode']['potential'] - X[ 2, :, :]
                 X[-1, :, :]   = 2*self.boundary_conditions['anode']['potential']- X[-3, :, :]


             
        elif(self.geo.dim == 2):            
            x0, x1, y0, y1, _,_ = self.boundary_conditions['field_cage']['index']
            
            # set BC at ghost cells for the laplacian stencil 
            X[x0:x1+1, 0, 0] = 2*self.boundary_conditions['field_cage']['gradient'] - X[x0:x1+1, 2,0]
            X[x0:x1+1,-1, 0] = 2*self.boundary_conditions['field_cage']['gradient'] - X[x0:x1+1,-3,0]

            
            if(self.geo.drift_forward):
                #anode
                x0, x1, y0, y1, _,_ =   self.boundary_conditions['anode']['index']
                X[ 0, y0:y1+1, :]   = 2*self.boundary_conditions['anode']['potential'] - X[ 2, y0:y1+1, :]
                #cathode
                x0, x1, y0, y1, _,_ =   self.boundary_conditions['cathode']['index']
                X[-1, y0:y1+1, :]   = 2*self.boundary_conditions['cathode']['potential']- X[-3, y0:y1+1, :]
            else:
                #cathode
                x0, x1, y0, y1, _,_ =   self.boundary_conditions['cathode']['index']
                X[ 0, y0:y1+1, :]   = 2*self.boundary_conditions['cathode']['potential'] - X[ 2, y0:y1+1, :]
                #anode
                x0, x1, y0, y1, _,_ =   self.boundary_conditions['anode']['index']
                X[-1, y0:y1+1, :]   = 2*self.boundary_conditions['anode']['potential']- X[-3, y0:y1+1, :]
                
            '''
            #attempt to study the case when the anode/cathode are small than FC
            #doesnt work well yet ... 
            # NEED SOME THINKING
            if(y0 != 1):
               X[ 0, :y0, :]  = X[ 1, :y0, :]
               X[ -1, :y0, :] = X[ -2, :y0, :]
            if(y1 != self.geo.Ny+1):
               X[ 0, y1:, :]   = X[ 1, y1:, :]
               X[ -1, y1:, :]   = X[ -2, y1:, :]
            '''

               
        elif(self.geo.dim == 3):
            x0, x1, y0, y1, z0, z1 = self.boundary_conditions['field_cage']['index']
            
            # set BC at ghost cells for the laplacian stencil


            ny = y1-y0+1
            nz = z1-z0+1
            bc_xy = np.repeat([self.boundary_conditions['field_cage']['gradient']] , ny, axis=0).T
            bc_xz = np.repeat([self.boundary_conditions['field_cage']['gradient']] , nz, axis=0).T


            
            X[x0:x1+1, 0, z0:z1+1] = 2*bc_xz - X[x0:x1+1, 2, z0:z1+1]
            X[x0:x1+1,-1, z0:z1+1] = 2*bc_xz - X[x0:x1+1,-3, z0:z1+1]
            
            X[x0:x1+1, y0:y1+1,  0] = 2*bc_xy - X[x0:x1+1, y0:y1+1,  2]
            X[x0:x1+1, y0:y1+1, -1] = 2*bc_xy - X[x0:x1+1, y0:y1+1, -3]

            if(self.geo.drift_forward):
                #anode
                x0, x1, y0, y1, z0, z1 =   self.boundary_conditions['anode']['index']
                X[ 0, y0:y1+1, :]      = 2*self.boundary_conditions['anode']['potential'] - X[ 2, y0:y1+1, :]
                X[ 0, :, z0:z0+1]      = 2*self.boundary_conditions['anode']['potential'] - X[ 2, :, z0:z0+1]
                
                #cathode
                x0, x1, y0, y1, z0, z1 =   self.boundary_conditions['cathode']['index']
                X[-1, y0:y1+1, :]   = 2*self.boundary_conditions['cathode']['potential']- X[-3, y0:y1+1, :]
                X[-1, :, z0:z0+1]   = 2*self.boundary_conditions['cathode']['potential']- X[-3, :, z0:z0+1]
            else:
                #anode
                x0, x1, y0, y1, z0, z1 =   self.boundary_conditions['anode']['index']
                X[-1, y0:y1+1, :]   = 2*self.boundary_conditions['anode']['potential']- X[-3, y0:y1+1, :]
                X[-1, :, z0:z0+1]   = 2*self.boundary_conditions['anode']['potential']- X[-3, :, z0:z0+1]

                #cathode
                x0, x1, y0, y1, z0, z1 =   self.boundary_conditions['cathode']['index']
                X[ 0, y0:y1+1, :]      = 2*self.boundary_conditions['cathode']['potential'] - X[ 2, y0:y1+1, :]
                X[ 0, :, z0:z0+1]      = 2*self.boundary_conditions['cathode']['potential'] - X[ 2, :, z0:z0+1]
                

                
    
   
