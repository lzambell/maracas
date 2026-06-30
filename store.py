from tables import *

class parameters(IsDescription):
    dim = UInt8Col()
    Lx = Float64Col()
    Ly = Float64Col()
    Lz = Float64Col()

    Nx = UInt32Col()
    Ny = UInt32Col()
    Nz = UInt32Col()

    Nx_path = UInt32Col()
    Ny_path = UInt32Col()
    Nz_path = UInt32Col()

    xmin = Float64Col()
    xmax = Float64Col()
    ymin = Float64Col()
    ymax = Float64Col()
    zmin = Float64Col()
    zmax = Float64Col()
    
    dx = Float64Col()
    dy = Float64Col()
    dz = Float64Col()
    
    dx_path = Float64Col()
    dy_path = Float64Col()
    dz_path = Float64Col()

    coll_xmin = Float64Col()
    coll_xmax = Float64Col()
    coll_ymin = Float64Col()
    coll_ymax = Float64Col()
    coll_zmin = Float64Col()
    coll_zmax = Float64Col()

    dt =  Float64Col()
    
    mu = Float64Col()
    D  = Float64Col()
    S  = Float64Col()

    rho0 = Float64Col()
    T    = Float64Col()
    E0   = Float64Col()
        
    alpha = Float64Col()

    
def create_tables(h5file, param):
    if(param.geo.dim == 1):
        create_tables_1D(h5file, param)
    elif(param.geo.dim == 2):
        create_tables_2D(h5file, param)
    else:
        create_tables_3D(h5file, param)

def create_tables_1D(h5file, param):
    table = h5file.create_table("/", 'parameters', parameters, 'parameters')

    # Build dynamic description
    desc = {
        "n_iter": Int64Col(),
        "rho": Float64Col(shape=(param.geo.Nx)),
        "Ex":  Float64Col(shape=(param.geo.Nx+1)),
        "flow_x":Float64Col(shape=(param.geo.Nx+1)),

    }
    SCE_1D = type("SCE_1D", (IsDescription,), desc)
    table = h5file.create_table("/", 'SCE_1D', SCE_1D, "SCE_1D")

 


def create_tables_2D(h5file, param):
    table = h5file.create_table("/", 'parameters', parameters, 'parameters')

    # Build dynamic description
    desc = {
        "n_iter": Int64Col(),
        "rho": Float64Col(shape=(param.geo.Nx, param.geo.Ny)),
        "Ex":  Float64Col(shape=(param.geo.Nx+1, param.geo.Ny+1)),
        "Ey":  Float64Col(shape=(param.geo.Nx+1, param.geo.Ny+1)),
        "flow_x":  Float64Col(shape=(param.geo.Nx+1, param.geo.Ny+1)),
        "flow_y":  Float64Col(shape=(param.geo.Nx+1, param.geo.Ny+1)),
        
    }
    SCE_2D = type("SCE_2D", (IsDescription,), desc)    
    table = h5file.create_table("/", 'SCE_2D', SCE_2D, "SCE_2D")


def create_tables_3D(h5file, param):
    table = h5file.create_table("/", 'parameters', parameters, 'parameters')

    desc = {
        "n_iter": Int64Col(),
        "rho": Float64Col(shape=(param.geo.Nx, param.geo.Ny, param.geo.Nz)),
        "Ex":  Float64Col(shape=(param.geo.Nx+1, param.geo.Ny+1, param.geo.Nz+1)),
        "Ey":  Float64Col(shape=(param.geo.Nx+1, param.geo.Ny+1, param.geo.Nz+1)),
        "Ez":  Float64Col(shape=(param.geo.Nx+1, param.geo.Ny+1, param.geo.Nz+1)),
        "flow_x":  Float64Col(shape=(param.geo.Nx+1, param.geo.Ny+1, param.geo.Nz+1)),
        "flow_y":  Float64Col(shape=(param.geo.Nx+1, param.geo.Ny+1, param.geo.Nz+1)),
        "flow_z":  Float64Col(shape=(param.geo.Nx+1, param.geo.Ny+1, param.geo.Nz+1)),
        
    }
    SCE_3D = type("SCE_3D", (IsDescription,), desc)
    table = h5file.create_table("/", 'SCE_3D', SCE_3D, "SCE_3D")




def create_tables_merge(h5file, param):
    if(param.geo.dim == 1):
        create_tables_1D_merge(h5file, param)
    elif(param.geo.dim == 2):
        create_tables_2D_merge(h5file, param)
    else:
        create_tables_3D_merge(h5file, param)
   
def create_tables_1D_merge(h5file, param):
    table = h5file.create_table("/", 'parameters', parameters, 'parameters')

    # Build dynamic description
    desc = {
        "n_iter": Int64Col(),
        "rho": Float64Col(shape=(param.geo.Nx+1)),
        "Ex":  Float64Col(shape=(param.geo.Nx+2)),
        "flow_x":Float64Col(shape=(param.geo.Nx+2)),

    }
    SCE_1D = type("SCE_1D", (IsDescription,), desc)
    table = h5file.create_table("/", 'SCE_1D', SCE_1D, "SCE_1D")
    
def create_tables_2D_merge(h5file, param):
    table = h5file.create_table("/", 'parameters', parameters, 'parameters')

    # Build dynamic description
    desc = {
        "n_iter": Int64Col(),
        "rho": Float64Col(shape=(param.geo.Nx+1, param.geo.Ny)),
        "Ex":  Float64Col(shape=(param.geo.Nx+2, param.geo.Ny+1)),
        "Ey":  Float64Col(shape=(param.geo.Nx+2, param.geo.Ny+1)),
        "flow_x":  Float64Col(shape=(param.geo.Nx+2, param.geo.Ny+1)),
        "flow_y":  Float64Col(shape=(param.geo.Nx+2, param.geo.Ny+1)),
        
    }
    SCE_2D = type("SCE_2D", (IsDescription,), desc)    
    table = h5file.create_table("/", 'SCE_2D', SCE_2D, "SCE_2D")


    
def create_tables_3D_merge(h5file, param):
    table = h5file.create_table("/", 'parameters', parameters, 'parameters')

    desc = {
        "n_iter": Int64Col(),
        "rho": Float64Col(shape=(param.geo.Nx+1, param.geo.Ny, param.geo.Nz)),
        "Ex":  Float64Col(shape=(param.geo.Nx+2, param.geo.Ny+1, param.geo.Nz+1)),
        "Ey":  Float64Col(shape=(param.geo.Nx+2, param.geo.Ny+1, param.geo.Nz+1)),
        "Ez":  Float64Col(shape=(param.geo.Nx+2, param.geo.Ny+1, param.geo.Nz+1)),
        "flow_x":  Float64Col(shape=(param.geo.Nx+2, param.geo.Ny+1, param.geo.Nz+1)),
        "flow_y":  Float64Col(shape=(param.geo.Nx+2, param.geo.Ny+1, param.geo.Nz+1)),
        "flow_z":  Float64Col(shape=(param.geo.Nx+2, param.geo.Ny+1, param.geo.Nz+1)),
        
    }
    SCE_3D = type("SCE_3D", (IsDescription,), desc)
    table = h5file.create_table("/", 'SCE_3D', SCE_3D, "SCE_3D")




def store_parameters(h5file, param):
    pm = h5file.root.parameters.row
    
    pm['dim'] = param.geo.dim
    pm['Lx'] = param.geo.Lx
    pm['Ly'] = param.geo.Ly
    pm['Lz'] = param.geo.Lz

    pm['Nx'] = param.geo.Nx
    pm['Ny'] = param.geo.Ny
    pm['Nz'] = param.geo.Nz

    
    pm['Nx_path'] = param.geo.Nx_path
    pm['Ny_path'] = param.geo.Ny_path
    pm['Nz_path'] = param.geo.Nz_path

    pm['dx'] = param.geo.dx
    pm['dy'] = param.geo.dy
    pm['dz'] = param.geo.dz

    pm['dx_path'] = param.geo.dx_path
    pm['dy_path'] = param.geo.dy_path
    pm['dz_path'] = param.geo.dz_path

    pm['xmin'] = param.geo.xmin
    pm['ymin'] = param.geo.ymin
    pm['zmin'] = param.geo.zmin

    pm['xmax'] = param.geo.xmax
    pm['ymax'] = param.geo.ymax
    pm['zmax'] = param.geo.zmax


    pm['coll_xmin'] = param.geo.coll_xmin
    pm['coll_ymin'] = param.geo.coll_ymin
    pm['coll_zmin'] = param.geo.coll_zmin

    pm['coll_xmax'] = param.geo.coll_xmax
    pm['coll_ymax'] = param.geo.coll_ymax
    pm['coll_zmax'] = param.geo.coll_zmax

    
    pm['dt'] = param.dt

    pm['mu'] = param.mu
    pm['D']  = param.D
    pm['S']  = param.S

    pm['E0'] = param.E0
    
    pm['alpha'] = param.alpha
    
    pm['rho0'] = param.rho0
    pm['T'] = param.T
    
    pm.append()

def store_SCE(h5file, niter, param):
    if(param.geo.dim == 1):
        store_SCE_1D(h5file, niter, param)
    elif(param.geo.dim == 2):
        store_SCE_2D(h5file, niter, param)
    else:
        store_SCE_3D(h5file, niter, param)

def store_SCE_1D(h5file, niter, param):
    res = h5file.root.SCE_1D.row
    res['n_iter'] = niter

    res['rho'] = param.rho[:,0,0]
    res['Ex'] = param.Ex[:,0,0]
    res['flow_x'] = param.flow_x[:,0,0]
    res.append()

    
def store_SCE_2D(h5file, niter, param):
    res = h5file.root.SCE_2D.row
    res['n_iter'] = niter

    res['rho'] = param.rho[:,:,0]
    res['Ex'] = param.Ex[:,:,0]
    res['Ey'] = param.Ey[:,:,0]
    res['flow_x'] = param.flow_x[:,:,0]
    res['flow_y'] = param.flow_y[:,:,0]
    
    res.append()

def store_SCE_3D(h5file, niter, param):
    res = h5file.root.SCE_3D.row
    res['n_iter'] = niter

    res['rho'] = param.rho
    res['Ex'] = param.Ex
    res['Ey'] = param.Ey
    res['Ez'] = param.Ez
    res['flow_x'] = param.flow_x
    res['flow_y'] = param.flow_y
    res['flow_z'] = param.flow_z
    
    res.append()


def create_tables_distortions(h5file, param):
    if(param.geo.dim == 1):
        pass
    elif(param.geo.dim == 2):
        create_tables_distortions_2D(h5file, param)
    else:
        create_tables_distortions_3D(h5file, param)

        
def create_tables_distortions_2D(h5file, param):
    desc = {                
        "forward_delta_x":  Float64Col(shape=(param.geo.Nx_path, param.geo.Ny_path)),
        "forward_delta_y":  Float64Col(shape=(param.geo.Nx_path, param.geo.Ny_path)),
        "backward_delta_x":  Float64Col(shape=(param.geo.Nx_path, param.geo.Ny_path)),
        "backward_delta_y":  Float64Col(shape=(param.geo.Nx_path, param.geo.Ny_path)),
        
    }
    dist_2D = type("dist_2D", (IsDescription,), desc)    
    table = h5file.create_table("/", 'dist_2D', dist_2D, "dist_2D")


def create_tables_distortions_3D(h5file, param):
    desc = {                
        "forward_delta_x":  Float64Col(shape=(param.geo.Nx_path, param.geo.Ny_path, param.geo.Nz_path)),
        "forward_delta_y":  Float64Col(shape=(param.geo.Nx_path, param.geo.Ny_path, param.geo.Nz_path)),
        "forward_delta_z":  Float64Col(shape=(param.geo.Nx_path, param.geo.Ny_path, param.geo.Nz_path)),        
        "backward_delta_x":  Float64Col(shape=(param.geo.Nx_path, param.geo.Ny_path, param.geo.Nz_path)),
        "backward_delta_y":  Float64Col(shape=(param.geo.Nx_path, param.geo.Ny_path, param.geo.Nz_path)),
        "backward_delta_z":  Float64Col(shape=(param.geo.Nx_path, param.geo.Ny_path, param.geo.Nz_path)),        
    }
    dist_3D = type("dist_3D", (IsDescription,), desc)    
    table = h5file.create_table("/", 'dist_3D', dist_3D, "dist_3D")



def store_distortions(h5file, param):
    if(param.geo.dim == 1):
        pass
    elif(param.geo.dim == 2):
        store_distortions_2D(h5file, param)
    else:
        store_distortions_3D(h5file, param)



def store_distortions_2D(h5file, param):
    dd = h5file.root.dist_2D.row
    dd['forward_delta_x'] = param.forward_delta_x[:,:,0]
    dd['forward_delta_y'] = param.forward_delta_y[:,:,0]
    dd['backward_delta_x'] = param.backward_delta_x[:,:,0]
    dd['backward_delta_y'] = param.backward_delta_y[:,:,0]

    
    dd.append()

def store_distortions_3D(h5file, param):
    dd = h5file.root.dist_3D.row
    dd['forward_delta_x'] = param.forward_delta_x
    dd['forward_delta_y'] = param.forward_delta_y
    dd['forward_delta_z'] = param.forward_delta_z
    
    dd['backward_delta_x'] = param.backward_delta_x
    dd['backward_delta_y'] = param.backward_delta_y
    dd['backward_delta_z'] = param.backward_delta_z
    
    dd.append()

