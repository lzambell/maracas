import jsonc as json

import numpy as np



class flow:
    def __init__(self, params, shape, x, y, z):

        self.Nx, self.Ny, self.Nz = shape
        self.parameters = params
        self.flow_x = np.zeros(shape)
        self.flow_y = np.zeros(shape)
        self.flow_z = np.zeros(shape)
        self.x = x
        self.y = y
        self.z = z
    

        for flow_type, param in self.parameters.items():
            print(flow_type, param)
            if("constant" in flow_type):
                self.make_contant_flow(param)
            elif("vortex" in flow_type):
                self.make_vortex_flow(param)                
            else:
                print("unknow function, sorry")

            

    def make_contant_flow(self, vel):

        self.flow_x += vel[0]
        self.flow_y += vel[1]
        self.flow_z += vel[2]

    def make_vortex_flow(self, p):

        xl, xr, scale_x, yb, yt, scale_y = p
        
        i,j= 0,0
        for ix in self.x:
            j=0
            for iy in self.y:
                if(xl <= ix <= xr and yb <= iy <= yt):
                    self.flow_x[i,j,0] += (1/scale_x) * pow((ix-xl),2)*pow((xr-ix),2)*(iy-yb)*(yt-iy)*(yb+yt-2*iy)
                    self.flow_y[i,j,0] += (1/scale_y) * pow((iy-yb),2)*pow((yt-iy),2)*(ix-xl)*(xr-ix)*(xl+xr-2*ix)
                j +=1
            i += 1
