import matplotlib.pyplot as plt
import matplotlib as mpl

import numpy as np
import colorcet as cc
import matplotlib.patches as patches
from abc import ABC, abstractmethod


class plotter(ABC):
    
    @abstractmethod
    def show_velocity(self):
        pass
    @abstractmethod
    def show_LAr_flow(self):
        pass
    
    @abstractmethod
    def show_evolution(self):
        pass
    
    @abstractmethod
    def show_projection_along_x(self):
        pass
    @abstractmethod
    def show_projection_along_y(self):
        pass
    @abstractmethod
    def show_projection_along_z(self):
        pass
    
    @abstractmethod
    def show_distortions(self):
        pass
    
    @abstractmethod
    def show_convergence(self):
        pass
    @abstractmethod
    def show_time_performance(self):
        pass



class plotter_2D(plotter):
    def __init__(self, param, out):
        self.dir = param.geo.drift_dir
        self.out = out
        

    def plot_2D(self, ax, X, min_max, xx, yy, cmap):
        if(self.dir == "horizontal"):
            im = ax.imshow(X.T,
                           origin='lower',
                           aspect = 'auto', 
                           interpolation='none',
                           vmin   = min_max[0], 
                           vmax   = min_max[1],
                           extent = xx + yy,
                           cmap = cmap)
        else:
            im = ax.imshow(X,
                           origin='lower',
                           aspect = 'auto', 
                           interpolation='none',
                           vmin   = min_max[0], 
                           vmax   = min_max[1],
                           extent = yy + xx,
                           cmap = cmap)
        return im


    def show_anode_cathode(self, ax, param):
        if(self.dir == 'horizontal'):
            fcn = ax.axvline
        else:
            fcn = ax.axhline
            
        for an in param.geo.anode_xpos:
            fcn(an, c='goldenrod', lw=2)
        for ca in param.geo.cathode_xpos:
            fcn(ca, c='lightsteelblue', lw=2, ls='dashed')


            
    def show_evolution(self, param, mag=1, iteration=-1):
        fig = plt.figure(figsize=(12,4))
        ax_rho = fig.add_subplot(131)
        ax_x = fig.add_subplot(132)
        ax_y = fig.add_subplot(133)


        im = self.plot_2D(ax = ax_rho,
                          X = param.rho[:,:,0]/param.rho0,
                          min_max = [0, param.alpha**2],                          
                          xx = [param.geo.xmin, param.geo.xmax],
                          yy = [param.geo.ymin, param.geo.ymax],
                          cmap = cc.cm.linear_tritanopic_krjcw_5_95_c24_r)
        
    
        
        vx = param.mu * param.Ex[:,:,0].T + param.flow_x[:,:,0].T
        vy = param.mu * param.Ey[:,:,0].T + param.flow_y[:,:,0].T
        vy *= mag
        
        stepx, stepy = 6, 6
        XX, YY = np.meshgrid(param.x_field, param.y_field)
        
        if(self.dir == "vertical"):
            vx,vy = vy,vx
            XX,YY = YY,XX


        ax_rho.quiver(XX[::stepx, ::stepy], YY[::stepx, ::stepy], 
                  vx[::stepx, ::stepy], vy[::stepx, ::stepy],angles='xy', scale_units='xy',
                  color='white', width=0.005, pivot='mid', zorder=100, headlength=2)

        equipot = np.arange(param.min_potential, param.max_potential, 20000)

        CS = ax_rho.contour(XX, YY, param.phi[1:-1,1:-1,0].T, levels=equipot ,colors='khaki', linestyles="solid", linewidths=0.6)
        

        im_x = self.plot_2D(ax = ax_x,
                            X = param.Ex[:,:,0]/param.E0,
                            min_max = [0.75, 1.25],
                            xx = [param.geo.xmin_field, param.geo.xmax_field],
                            yy = [param.geo.ymin_field, param.geo.ymax_field],
                            cmap = cc.cm.CET_CBTD1)

        im_y = self.plot_2D(ax = ax_y,
                            X = param.Ey[:,:,0]/param.E0,
                            min_max = [-0.2, 0.2],
                            xx = [param.geo.xmin_field, param.geo.xmax_field],
                            yy = [param.geo.ymin_field, param.geo.ymax_field],
                            cmap = cc.cm.CET_CBTD1)
                
        

        ax_rho.set_title(r'Ion Density/$\rho_0$')
        ax_x.set_title(r'$E_x/E_0$')
        ax_y.set_title(r'$E_y/E_0$')
        
        fig.colorbar(im,   ax=ax_rho)
        fig.colorbar(im_x, ax=ax_x)
        fig.colorbar(im_y, ax=ax_y)

        for ax in [ax_rho, ax_x, ax_y]:            
            ax.set_xlabel('X (drift) [m]' if self.dir == "horizontal" else 'Y [m]')
            ax.set_ylabel('Y [m]' if self.dir == "horizontal" else 'X (drift) [m]')
            self.show_anode_cathode(ax, param)

            
        fig.tight_layout()
        fig.savefig('results/'+self.out+'_evolution_at_step'+str(iteration)+'.png',dpi=200)
        plt.show()


    def show_LAr_flow(self, param):
        fig = plt.figure(figsize=(4,4))
        ax = fig.add_subplot(111)

        fx = param.flow_x[:,:,0]
        fy = param.flow_y[:,:,0]
        
        stepx, stepy = 6, 6
        XX, YY = np.meshgrid(param.x_field, param.y_field)
        xmin, xmax = param.geo.xmin_field, param.geo.xmax_field
        ymin, ymax = param.geo.ymin_field, param.geo.ymax_field
        
        if(self.dir == "vertical"):
            fx,fy = fy,fx
            XX,YY = YY,XX
            xmin, xmax, ymin, ymax = ymin, ymax, xmin, xmax
            
        ax.quiver(XX[::stepx, ::stepy], YY[::stepx, ::stepy], 
                  fx[::stepx, ::stepy].T, fy[::stepx, ::stepy].T,angles='xy', scale_units='xy',
                  color='tab:cyan')
        
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        ax.set_title('LAr Flow')
        

        ax.set_xlabel('X (drift) [m]' if self.dir == "horizontal" else 'Y [m]')
        ax.set_ylabel('Y [m]' if self.dir == "horizontal" else 'X (drift) [m]')
        self.show_anode_cathode(ax, param)

        
        fig.tight_layout()
        fig.savefig('results/'+self.out+'_LAr_flow.png',dpi=200)
        plt.show()

        
        
    def show_velocity(self, param):

        fig = plt.figure(figsize=(12,4))
        ax_f = fig.add_subplot(131)
        ax_x = fig.add_subplot(132)
        ax_y = fig.add_subplot(133)

        fx = param.flow_x[:,:,0]
        fy = param.flow_y[:,:,0]
        
        vx = param.mu * param.Ex[:,:,0] + param.flow_x[:,:,0]
        vy = param.mu * param.Ey[:,:,0] + param.flow_y[:,:,0]

        
        stepx, stepy = 6, 6
        XX, YY = np.meshgrid(param.x_field, param.y_field)
        xmin, xmax = param.geo.xmin_field, param.geo.xmax_field
        ymin, ymax = param.geo.ymin_field, param.geo.ymax_field
        
        if(self.dir == "vertical"):
            fx,fy = fy,fx
            vx,vy = vy,vx
            XX,YY = YY,XX
            xmin, xmax, ymin, ymax = ymin, ymax, xmin, xmax
            
        f = ax_f.quiver(XX[::stepx, ::stepy], YY[::stepx, ::stepy], 
                        fx[::stepx, ::stepy].T, fy[::stepx, ::stepy].T,angles='xy', scale_units='xy',
                        color='tab:cyan')
        ax_f.quiverkey(f, 0.2, 1.1, U=0.0002, label='LAr Flow', labelpos='E', coordinates='axes')
        v = ax_f.quiver(XX[::stepx, ::stepy], YY[::stepx, ::stepy], 
                        vx[::stepx, ::stepy].T, vy[::stepx, ::stepy].T,angles='xy', scale_units='xy',
                        color='k')
        ax_f.quiverkey(v, 0.6, 1.1, U=0.002, label='Ion flow', labelpos='E', coordinates='axes')

        ax_f.set_xlim(xmin, xmax)
        ax_f.set_ylim(ymin, ymax)


        
        vx = param.mu * param.Ex[:,:,0] + param.flow_x[:,:,0]
        vy = param.mu * param.Ey[:,:,0] + param.flow_y[:,:,0]

        im_x = self.plot_2D(ax = ax_x,
                            X = vx,
                            min_max = [-0.01, 0.01],                          
                            xx = [param.geo.xmin, param.geo.xmax],
                            yy = [param.geo.ymin, param.geo.ymax],
                            cmap = cc.cm.CET_CBTD1)
        im_y = self.plot_2D(ax = ax_y,
                            X = vy,
                            min_max = [-0.01, 0.01],                          
                            xx = [param.geo.xmin, param.geo.xmax],
                            yy = [param.geo.ymin, param.geo.ymax],
                            cmap = cc.cm.CET_CBTD1)



        
        ax_f.set_title('LAr Flow and total velocity')
        ax_x.set_title(r'$v_x$ [m/s]')
        ax_y.set_title(r'$v_y$ [m/s]')
        

        fig.colorbar(im_x, ax=ax_x)
        fig.colorbar(im_y, ax=ax_y)

        for ax in [ax_f, ax_x, ax_y]:            
            ax.set_xlabel('X (drift) [m]' if self.dir == "horizontal" else 'Y [m]')
            ax.set_ylabel('Y [m]' if self.dir == "horizontal" else 'X (drift) [m]')
            self.show_anode_cathode(ax, param)
        fig.tight_layout()
        fig.savefig('results/'+self.out+'_velocities.png',dpi=200)

        plt.show()


    def show_projection_along_y(self, param, what):
        if(what == "rho"):
            data = param.rho
            data_norm = param.rho0
            nx = param.rho.shape[0]
            title='Charge Densities'

        elif(what == "Ex"):
            data = param.Ex
            data_norm = param.E0
            nx = param.Ex.shape[0]
            title='Ex'

        elif(what == "Ey"):
            data = param.Ey
            data_norm = param.E0
            nx = param.Ey.shape[0]
            title='Ey'

        elif(what == "phi"):
            data = param.phi
            data_norm = 1.
            nx = param.phi.shape[0]
            title='Potential'

        else:
            print('... Projection of ', what,' is not recognized. \n---> Choose between rho, Ex, Ey, phi')
            return


        cmap = plt.cm.Blues
        norm = mpl.colors.Normalize(vmin=0, vmax=nx)

        fig = plt.figure(figsize=(8,5))
        ax = fig.add_subplot(111)


        for i in range(0, nx):
            r=i/(nx)
            ax.plot(data[i,:,0]/data_norm, color = cmap(r))#plt.cm.Blues(r))

        ax.set_xlabel('y bin')
        ax.set_title(title)

        # Create a ScalarMappable for the colorbar
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # required for older matplotlib versions

        # Add colorbar
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label("x bin")

        fig.tight_layout()
        fig.savefig('results/'+self.out+what+'_proj_y.png',dpi=200)
        plt.show()

    def show_projection_along_x(self):
        pass
    def show_projection_along_z(self):
        pass
    
    def show_trajectories(self, trajectories, param):
        fig = plt.figure(figsize=(4,4))
        ax = fig.add_subplot(111)
        for t in trajectories:
            traj = np.array([pos for pos in t])
            ax.plot(traj[:, 1], traj[:, 0], c='grey', lw=0.5, alpha=0.6)


        ax.set_xlabel('X (drift) [m]' if self.dir == "horizontal" else 'Y [m]')
        ax.set_ylabel('Y [m]' if self.dir == "horizontal" else 'X (drift) [m]')
        self.show_anode_cathode(ax, param)

        xmin, xmax = param.geo.xmin, param.geo.xmax
        ymin, ymax = param.geo.ymin, param.geo.ymax
        
        if(self.dir == "vertical"):
            xmin, xmax, ymin, ymax = ymin, ymax, xmin, xmax

        
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_title('Electron trajectories')

        fig.tight_layout()
        fig.savefig('results/'+self.out+'_trajectories.png',dpi=200)
        plt.show()

    def show_distortions(self, param):
        fig = plt.figure(figsize=(12,4))
        ax   = fig.add_subplot(131)
        ax_x = fig.add_subplot(132)
        ax_y = fig.add_subplot(133)


        dx = -1.*param.delta_x[:,:,0]
        dy = -1.*param.delta_y[:,:,0]

        
        stepx, stepy = 20,20
        x = np.linspace(param.geo.xmin+param.geo.dx_path/2, param.geo.xmax-param.geo.dx_path/2, param.geo.Nx_path)
        y = np.linspace(param.geo.ymin+param.geo.dy_path/2, param.geo.ymax-param.geo.dy_path/2, param.geo.Ny_path)
        
        XX, YY = np.meshgrid(x, y)
        xmin, xmax = param.geo.xmin, param.geo.xmax
        ymin, ymax = param.geo.ymin, param.geo.ymax
        
        if(self.dir == "vertical"):
            dx,dy = dy,dx
            XX,YY = YY,XX
            xmin, xmax, ymin, ymax = ymin, ymax, xmin, xmax
            
        ax.quiver(XX[::stepx, ::stepy], YY[::stepx, ::stepy], 
                  dx[::stepx, ::stepy].T, dy[::stepx, ::stepy].T,
                  angles='xy', scale_units='xy',units='xy', scale=0.5,
                  color='k')
        
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        im_x = self.plot_2D(ax_x,
                            param.delta_x[:,:,0]*1e2,
                            min_max = [-15, 15],                          
                            xx = [param.geo.xmin, param.geo.xmax],
                            yy = [param.geo.ymin, param.geo.ymax],
                            cmap = cc.cm.CET_CBTD1)

        im_y = self.plot_2D(ax_y,
                            param.delta_y[:,:,0]*1e2,
                            min_max = [-30, 30],                          
                            xx = [param.geo.xmin, param.geo.xmax],
                            yy = [param.geo.ymin, param.geo.ymax],
                            cmap = cc.cm.CET_CBTD1)


        fig.colorbar(im_x, ax=ax_x)
        fig.colorbar(im_y, ax=ax_y)

        ax.set_title(r'Distortions (true to reco)')
        ax_x.set_title(r'$\Delta x$ (true-reco) [cm]')
        ax_y.set_title(r'$\Delta y$ (true-reco) [cm]')

        for ax in [ax, ax_x, ax_y]:            
            ax.set_xlabel('X (drift) [m]' if self.dir == "horizontal" else 'Y [m]')
            ax.set_ylabel('Y [m]' if self.dir == "horizontal" else 'X (drift) [m]')
            self.show_anode_cathode(ax, param)
        fig.tight_layout()
        fig.savefig('results/'+self.out+'_distortions.png',dpi=200)

        plt.show()

    def show_convergence(self, conv):
        fig = plt.figure(figsize=(10,5))
        ax = fig.add_subplot(111)
        ax.plot(conv)
        ax.set_yscale('log')
        ax.set_xlabel('iteration')
        ax.set_ylabel('convergence')
        fig.tight_layout()
        plt.show()
    
    def show_time_performance(self, t_transport, t_poisson, t_field):
        fig = plt.figure(figsize=(10,5))
        ax_a = fig.add_subplot(131)
        ax_b = fig.add_subplot(132)
        ax_c = fig.add_subplot(133)
        
            
        ax_a.hist(t_transport, bins = 50, range = [0, 5], histtype='stepfilled', fc='None', edgecolor='k')
        ax_a.set_xlabel('Transport [ms]')
        
        ax_b.hist(t_poisson, bins = 50, range = [0, 5], histtype='stepfilled', fc='None', edgecolor='k')
        ax_b.set_xlabel('Poisson solve [ms]')
        
        ax_c.hist(t_field, bins = 50, range = [0, 5], histtype='stepfilled', fc='None', edgecolor='k')
        ax_c.set_xlabel('Field computation [ms]')
        fig.tight_layout()
        plt.show()
    



    
def show_fc_potential(param):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(param.field_cage_potential)
    #ax.set_xlim(param.geo.xmin, param.geo.xmax)
    ax.set_title("Degrading potential")
    plt.show()



    
def show_phi(param):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(param.phi[:,:,0].T,#/param.rho0,
                   origin='lower',
                   aspect = 'auto', 
                   interpolation='none',
                   #vmin   = -180000, 
                   #vmax   = 0,
                   extent = [param.geo.xmin_ghost, param.geo.xmax_ghost, param.geo.ymin_ghost, param.geo.ymax_ghost],
                   cmap = cc.cm.linear_tritanopic_krjcw_5_95_c24_r)

    plot_anode_cathode(ax, param)
        
    fig.colorbar(im, ax=ax)
    ax.set_xlabel('X (drift) [m]')
    ax.set_ylabel('Y [m]')
    
    plt.show()



def show_vel_proj(param):
    fig = plt.figure(figsize=(8,5))
    ax = fig.add_subplot(111)

    vx = param.mu * param.Ex + param.flow_x
    vy = param.mu * param.Ey + param.flow_y

    nx = vx.shape[0]
    ny = vx.shape[1]

    for i in range(0, nx, 4):
        r=i/(nx)
        ax.plot(vx[i,:,0], color = plt.cm.Blues(r))#, label="x bin"+str(i))
        ax.plot(vy[i,:,0], color = plt.cm.Reds(r))#, label="x bin"+str(i))

    ax.set_xlabel('ybin')
    ax.set_title('Total velocities')
    plt.show()



    



