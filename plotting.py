import matplotlib.pyplot as plt
import matplotlib as mpl

import numpy as np
import colorcet as cc
import matplotlib.patches as patches
from abc import ABC, abstractmethod
import pyvista as pv

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


    
    def show_convergence(self, conv):
        fig = plt.figure(figsize=(10,4))
        ax = fig.add_subplot(111)
        ax.plot(conv)
        ax.set_yscale('log')
        ax.set_xlabel('iteration')
        ax.set_ylabel('convergence')
        fig.tight_layout()
        plt.show()
    
    def show_time_performance(self, t_transport, t_poisson, t_field):
        fig = plt.figure(figsize=(10,4))
        ax_a = fig.add_subplot(131)
        ax_b = fig.add_subplot(132)
        ax_c = fig.add_subplot(133)
        
        
        ax_a.hist(t_transport, bins = 50, range = [0, 50], histtype='stepfilled', fc='None', edgecolor='k')
        ax_a.set_xlabel('Transport [ms]')
        
        ax_b.hist(t_poisson, bins = 50, range = [0, 50], histtype='stepfilled', fc='None', edgecolor='k')
        ax_b.set_xlabel('Poisson solve [ms]')
        
        ax_c.hist(t_field, bins = 50, range = [0, 50], histtype='stepfilled', fc='None', edgecolor='k')
        ax_c.set_xlabel('Field computation [ms]')
        fig.tight_layout()
        plt.show()






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


    def show_Etot(self, param, iteration=-1):
        fig = plt.figure(figsize=(4,4))
        ax = fig.add_subplot(111)
        Etot = np.sqrt(param.Ex**2 + param.Ey**2)*1e-2
        
        im = self.plot_2D(ax = ax,
                          X = Etot,
                          min_max = [0.75*param.E0*1e-2, 1.25*param.E0*1e-2],                          
                          xx = [param.geo.xmin, param.geo.xmax],
                          yy = [param.geo.ymin, param.geo.ymax],
                          cmap = cc.cm.CET_CBTD1)
        ax.set_title(r'$E_{tot}$ [V/cm]')
        
        fig.colorbar(im,   ax=ax)

        
        ax.set_xlabel('X (drift) [m]' if self.dir == "horizontal" else 'Y [m]')
        ax.set_ylabel('Y [m]' if self.dir == "horizontal" else 'X (drift) [m]')
        self.show_anode_cathode(ax, param)
        fig.tight_layout()
        fig.savefig('results/'+self.out+'_Etot_at_step'+str(iteration)+'.png',dpi=200)
        plt.show()
        
    def show_evolution(self, param, mag=1, iteration=-1):
        fig = plt.figure(figsize=(12,4))
        ax_rho = fig.add_subplot(131)
        ax_x = fig.add_subplot(132)
        ax_y = fig.add_subplot(133)


        im = self.plot_2D(ax = ax_rho,
                          X = param.rho[:,:,0]/param.rho0,
                          min_max = [0, 2*param.alpha**2],                          
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
                            min_max = [0.75, 1.25] if param.geo.drift_forward else [-1.25, -0.75],
                            xx = [param.geo.xmin_field, param.geo.xmax_field],
                            yy = [param.geo.ymin_field, param.geo.ymax_field],
                            cmap = cc.cm.CET_CBTD1 )#if param.geo.drift_forward else cc.cm.CET_CBTD1_r)

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

    

class plotter_3D(plotter):
    def __init__(self, param, out):
        self.dir = param.geo.drift_dir
        self.out = out


    
    def show_velocity(self, param):
        pass
    
    def show_LAr_flow(self, param):
        pass

    def show_Etot(self, param, iteration=-1):
        cmap_field = cc.cm.CET_CBTD1
        
        Etot = np.sqrt(param.Ex**2 + param.Ey**2 + param.Ez**2)*1e-2



        stepx, stepy, stepz = 2, 2, 2
        XX, YY, ZZ = np.meshgrid(param.x_field, param.y_field, param.z_field, indexing = 'ij')


        _XX = XX[::stepx, ::stepy, ::stepz]
        _YY = YY[::stepx, ::stepy, ::stepz]
        _ZZ = ZZ[::stepx, ::stepy, ::stepz]
        
        _Etot = Etot[::stepx, ::stepy, ::stepz]

        

        grid = pv.StructuredGrid(_XX, _YY, _ZZ)
        grid["Etot"] = _Etot.ravel('F')
        
        
        p = pv.Plotter( window_size=[800, 700], border=False)
        
        slices = grid.slice_orthogonal()        
        p.add_mesh(slices, scalars='Etot', cmap=cmap_field, lighting=False, clim=[0.75*param.E0*1e-2, 1.25*param.E0*1e-2],
                   scalar_bar_args={
                       #"title": "(Ex-E0)/E0",
                       "title_font_size": 20,
                       "label_font_size": 16,
                       "color": "black",  # color of text and ticks
                       "vertical": True,  # make it vertical (optional)
                   })



        p.show_bounds(
            grid='back',                  # shows bounding cube
            xtitle='X [m]',                # custom X label
            ytitle='Y [m]',                # custom Y label
            ztitle='Z [m]',                # custom Z label
            location='outer',
            all_edges=True,
            font_size=12,
            color='black',
            show_xaxis=True,
            show_yaxis=True,
            show_zaxis=True,
            show_xlabels=True,
            show_ylabels=True,
            show_zlabels=True
        )
        actor = p.add_title('Etot', color='k', font_size=10)
        p.link_views()
        p.view_isometric()
        
        #p.render()
        p.show(screenshot='results/'+self.out+'_Etot_step'+f'{iteration:05d}'+'.png', auto_close=True)
        p.close()
    
    def show_evolution(self, param, iteration):

        cmap_field = cc.cm.CET_CBTD1
        cmap_phi = cc.cm.linear_tritanopic_krjcw_5_95_c24
        cmap_rho = cc.cm.linear_tritanopic_krjcw_5_95_c24_r

        vx = param.mu * param.Ex + param.flow_x
        vy = param.mu * param.Ey + param.flow_y
        vz = param.mu * param.Ez + param.flow_z

        vy *= 5
        vz *= 5
        if(param.geo.drift_forward):
            Ex = (param.Ex - param.E0)/param.E0
        else:
            Ex = (-1*param.Ex - param.E0)/param.E0
        Ey = param.Ey/param.E0
        Ez = param.Ez/param.E0

        stepx, stepy, stepz = 2, 2, 2

        XX, YY, ZZ = np.meshgrid(param.x, param.y, param.z, indexing = 'ij')


        _XX = XX[::stepx, ::stepy, ::stepz]
        _YY = YY[::stepx, ::stepy, ::stepz]
        _ZZ = ZZ[::stepx, ::stepy, ::stepz]
        _rho = param.rho[::stepx, ::stepy, ::stepz]
        
        grid = pv.StructuredGrid(_XX, _YY, _ZZ)
        grid["rho"] = _rho.ravel('F')

        



        
        XX_f, YY_f, ZZ_f = np.meshgrid(param.x_field, param.y_field, param.z_field, indexing = 'ij')
        stepx, stepy, stepz = 1, 1, 1
        

        _XX_f = XX_f[::stepx, ::stepy, ::stepz]
        _YY_f = YY_f[::stepx, ::stepy, ::stepz]
        _ZZ_f = ZZ_f[::stepx, ::stepy, ::stepz]
        
        _Ex = Ex[::stepx, ::stepy, ::stepz]
        _Ey = Ey[::stepx, ::stepy, ::stepz]
        _Ez = Ez[::stepx, ::stepy, ::stepz]
        
        
        _vx = vx[::stepx, ::stepy, ::stepz]
        _vy = vy[::stepx, ::stepy, ::stepz]
        _vz = vz[::stepx, ::stepy, ::stepz]

        _vtot  = np.column_stack((_vx.ravel('F'), _vy.ravel('F'), _vz.ravel('F')))

        
        
        #_E = np.column_stack((_Ex.ravel('F'), _Ey.ravel('F'), _Ez.ravel('F')))
        

        
        grid_f = pv.StructuredGrid(_XX_f, _YY_f, _ZZ_f)
        grid_f["vlar"] = _vtot
        grid_f["Ex"] = _Ex.ravel('F')
        grid_f["Ey"] = _Ey.ravel('F')
        grid_f["Ez"] = _Ez.ravel('F')



        grid = grid.sample(grid_f)   # add "vlar" interpolated to rho-grid nodes
        
        x_center = param.geo.xmin + (param.geo.xmax - param.geo.xmin)/2.
        y_center = param.geo.ymin + (param.geo.ymax - param.geo.ymin)/2.
        z_center = param.geo.zmin + (param.geo.zmax - param.geo.zmin)/2.

        bounds = grid.bounds
        
        seed = pv.Plane(
            center=(bounds[0]+param.geo.dx/2, y_center, z_center),  # center of the plane at x = 0
            direction=(1.0, 0.0, 0.0),  # normal vector pointing in x-direction
            i_size=param.geo.Lz,  # length in y-direction
            j_size=param.geo.Ly,  # length in z-direction
            i_resolution=20,  # number of seed points along y
            j_resolution=20   # number of seed points along z
        )

        streamlines = grid.streamlines_from_source(
            seed,
            vectors="vlar",
            integrator_type=45,
            max_length=2*param.geo.Lx,
        )
        
        p = pv.Plotter(notebook=0, shape=(1,4), window_size=[2500, 600], border=False)

        p.subplot(0,0)
        slices_rho = grid.slice_orthogonal()

        p.add_mesh(slices_rho, scalars="rho", cmap=cmap_rho, opacity=0.7,lighting=False, 
                   scalar_bar_args={
                       "title": "rho/rho0",
                       "title_font_size": 16,
                       "label_font_size": 12,
                       "color": "black",  # color of text and ticks
                       "vertical": True,  # make it vertical (optional)
                   })
        p.add_mesh(streamlines, line_width=1, color='black')

        p.show_bounds(
            grid='back',                  # shows bounding cube
            xtitle='X [m]',                # custom X label
            ytitle='Y [m]',                # custom Y label
            ztitle='Z [m]',                # custom Z label
            location='outer',
            all_edges=True,
            font_size=12,
            color='black',          # set color so text is visible against background
            show_xaxis=True,
            show_yaxis=True,
            show_zaxis=True,
            show_xlabels=True,
            show_ylabels=True,
        show_zlabels=True

        )
        actor = p.add_title('Charge density and flow stream', color='k', font_size=10)


        p.subplot(0,1)
        
        slices_Ex = grid.slice_orthogonal()        
        p.add_mesh(slices_Ex, scalars='Ex', cmap=cmap_field, lighting=False, clim=[-0.2,0.2],
                   scalar_bar_args={
                       #"title": "(Ex-E0)/E0",
                       "title_font_size": 16,
                       "label_font_size": 12,
                       "color": "black",  # color of text and ticks
                       "vertical": True,  # make it vertical (optional)
                   })



        p.show_bounds(
            grid='back',                  # shows bounding cube
            xtitle='X [m]',                # custom X label
            ytitle='Y [m]',                # custom Y label
            ztitle='Z [m]',                # custom Z label
            location='outer',
            all_edges=True,
            font_size=12,
            color='black',
            show_xaxis=True,
            show_yaxis=True,
            show_zaxis=True,
            show_xlabels=True,
            show_ylabels=True,
            show_zlabels=True
        )
        actor = p.add_title('(Ex-E0)/E0', color='k', font_size=10)


        p.subplot(0,2)
        slices_Ey = grid.slice_orthogonal()        
        p.add_mesh(slices_Ey, scalars='Ey', cmap=cmap_field, lighting=False, clim=[-0.2,0.2],
                   scalar_bar_args={
                       #"title": "Ey/E0",
                       "title_font_size": 16,
                       "label_font_size": 12,
                       "color": "black",  # color of text and ticks
                       "vertical": True,  # make it vertical (optional)
                   })
        
        #p.show_grid()
        p.show_bounds(
            grid='back',                  # shows bounding cube
            xtitle='X [cm]',                # custom X label
            ytitle='Y [cm]',                # custom Y label
            ztitle='Z [cm]',                # custom Z label
            location='outer',
            all_edges=True,
            font_size=12,
            color='black',
            show_xaxis=True,
            show_yaxis=True,
            show_zaxis=True,
            show_xlabels=True,
            show_ylabels=True,
            show_zlabels=True
        )
        actor = p.add_title('Ey/E0', color='k', font_size=10)


        p.subplot(0,3)

        slices_Ez = grid.slice_orthogonal()        
        p.add_mesh(slices_Ez, scalars='Ez', cmap=cmap_field, lighting=False, clim=[-0.2,0.2],
                   scalar_bar_args={
                       #"title": "Ez/E0",
                       "title_font_size": 16,
                       "label_font_size": 12,
                       "color": "black",  # color of text and ticks
                       "vertical": True,  # make it vertical (optional)
                   })

        #p.show_grid()
        p.show_bounds(
            grid='back',                  # shows bounding cube
            xtitle='X [m]',                # custom X label
            ytitle='Y [m]',                # custom Y label
            ztitle='Z [m]',                # custom Z label
            location='outer',
            all_edges=True,
            font_size=12,
            color='black',
            show_xaxis=True,
            show_yaxis=True,
            show_zaxis=True,
            show_xlabels=True,
            show_ylabels=True,
            show_zlabels=True
        )
        actor = p.add_title('Ez/E0', color='k', font_size=10)
        p.link_views()
        p.view_isometric()
        
        #p.render()
        p.show(screenshot='results/'+self.out+'_'+f'{iteration:05d}'+'.png')



    def plot_2D(self, ax, X, min_max, xx, yy, cmap):
        im = ax.imshow(X,
                       origin='lower',
                       aspect = 'auto', 
                       interpolation='none',
                       vmin   = min_max[0], 
                       vmax   = min_max[1],
                       extent = xx + yy,
                       cmap = cmap)
        '''
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
        '''
        return im

    
    def show_slices(self, param, ybin, zbin):
        fig = plt.figure(figsize=(16,8))
        ax_xy_r = fig.add_subplot(241)
        ax_xy_x = fig.add_subplot(242)
        ax_xy_y = fig.add_subplot(243)
        ax_xy_z = fig.add_subplot(244)


        ax_xz_r = fig.add_subplot(245)
        ax_xz_x = fig.add_subplot(246)
        ax_xz_y = fig.add_subplot(247)
        ax_xz_z = fig.add_subplot(248)

        im_xy_r = self.plot_2D(ax = ax_xy_r,
                               X = param.rho[:,:,zbin]/param.rho0,
                               min_max = [0, 2*param.alpha**2],                          
                               xx = [param.geo.xmin_field, param.geo.xmax_field],
                               yy = [param.geo.ymin_field, param.geo.ymax_field],
                               cmap = cc.cm.linear_tritanopic_krjcw_5_95_c24_r)

        im_xy_x = self.plot_2D(ax = ax_xy_x,
                            X = param.Ex[:,:,zbin]/param.E0,
                            min_max = [0.75, 1.25],
                            xx = [param.geo.xmin_field, param.geo.xmax_field],
                            yy = [param.geo.ymin_field, param.geo.ymax_field],
                            cmap = cc.cm.CET_CBTD1)

        im_xy_y = self.plot_2D(ax = ax_xy_y,
                            X = param.Ey[:,:,zbin]/param.E0,
                            min_max = [-0.2, 0.2],
                            xx = [param.geo.xmin_field, param.geo.xmax_field],
                            yy = [param.geo.ymin_field, param.geo.ymax_field],
                            cmap = cc.cm.CET_CBTD1)

        im_xy_z = self.plot_2D(ax = ax_xy_z,
                            X = param.Ez[:,:,zbin]/param.E0,
                            min_max = [-0.2, 0.2],
                            xx = [param.geo.xmin_field, param.geo.xmax_field],
                            yy = [param.geo.ymin_field, param.geo.ymax_field],
                            cmap = cc.cm.CET_CBTD1)

        ax_xy_r.set_title(r'$\rho/\rho_0$ at zbin='+str(zbin))
        ax_xy_x.set_title(r'$E_x/E_0$ at zbin='+str(zbin))
        ax_xy_y.set_title(r'$E_y/E_0$ at zbin='+str(zbin))
        ax_xy_z.set_title(r'$E_z/E_0$ at zbin='+str(zbin))
        

        fig.colorbar(im_xy_r, ax=ax_xy_r)
        fig.colorbar(im_xy_x, ax=ax_xy_x)
        fig.colorbar(im_xy_y, ax=ax_xy_y)
        fig.colorbar(im_xy_z, ax=ax_xy_z)

        for ax in [ax_xy_r, ax_xy_x, ax_xy_y, ax_xy_z]:            
            ax.set_xlabel('X (drift) [m]' if self.dir == "horizontal" else 'Y [m]')
            ax.set_ylabel('Y [m]' if self.dir == "horizontal" else 'X (drift) [m]')
            #self.show_anode_cathode(ax, param)



        im_xz_r = self.plot_2D(ax = ax_xz_r,
                               X = param.rho[:,ybin, :]/param.rho0,
                               min_max = [0, param.alpha**2],                          
                               xx = [param.geo.xmin_field, param.geo.xmax_field],
                               yy = [param.geo.zmin_field, param.geo.zmax_field],
                               cmap = cc.cm.linear_tritanopic_krjcw_5_95_c24_r)

            
        im_xz_x = self.plot_2D(ax = ax_xz_x,
                            X = param.Ex[:,ybin,:]/param.E0,
                            min_max = [0.75, 1.25],
                            xx = [param.geo.xmin_field, param.geo.xmax_field],
                            yy = [param.geo.zmin_field, param.geo.zmax_field],
                            cmap = cc.cm.CET_CBTD1)

        im_xz_y = self.plot_2D(ax = ax_xz_y,
                            X = param.Ey[:,ybin,:]/param.E0,
                            min_max = [-0.2, 0.2],
                            xx = [param.geo.xmin_field, param.geo.xmax_field],
                            yy = [param.geo.zmin_field, param.geo.zmax_field],
                            cmap = cc.cm.CET_CBTD1)

        im_xz_z = self.plot_2D(ax = ax_xz_z,
                            X = param.Ez[:,ybin,:]/param.E0,
                            min_max = [-0.2, 0.2],
                            xx = [param.geo.xmin_field, param.geo.xmax_field],
                            yy = [param.geo.zmin_field, param.geo.zmax_field],
                            cmap = cc.cm.CET_CBTD1)

        ax_xz_r.set_title(r'$\rho/\rho_0$ at ybin='+str(ybin))
        ax_xz_x.set_title(r'$E_x/E_0$ at ybin='+str(ybin))
        ax_xz_y.set_title(r'$E_y/E_0$ at ybin='+str(ybin))
        ax_xz_z.set_title(r'$E_z/E_0$ at ybin='+str(ybin))
        

        fig.colorbar(im_xz_r, ax=ax_xz_r)
        fig.colorbar(im_xz_x, ax=ax_xz_x)
        fig.colorbar(im_xz_y, ax=ax_xz_y)
        fig.colorbar(im_xz_z, ax=ax_xz_z)

        for ax in [ax_xz_r, ax_xz_x, ax_xz_y, ax_xz_z]:            
            ax.set_xlabel('X (drift) [m]' if self.dir == "horizontal" else 'Z [m]')
            ax.set_ylabel('Z [m]' if self.dir == "horizontal" else 'X (drift) [m]')
            #self.show_anode_cathode(ax, param)

            
        fig.tight_layout()
        #fig.savefig('results/'+self.out+'_velocities.png',dpi=200)

        plt.show()

    
    def show_projection_along_x(self):
        pass
    
    def show_projection_along_y(self, param, what, zbin=0):
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
            
        elif(what == "Ez"):
            data = param.Ez
            data_norm = param.E0
            nx = param.Ez.shape[0]
            title='Ez'

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
            ax.plot(data[i,:,zbin]/data_norm, color = cmap(r))

        ax.set_xlabel('y bin')
        ax.set_title(title+' at zbin '+str(zbin))

        # Create a ScalarMappable for the colorbar
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # required for older matplotlib versions

        # Add colorbar
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label("x bin")

        fig.tight_layout()
        fig.savefig('results/'+self.out+what+'_proj_y.png',dpi=200)
        plt.show()

        pass
    
    def show_projection_along_z(self):
        pass
    
    
    def show_distortions(self, param):
        """ one plot for delta x, delta y and delta z """        
        fig = [plt.figure(i, figsize=(12,4)) for i in range(3)]

        ax_yz_xmid = [fig[i].add_subplot(221) for i in range(3)]
        ax_xz_ymid = [fig[i].add_subplot(222) for i in range(3)]
        ax_xy_zmid = [fig[i].add_subplot(223) for i in range(3)]
        ax_yz_xmax = [fig[i].add_subplot(224) for i in range(3)]

        
        bin_xmid = int((param.geo.Nx+1)/2)
        bin_ymid = int((param.geo.Ny+1)/2)
        bin_zmid = int((param.geo.Nz+1)/2)
        print('mid bins x ', bin_xmid, 'y', bin_ymid, 'z', bin_zmid)
        
        for i, delta, name in zip([0, 1, 2], [param.delta_x, param.delta_y, param.delta_z], ["x","y","z"]):
            #delta *= -1.
            

            im_yz_xmid = self.plot_2D(ax_yz_xmid[i],
                            delta[bin_xmid,:,:].T*1e2,
                            min_max = [-15, 15],                          
                            xx = [param.geo.ymin, param.geo.ymax],
                            yy = [param.geo.zmin, param.geo.zmax],
                            cmap = cc.cm.CET_CBTD1)

            im_xz_ymid = self.plot_2D(ax_xz_ymid[i],
                            delta[:,:,-1].T*1e2,
                            min_max = [-15, 15],                          
                            xx = [param.geo.xmin, param.geo.xmax],
                            yy = [param.geo.ymin, param.geo.ymax],
                            cmap = cc.cm.CET_CBTD1)

            im_xy_zmid = self.plot_2D(ax_xy_zmid[i],
                            delta[:,:,bin_zmid].T*1e2,
                            min_max = [-15, 15],                          
                            xx = [param.geo.xmin, param.geo.xmax],
                            yy = [param.geo.ymin, param.geo.ymax],
                            cmap = cc.cm.CET_CBTD1)

            im_yz_xmax = self.plot_2D(ax_yz_xmax[i],
                            delta[-1,:,:].T*1e2,
                            min_max = [-15, 15],                          
                            xx = [param.geo.ymin, param.geo.ymax],
                            yy = [param.geo.zmin, param.geo.zmax],
                            cmap = cc.cm.CET_CBTD1)


            fig[i].colorbar(im_yz_xmid, ax=ax_yz_xmid[i])
            fig[i].colorbar(im_xz_ymid, ax=ax_xz_ymid[i])
            fig[i].colorbar(im_xy_zmid, ax=ax_xy_zmid[i])
            fig[i].colorbar(im_yz_xmax, ax=ax_yz_xmax[i])
            

            ax_yz_xmid[i].set_title(r'$\Delta '+name+'$ (true-reco) [cm] at xmid')
            ax_xz_ymid[i].set_title(r'$\Delta '+name+'$ (true-reco) [cm] at ymid')
            ax_xy_zmid[i].set_title(r'$\Delta '+name+'$ (true-reco) [cm] at zmid')
            ax_yz_xmax[i].set_title(r'$\Delta '+name+'$ (true-reco) [cm] at xmax')


        for axs in [ax_yz_xmid, ax_yz_xmax]:
            for ax in axs:
                ax.set_xlabel('Y [m]')# if self.dir == "horizontal" else 'Y [m]')
                ax.set_ylabel('Z [m]')# if self.dir == "horizontal" else 'X (drift) [m]')
                #self.show_anode_cathode(ax, param)
                
        for axs in [ax_xz_ymid]:#, ax_xz_ymax]:
            for ax in axs:
                ax.set_xlabel('X (drift) [m]')# if self.dir == "horizontal" else 'Y [m]')
                ax.set_ylabel('Z [m]')# if self.dir == "horizontal" else 'X (drift) [m]')
        for axs in [ax_xy_zmid]:#, ax_xz_ymax]:
            for ax in axs:
                ax.set_xlabel('X (drift) [m]')# if self.dir == "horizontal" else 'Y [m]')
                ax.set_ylabel('Y [m]')# if self.dir == "horizontal" else 'X (drift) [m]')

        [f.tight_layout() for f in fig]
        [f.savefig('results/'+self.out+'_distortions_delta_'+name+'.png',dpi=200) for f, name in zip(fig, ["x","y","z"])]

        plt.show()
    
    def show_trajectories(self, trajectories, param):
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111, projection='3d')
        
        for t in trajectories:
            traj = np.array([pos for pos in t])
            #ax.plot(traj[:,2], traj[:, 1], traj[:, 0], c='grey', lw=0.5, alpha=0.6)
            ax.plot(traj[:,0], traj[:, 1], traj[:, 2], c='grey', lw=0.5, alpha=0.6)


        ax.set_xlabel('X (drift) [m]')# if self.dir == "horizontal" else 'Y [m]')
        ax.set_ylabel('Y [m]')# if self.dir == "horizontal" else 'X (drift) [m]')
        ax.set_zlabel('Z [m]')# if self.dir == "horizontal" else 'X (drift) [m]')
        #self.show_anode_cathode(ax, param)

        xmin, xmax = param.geo.xmin, param.geo.xmax
        ymin, ymax = param.geo.ymin, param.geo.ymax
        zmin, zmax = param.geo.zmin, param.geo.zmax
        #if(self.dir == "vertical"):
        #    xmin, xmax, ymin, ymax = ymin, ymax, xmin, xmax

        
        ax.set_xlim3d(xmin, xmax)
        ax.set_ylim3d(ymin, ymax)
        ax.set_zlim3d(zmin, zmax)
        ax.set_title('Electron trajectories')

        fig.tight_layout()
        #fig.savefig('results/'+self.out+'_trajectories.png',dpi=200)
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



    



