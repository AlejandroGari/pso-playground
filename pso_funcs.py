import numpy as np
import matplotlib.pyplot as plt

plt.style.use('bmh')


def plotPSO_2D(function, limits=([-5,5],[-5,5]),
               particles_xy=([],[]), particles_uv=([],[]),
               n_points=100, *arg):
    """Creates a figure of 1x2 with a 3D projection representation of a 2D function and a its projection
    
    Params:
        function: a 2D or nD objective function
        limits: define the bounds of the function
        particles_xy a tuple contatining 2 lists with the x and y coordinate of the particles
        particles_xy a tuple contatining 2 lists with the u and v velocities of the particles
        n_points: number of points where the function is evaluated to be plotted, the bigger the finner"""
    

    # Grid points 
    x_lo = limits[0][0]
    x_up = limits[0][1]
    y_lo = limits[1][0]
    y_up = limits[1][1]
    
    assert x_lo<x_up, "Unbound x limits, the first value of the list needs to be higher"
    assert y_lo<y_up, "Unbound x limits, the first value of the list needs to be higher"
                                 
    x = np.linspace(x_lo, x_up, n_points) # x coordinates of the grid
    y = np.linspace(y_lo, y_up, n_points) # y coordinates of the grid

    XX, YY = np.meshgrid(x,y)
    ZZ = np.zeros_like(XX)
    
    for i in range(n_points):
        for j in range(n_points):
            ZZ[i,j] = function((XX[i,j], YY[i,j]))
    
    fig = plt.figure(figsize=(12,4))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')

    ax1.plot_surface(XX,YY,ZZ,
                    rstride=3, cstride=3, alpha=0.4,
                    cmap=plt.cm.viridis, zorder=1)
        
    z_cut_plane = 0 
    
    ax1.set_xlabel('$x$')
    ax1.set_ylabel('$y$')
    ax1.set_zlabel('$z$')
    
    ax1.set_title(function.__name__)

    # Projection of function
    z_proj = ax1.contourf(XX,YY,ZZ,
                              zdir='z', offset=z_cut_plane,
                              cmap=plt.cm.viridis, zorder=1)
    
    # Particle points
    x_particles = particles_xy[0]
    y_particles = particles_xy[1]
    
    # Particle velocities
    u_particles = particles_uv[0]
    v_particles = particles_uv[1]
    
    assert len(x_particles) == len(y_particles), "Tuple with arrays containing particle coordinates are different dimmension"
    assert len(u_particles) == len(v_particles), "Tuple with arrays containing particle velocities are different dimmension"
    
    n_particles = len(x_particles)
    n_velocities = len(u_particles)
    
    if n_particles>=1:
        z_particles = np.zeros(n_particles)
    
        for i in range(n_particles):
            z_particles[i] = function((x_particles[i],y_particles[i]))

        # Plot particles over the function 
        ax1.scatter(x_particles, y_particles, z_particles,
               s=50, c='magenta',
               depthshade=False, zorder=1000)

        z_particles_projection = z_cut_plane*np.ones(n_particles)

        # Plot particles below the function (projection)
        ax1.scatter(x_particles, y_particles, z_particles_projection,
               s=50, c='red',
               depthshade=False, zorder=1000)
    
    
    # 2D projection (right figure)
    ax2 = fig.add_subplot(1, 2, 2)
    
    # Projection of function
    cf2d = ax2.contourf(XX,YY,ZZ,
                 zdir='z', offset=z_cut_plane,
                 cmap=plt.cm.viridis, zorder=1)
    
    # Particles (2D)
    if n_particles>=1:
        ax2.scatter(x_particles, y_particles,
               s=50, c='red', zorder=2)
        
        if n_velocities>=1:
            ax2.quiver(x_particles,y_particles,u_particles,v_particles,
                      angles='xy', scale_units='xy', scale=1)

            tag_particles = range(n_particles)

            for j, txt in enumerate(tag_particles):
                ax2.annotate(txt, (x_particles[j],y_particles[j]), zorder=3)
    
    
    ax2.set_title('xy plane')
    fig.colorbar(cf2d, shrink=1)
    
    #plt.savefig(function.__name__+'_2D', bbox_inches='tight')
    #plt.show()


  
    return fig, (ax1, ax2)

def plotPSO_1D(function, limits=([-5,5]), particles_coordinates=([]), particles_velocities=([]), n_points=100, *arg):  
    """Returns and shows a figure of a 2D representation of a 1D function
    
    Params:
        function: a 2D or nD objective function
        limits: define the bounds of the function
        particles_coordinates: a tuple contatining 2 lists with the x and y coordinate of the particles
        particles_velocities: a tuple contatining 2 lists with the u and v velocities of the particles
        n_points: number of points where the function is evaluated to be plotted, the bigger the finner"""
    
    # Grid points 
    x_lo = limits[0]
    x_up = limits[1]                         
                              
    x = np.linspace(x_lo, x_up, n_points) # x coordinates of the grid
    z = np.zeros(n_points)
   
    for i in range(n_points):
        z[i] = function(x[i])
    
    fig = plt.figure()
    ax = fig.add_subplot(111) # 111 stands for subplot(nrows, ncols, plot_number) 
    ax.plot(x,z, zorder=1)
    
    particles_coordinates = np.array(particles_coordinates)
    particles_velocities = np.array(particles_velocities)
    
    assert particles_coordinates.ndim <=1, \
    "Arrays containing particle coordinates have more than 1 dimmension"
    
    if particles_coordinates.shape[0] is not 0: 
        x_particles = particles_coordinates
        n_particles = x_particles.shape[0]

        z_particles = np.zeros(n_particles)


        for i in range(n_particles):
            z_particles[i] = function(x_particles[i])

        # Plot particles over the function
        ax.scatter(x_particles, z_particles,
               s=50, c='red', zorder=2)
        
        if particles_velocities.shape[0] is not 0:  
            u_particles = particles_velocities
            
            n_velocities = u_particles.shape[0]
            
            v_particles = np.zeros(n_particles)
    
            ax1.quiver(x_particles,z_particles,u_particles,v_particles,
                      angles='xy', scale_units='xy', scale=1)

            tag_particles = range(n_particles)

            for j, txt in enumerate(tag_particles):
                ax1.annotate(txt, (x_particles[j,i],z_particles[j,i]))
    
    
    
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')

    ax.set_title(function.__name__)
    
    #plt.savefig(function.__name__+'_1D', bbox_inches='tight')
    plt.show()
    
    return fig, ax


def ackley(x):
    """Ackley n-dimensional function

    Params:
    x =  numpy array or list containing the independent variables
    returns y = objective function value
    
    Best solution:
    f(x_i*) = y = 0  (i dimensions)
    x_i* = 0
    
    -30 <= x_i <= 30
    """
    

    x = np.array(x)  # converts list to numpy array
    n = x.size  # n-dimensions of the vector

    y = -20 * np.exp(-0.2 * (1 / n * np.sum(x ** 2)) ** 0.5) + \
        -np.exp(1 / n * np.sum(np.cos(2 * np.pi * x))) + 20 + np.exp(1)

    return y


def griewangk(x):
    """Griewank n-dimensional function

    Params:
    x =  numpy array or list containing the independent variables
    returns y = objective function value
    
    Best solution:
    f(x_i*) = y = 0  (i dimensions)
    x_i* = 0
    
    -100 <= x_i <= 100
    """

    x = np.array(x)  # converts list to numpy array
    n = x.size  # n-dimensions of the vector
    j = np.arange(n)
    
    y = 1/4000 * np.sum(x**2) - np.prod(np.cos(x/(j + 1)**0.5)) + 1

    return y

def rastrigin(x):
    """Rastrigin n-dimensional function

    Params:
    x =  numpy array or list containing the independent variables
    returns y = objective function value
    
    Best solution:
    f(x_i*) = y = 0  (i dimensions)
    x_i* = 0
    
    -5.12 <= x_i <= 5.12
    """

    x = np.array(x)  # converts list to numpy array
    n = x.size  # n-dimensions of the vector
    
    y = np.sum(x**2 - 10*np.cos(2*np.pi*x)+10)

    return y

def salomon(x):
    """Salomon n-dimensional function

    Params:
    x =  numpy array or list containing the independent variables
    returns y = objective function value
    
    Best solution:
    f(x_i*) = y = 0  (i dimensions)
    x_i* = 0
    
    -100 <= x_i <= 100
    """

    x = np.array(x)  # converts list to numpy array
    n = x.size  # n-dimensions of the vector
    
    x_norm = np.sqrt(np.sum(x**2))
    
    y = -np.cos(2*np.pi*x_norm) + 0.1*x_norm+1

    return y

def odd_square(x):
    """Whitley n-dimensional function

    Params:
    x =  numpy array or list containing the independent variables
    returns y = objective function value
    
    Best solution:
    f(x_i*) = y = -1.14383  (i dimensions)
    x_i* = many solutions near b
    
    -5*pi <= x_i <= 5*pi
    """
    
    x = np.array(x)  # converts list to numpy array
    n = x.size  # n-dimensions of the vector
    
    assert n<=10, "Error: more than 10 dimensions were given, you need to modify function params to run"
    b = np.array([1, 1.3, 0.8, -0.4, -1.3, 1.6, -0.2, -0.6, 0.5, 1.4,
                  1, 1.3, 0.8, -0.4, -1.3, 1.6, -0.2, -0.6, 0.5, 1.4])
    
    b = b[0:n]
    
    d = n*np.max((x-b)**2)
    h = np.sum((x-b)**2)
        
    y = -np.exp(-d/(2*np.pi))*np.cos(np.pi*d)*(1 + (0.02*h)/(d+0.01))

    return y

def schwefel(x):
    """Schwefel n-dimensional function

    Params:
    x =  numpy array or list containing the independent variables
    returns y = objective function value
    
    Best solution:
    f(x_i*) = y = -418.983 (i dimensions)
    x_i* = 420.968746
    
    -500 <= x_i <= 500
    """

    x = np.array(x)  # converts list to numpy array
    n = x.size  # n-dimensions of the vector
    
    y = -1/n*np.sum(x*np.sin(np.sqrt(np.abs(x))))

    return y

def rana(x):
    """Rana n-dimensional function

    Params:
    x =  numpy array or list containing the independent variables
    returns y = objective function value
    
    Best solution:
    f(x_i*) = y = -511.708 (i dimensions)
    x_i* = -512
    
    -512 <= x_i <= 512
    """

    x = np.array(x)  # converts list to numpy array
    n = x.size  # n-dimensions of the vector
    assert n>=2, "Error: Rana function requires at least 2D"
    
    #import pdb; pdb.set_trace()
    
    x_j = x[:-1]
    x_j1 = x[1:]
    alpha = np.sqrt(np.abs(x_j1+1-x_j))
    beta = np.sqrt(np.abs(x_j1+1+x_j))
    
    fo = np.sum(x_j*np.sin(alpha)*np.cos(beta)+x_j1*np.cos(alpha)*np.sin(beta))

    return fo

def run_PSO(objective_f, n_particles=10, omega=0.3, phi_p=0.7, phi_g=0.7, n_iterations=50, lo_b=-5, up_b =  5
            ):
    
    limits=([lo_b, up_b], # x bounds
        [lo_b, up_b])
    """ PSO algorithm to a funcion already defined.
    Params:
        omega = 0.3  # Particle weight (intertial)
        phi_p = 0.7  # particle best weight
        phi_g = 0.7  # global global weight
    
    """   
    # Note: we are using global variables to ease the use of interactive widgets
    # This code will work fine without the global (and actually it will be safer)


    # Initialazing x postion of particles
    
    x_particles = np.zeros((n_iterations, n_particles))
    x_particles[0,:] = np.random.uniform(lo_b, up_b, size=n_particles)

    # Initialazing y postion of particles
    y_particles = np.zeros((n_iterations, n_particles))
    y_particles[0,:] = np.random.uniform(lo_b, up_b, size=n_particles)

    # Initialazing best praticles
    x_best_particles = np.copy(x_particles[0,:])
    y_best_particles = np.copy(y_particles[0,:])
    
    # Calculate Objective function (aka fitness function)
    z_particles = np.zeros((n_iterations, n_particles))

    for i in range(n_particles):
        z_particles[0,i] = objective_f((x_particles[0,i],y_particles[0,i]))

    z_best_global = np.min(z_particles[0,:])
    index_best_global = np.argmin(z_particles[0,:])

    x_best_p_global = x_particles[0,index_best_global]
    y_best_p_global = y_particles[0,index_best_global]

    # Initialazin velocity
    velocity_lo = lo_b-up_b  # [L/iteration]
    velocity_up = up_b-lo_b  # [L/iteration] 

    v_max = 0.07 # [L/iteration]

    u_particles = np.zeros((n_iterations, n_particles))
    u_particles[0,:] = 0.1*np.random.uniform(velocity_lo, velocity_up, size=n_particles)

    v_particles = np.zeros((n_iterations, n_particles))
    v_particles[0,:] = 0.1*np.random.uniform(velocity_lo, velocity_up, size=n_particles)

    

    # PSO STARTS
    iteration = 1
    while iteration <= n_iterations-1:
        for i in range(n_particles):
            x_p = x_particles[iteration-1, i]
            y_p = y_particles[iteration-1, i]

            u_p = u_particles[iteration-1, i]
            v_p = v_particles[iteration-1, i]

            x_best_p = x_best_particles[i]
            y_best_p = y_best_particles[i]

            r_p = np.random.uniform(0, 1)
            r_g = np.random.uniform(0, 1)

            u_p_new = omega*u_p + \
                        phi_p*r_p*(x_best_p-x_p) + \
                        phi_g*r_g*(x_best_p_global-x_p)

            v_p_new = omega*v_p + \
                        phi_p*r_p*(y_best_p-y_p) + \
                        phi_g*r_g*(y_best_p_global-y_p)

            # # Velocity control
            # while not (-v_max <= u_p_new <= v_max):  
            #     u_p_new = 0.9*u_p_new 
            # while not (-v_max <= u_p_new <= v_max):  
            #     u_p_new = 0.9*u_p_new 

            x_p_new = x_p + u_p_new
            y_p_new = y_p + v_p_new


            # Ignore new position if it's out of the domain
            if not ((lo_b <= x_p_new <= up_b) and (lo_b <= y_p_new <= up_b)): 
                x_p_new = x_p 
                y_p_new = y_p 

            x_particles[iteration, i] = x_p_new
            y_particles[iteration, i] = y_p_new

            u_particles[iteration, i] = u_p_new
            v_particles[iteration, i] = v_p_new

            # Evaluation            
            z_p_new = objective_f((x_p_new,  y_p_new))
            z_p_best = objective_f((x_best_p, y_best_p))
            
            z_particles[iteration, i] = z_p_new

            if z_p_new < z_p_best:
                x_best_particles[i] = x_p_new
                y_best_particles[i] = y_p_new

                z_p_best_global = objective_f([x_best_p_global, y_best_p_global])

                if z_p_new < z_p_best_global:
                    x_best_p_global = x_p_new
                    y_best_p_global = y_p_new

        # end while loop particles
        iteration = iteration + 1
        
            
    # Plotting convergence
    z_particles_best_hist = np.min(z_particles, axis=1)
    z_particles_worst_hist = np.max(z_particles, axis=1)

    z_best_global = np.min(z_particles)
    index_best_global = np.argmin(z_particles)


    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(10, 2))
    
    # Grid points 
    x_lo = limits[0][0]
    x_up = limits[0][1]
    y_lo = limits[1][0]
    y_up = limits[1][1]
    
    assert x_lo<x_up, "Unbound x limits, the first value of the list needs to be higher"
    assert y_lo<y_up, "Unbound x limits, the first value of the list needs to be higher"
    
    n_points = 100
                                 
    x = np.linspace(x_lo, x_up, n_points) # x coordinates of the grid
    y = np.linspace(y_lo, y_up, n_points) # y coordinates of the grid

    XX, YY = np.meshgrid(x,y)
    ZZ = np.zeros_like(XX)
    
    for i in range(n_points):
        for j in range(n_points):
            ZZ[i,j] = objective_f((XX[i,j], YY[i,j]))
            
    # Limits of the function being plotted   
    ax1.plot((0,n_iterations),(np.min(ZZ),np.min(ZZ)), '--g', label="min$f(x)$")
    ax1.plot((0,n_iterations),(np.max(ZZ),np.max(ZZ)),'--r',  label="max$f(x)$")

    # Convergence of the best particle and worst particle value
    ax1.plot(np.arange(n_iterations),z_particles_best_hist,'b',  label="$p_{best}$")
    ax1.plot(np.arange(n_iterations),z_particles_worst_hist,'k', label="$p_{worst}$")

    ax1.set_xlim((0,n_iterations))

    ax1.set_ylabel('$f(x)$')
    ax1.set_xlabel('$i$ (iteration)')
    ax1.set_title('Convergence')   

    ax1.legend()
    
    return (fig, ax1), (x_particles, y_particles), (u_particles, v_particles)