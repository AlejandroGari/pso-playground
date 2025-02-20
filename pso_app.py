import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import streamlit as st
from pso_funcs import *
# from plotPSO import plotPSO_2D
# import optitestfuns

def objective1(point):
  x = point[0]
  y = point[1]
  return x**2 + (y+1)**2 - 5*np.cos(3/2*x+3/2) - 3*np.cos(2*y-3/2)




objective_pointers ={
  'Modified Potential': objective1,
  'Ackley': ackley,
  'Griewangk': griewangk,
  'Rastrigin': rastrigin,
  'Salomon': salomon, 
  'Odd_square': odd_square,
  'Schwefel':schwefel,
  'Rana': rana
}

objective_msg ={
  'Modified Potential': {'name':'Modified Potential',
                         'function': r'$f(x, y) = x^2 + (y+1)^2 - 5\cos( \frac{3}{2}x + \frac{3}{2}) - 3\cos(2y - \frac{3}{2})$',
                         'text': r"The Modified Potential function is a custom mathematical function that combines a quadratic growth term with oscillatory cosine terms. It creates a smooth landscape with multiple local minima and maxima, making it useful for testing optimization algorithms like PSO. The function's structure allows for evaluating how well an algorithm balances exploration (searching widely) and exploitation (refining known good solutions)."}, 
  'Ackley': {'name':'Ackley',
                         'function': r'$f(x) = -20 \exp \left(-0.2 \sqrt{\frac{1}{d} \sum x_i^2} \right) - \exp \left( \frac{1}{d} \sum \cos(2\pi x_i) \right) + 20 + e $',
                         'text': r"The Ackley function is a widely used benchmark function for optimization, known for its nearly flat outer regions and a large number of local minima surrounding a single global minimum. It represents a challenging problem for optimization algorithms due to its steep slopes and oscillatory behavior. The function is commonly used to evaluate how well an algorithm can escape local minima and converge to the global minimum efficiently."},
  'Griewangk': {'name':'Griewangk',
                         'function': r'$ f(x) = \sum \frac{x_i^2}{4000} - \prod \cos\left(\frac{x_i}{\sqrt{i}}\right) + 1 $',
                         'text': r"The Griewangk function is a complex, multimodal function used in optimization studies. It features many local minima but has a well-defined global minimum at the origin. The function includes both a sum-of-squares term and a cosine product term, making it an excellent test for algorithms that need to handle smooth, non-linear landscapes while avoiding premature convergence."},
  'Rastrigin': {'name':'Rastrigin',
                         'function': r'$ f(x) = \sum \left( x_i^2 - 10\cos(2\pi x_i) + 10 \right) $',
                         'text': r"The Rastrigin function is a non-convex optimization problem that is highly multimodal, meaning it has many local minima. Despite this, the global minimum is easy to identify at the origin. The function is useful for testing optimization algorithms because it evaluates their ability to navigate through a highly oscillatory landscape while avoiding getting stuck in local minima."},
  'Salomon': {'name':'Salomon',
                         'function': r'$ f(x) = 1 - \cos(2\pi r) + 0.1r$ , where  $r = \sqrt{\sum x_i^2} $',
                         'text': r"The Salomon function is a radial benchmark function that depends on the Euclidean distance from the origin. It is often used to test optimization algorithms because of its smooth surface and relatively simple global minimum. This function helps evaluate how well an algorithm can handle problems where solutions are distributed in a circular manner around an optimal point."}, 
  'Odd_square': {'name':'Odd_square',
                         'function': '',
                         'text': r"The Odd Square function is a lesser-known or custom function that likely involves squared terms with additional modifications, such as odd-numbered polynomial exponents or asymmetrical behavior. This function is useful for testing optimization algorithms against unique landscapes where standard benchmark functions may not provide adequate complexity."},
  'Schwefel':{'name':'Schwefel',
                         'function': r'$ f(x) = 418.9829d - \sum x_i \sin(\sqrt{|x_i|}) $',
                         'text': r"The Schwefel function is designed to be highly deceptive for optimization algorithms, featuring a complex landscape with many deep local minima. Its global minimum is far from the origin, making it an excellent test case for evaluating the exploration capabilities of algorithms like PSO. Since the function includes sinusoidal terms, it rewards algorithms that effectively balance local refinement with global search."},
  'Rana': {'name':'Rana',
                         'function': r'$$ f(x) = \sum x_i \sin \left(\sqrt{|x_{i+1} - x_i + 1|} \right) \cos \left(\sqrt{|x_{i+1} + x_i + 1|} \right) $$',
                         'text': r"The Rana function is a highly nonlinear and complex optimization function that involves sinusoidal and square root terms. It creates a rugged landscape with numerous local minima, making it particularly challenging for optimization algorithms. This function is used to test an algorithm’s ability to handle irregular and discontinuous search spaces, requiring sophisticated exploration strategies."}
}





@st.cache_data
def run_pso_grapper(objective_name, n_particles=10, omega=0.3, phi_p=0.7, phi_g=0.7, n_iterations=50):
  objective = objective_pointers[objective_name]
  (fig_conv, _), (x_particles, y_particles), (u_particles, v_particles) = run_PSO(objective,  n_particles=n_particles, omega=omega, phi_p=phi_p, phi_g=phi_g, n_iterations=n_iterations)
  return (fig_conv, _), (x_particles, y_particles), (u_particles, v_particles)

if not 'pso_params' in st.session_state:
  st.session_state.pso_params = {
    'n_particles':10, 'omega':0.3, 'phi_p':0.7, 'phi_g':0.7, 'n_iterations':50
  }

if not 'objective' in st.session_state:
  st.session_state.objective = 'Modified Potential'



  
  # if run_option:
  #   st.session_state.objective = run_option
 
st.session_state.objective = st.selectbox("Objective function",list(objective_pointers.keys()))
    
function_info = objective_msg[st.session_state.objective] 
st.markdown('### About ' + function_info['name']+ ' function')
if function_info['function']:
  st.markdown('**Formula:**')
  st.markdown(function_info['function'])
st.markdown(function_info['text'])


lo_b = -5 # lower bound
up_b =  5 # upper bound

limits=([lo_b, up_b], # x bounds
        [lo_b, up_b]) # y bounds

x_lo = limits[0][0]
x_up = limits[0][1]
y_lo = limits[1][0]
y_up = limits[1][1]

fig_obj, _ = plotPSO_2D(objective_pointers[st.session_state.objective],  limits)


st.pyplot(fig_obj)



st.markdown('### About PSO')
st.write('Particle Swarm Optimization (PSO) is a population-based optimization algorithm inspired by the collective behavior of swarms, such as birds flocking or fish schooling. It works by initializing a set of candidate solutions (particles) that move through the search space, guided by their own best-known position and the best position found by the swarm. Each particle updates its velocity and position based on two key influences: personal experience (exploitation) and collective knowledge (exploration). This balance allows PSO to efficiently search for optimal solutions in complex, high-dimensional spaces, making it useful for function optimization, machine learning, and engineering problems.')
st.markdown('#### Parameters')
st.markdown("""
            ### **PSO Parameters Explained**  

- **`n_particles`** : The number of particles in the swarm; more particles improve search accuracy but increase computation time.  
- **`omega`** (Inertia Weight) : Controls how much a particle retains its previous velocity; higher values favor exploration, lower values favor exploitation.  
- **`phi_p`** (Personal Best Weight) : Determines how strongly a particle is influenced by its own best-known position.  
- **`phi_g`** (Global Best Weight) : Determines how strongly a particle is influenced by the best-known position of the entire swarm.  
- **`n_iterations`** : The number of iterations the swarm runs; more iterations allow better convergence but increase runtime.   
            """)

col1, col2, col3 = st.columns(3)

with st.container():
  st.session_state.pso_params['n_particles'] = col1.number_input('N. Particles', min_value=10, max_value=200,value=10)
  st.session_state.pso_params['omega'] = col2.number_input('Inertia', min_value=0.1, max_value=1.2,value=0.3)
  st.session_state.pso_params['phi_p'] = col3.number_input('Personal best weight', min_value=0.1, max_value=2.5,value=0.7)
  st.session_state.pso_params['phi_g'] = col1.number_input('Global best weight', min_value=0.1, max_value=2.5,value=0.7)
  st.session_state.pso_params['n_iterations'] = col2.number_input('N. Iterations', min_value=10, max_value=500,value=50)

(fig_conv, _), (x_particles, y_particles), (u_particles, v_particles) = run_pso_grapper(st.session_state.objective,**st.session_state.pso_params)

st.pyplot(fig_conv)

def plotPSO_iter(i=0): #iteration
    """Visualization of particles and obj. function"""
        
    fig, (ax1, ax2) = plotPSO_2D(objective_pointers[st.session_state.objective], limits,
               particles_xy=(x_particles[i, :],y_particles[i, :]),
               particles_uv=(u_particles[i, :],v_particles[i, :]))
    return fig, (ax1, ax2)

with st.container():
  iter_slider = st.slider('Iteration', max_value=49, min_value=0, value=0)
  fig_iter, (_, _) = plotPSO_iter(iter_slider)
  st.pyplot(fig_iter)