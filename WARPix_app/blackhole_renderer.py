import os
# Force JAX onto CPU:
os.environ["JAX_PLATFORM_NAME"] = "cpu"        # Tells JAX "use CPU"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"  

import sympy as sp
import jax
import jax.numpy as jnp
from jax import jit
import diffrax
import numpy as np
import matplotlib.pyplot as plt




m, a, q = sp.symbols('m a q')
t_sym, r, th, phi = sp.symbols('t_sym r th phi')
pr, pth, pt, pphi = sp.symbols('pr pth pt pphi')


g_tphi = -2*m*r*a*sp.sin(th)**2/(r**2+(a*sp.cos(th))**2)

g_phiphi = (r**2+a**2+(2*m*r*a**2*sp.sin(th)**2)/(r**2+(a*sp.cos(th))**2))*sp.sin(th)**2

g_tt = (2*m*r)/(r**2+(a*sp.cos(th))**2)-1

g_thth = r**2+(a*sp.cos(th))**2

g_rr = (r**2+(a*sp.cos(th))**2)/(r**2-2*m*r+a**2)
D = g_tphi**2 - g_tt * g_phiphi


E = -pt
L = pphi
H = (1/2) * (
    pr**2 / g_rr
    + pth**2 / g_thth
    - (L**2 * g_tt + 2*E*L * g_tphi + E**2 * g_phiphi) / D
)

@jax.jit
def initial_vector_batch(r0, th0, b_list, al_list, metric_params, E=1.0, inward=True):
    batch_size = b_list.shape[0]
    t0 = jnp.zeros(batch_size)
    L_list = -b_list * E
    vth0_list = al_list / (r0 ** 2)
    m_val = metric_params['m']
    a_val = metric_params['a']
    q_val = 0

    Σ = r0**2 + a_val**2 * jnp.cos(th0)**2
    Δ = r0**2 - 2*m_val*r0 + a_val**2
    h = q_val * ((m_val**3 * r0) / Σ**2)

    g_tt_val = -(1 - (2*m_val*r0)/Σ) * (1 + h)
    g_tphi_val = -((2*m_val*a_val*r0)/Σ) * jnp.sin(th0)**2 * (1 + h)
    g_phiphi_val = (
        (r0**2 + a_val**2 + (2*m_val*a_val**2*r0*jnp.sin(th0)**2)/Σ
         + h * ((a_val**2*(Σ + 2*m_val*r0))/Σ) * jnp.sin(th0)**2)
        * jnp.sin(th0)**2
    )
    g_rr_val = (Σ * (1 + h)) / (Δ + h * a_val**2 * jnp.sin(th0)**2)
    g_thth_val = Σ
    D_val = g_tphi_val**2 - g_tt_val * g_phiphi_val

    E_val = E
    L_val = L_list

    vt0 = (E_val * g_phiphi_val + L_val * g_tphi_val) / D_val
    vth0_sq = vth0_list ** 2
    numerator = (
        (L_val ** 2 * g_tt_val + 2 * E_val * L_val * g_tphi_val +
         E_val ** 2 * g_phiphi_val) / D_val
        - vth0_sq * g_thth_val
    )
    vr0_sq = numerator / g_rr_val
    vr0 = jnp.sqrt(jnp.abs(vr0_sq))
    vr0 = jnp.where(inward, -vr0, vr0)
    pr0 = g_rr_val * vr0
    pth0 = g_thth_val * vth0_list
    pt0 = -E_val * jnp.ones(batch_size)
    pphi0 = L_val
    φ0 = jnp.zeros(batch_size)

    I0 = jnp.zeros(batch_size)  

    y0 = jnp.stack([t0, r0 * jnp.ones(batch_size), th0 * jnp.ones(batch_size),
                    φ0, pr0, pth0, pt0, pphi0, I0], axis=1)
    return y0

import jax.numpy as jnp

@jax.jit
def doppler_width(v0, T, m_a):
    k = 8.617333262145e-5  
    return jnp.sqrt(2 * k * T / m_a)


@jax.jit
def doppler_profile(nu, nu0, delta_nu_D):
    return (1 / (delta_nu_D * jnp.sqrt(jnp.pi))) * jnp.exp(-((nu - nu0) ** 2) / (delta_nu_D ** 2))


@jax.jit
def calculate_azimuthal_velocity(r, theta, a_value, m_value):
    omega=(jnp.sqrt(m_value)/((r*jnp.sin(theta))**(3/2)+a_value*jnp.sqrt(m_value))) 
    
    vphi_fluid = omega
    return vphi_fluid


@jax.jit
def intensity_at_point(points_4_velocity, E0, a_value, m_value):
    λ, t, r, theta, phi, vt, vr, vtheta, vphi = points_4_velocity
    z = r * jnp.cos(theta)
    density = 1 / r ** 2
    #density=  density_BLR_3d(r,theta,phi)
    q_value=0
    Σ = r**2 + a_value**2 * jnp.cos(theta)**2
    Δ = r**2 - 2 * m_value * r + a_value**2
    h = q_value * ((m_value**3 * r) / Σ**2)

    g = jnp.array([
        [(-(1 - (2 * m_value * r) / Σ)) * (1 + h), 0, 0, (-((2 * m_value * a_value * r) / Σ)) * jnp.sin(theta)**2 * (1 + h)],
        [0, (Σ * (1 + h)) / (Δ + h * a_value**2 * jnp.sin(theta)**2), 0, 0],
        [0, 0, Σ, 0],
        [(-((2 * m_value * a_value * r) / Σ)) * jnp.sin(theta)**2 * (1 + h), 0, 0, (r**2 + a_value**2 + (2 * m_value * a_value**2 * r * jnp.sin(theta)**2) / Σ + h * ((a_value**2 * (Σ + 2 * m_value * r)) / Σ) * jnp.sin(theta)**2) * jnp.sin(theta)**2]
    ])

    k_0 = jnp.dot(g, jnp.array([vt, vr, vtheta, vphi]))

    vphi_fluid = calculate_azimuthal_velocity(r, theta, a_value, m_value)
    u_0_candidate = jnp.array([1.0, 0.0, 0.0, vphi_fluid])

    dot_product = jnp.dot(u_0_candidate, jnp.dot(g, u_0_candidate))
    vt_fluid = jnp.sqrt(-1 / dot_product)
    vphi_fluid = vphi_fluid * vt_fluid
    u_0 = jnp.array([vt_fluid, 0.0, 0.0, vphi_fluid])

    k_0_u_0 = jnp.dot(k_0, u_0)
    E_obs = -1 / k_0_u_0
    gamma_inv = 1 / E_obs


    alpha0 = 0.0
    optical_depth = gamma_inv * alpha0


    j0 = density
    intensity = gamma_inv * (j0 / E0**3) * E_obs**3
    ratio = E_obs / E0

    return intensity, ratio
@jax.jit
def eom_system(t, y, args):

    t_var, r_val, th_val, phi_val, pr_val, pth_val, pt_val, pphi_val, I_val = y
    m_val, a_val, E0 = args  


    dt_dλ = -0.5*(4*a_val*jnp.sin(th_val)**2*m_val*pphi_val*r_val/(a_val**2*jnp.cos(th_val)**2 + r_val**2) + 2*jnp.sin(th_val)**2*pt_val*(2*a_val**2*jnp.sin(th_val)**2*m_val*r_val/(a_val**2*jnp.cos(th_val)**2 + r_val**2) + a_val**2 + r_val**2))/(4*a_val**2*jnp.sin(th_val)**4*m_val**2*r_val**2/(a_val**2*jnp.cos(th_val)**2 + r_val**2)**2 - jnp.sin(th_val)**2*(2*m_val*r_val/(a_val**2*jnp.cos(th_val)**2 + r_val**2) - 1)*(2*a_val**2*jnp.sin(th_val)**2*m_val*r_val/(a_val**2*jnp.cos(th_val)**2 + r_val**2) + a_val**2 + r_val**2))
    dr_dλ = 1.0*pr_val*(a_val**2 - 2*m_val*r_val + r_val**2)/(a_val**2*jnp.cos(th_val)**2 + r_val**2)
    dth_dλ = 1.0*pth_val/(a_val**2*jnp.cos(th_val)**2 + r_val**2)
    dphi_dλ = -0.5*(4*a_val*jnp.sin(th_val)**2*m_val*pt_val*r_val/(a_val**2*jnp.cos(th_val)**2 + r_val**2) + 2*pphi_val*(2*m_val*r_val/(a_val**2*jnp.cos(th_val)**2 + r_val**2) - 1))/(4*a_val**2*jnp.sin(th_val)**4*m_val**2*r_val**2/(a_val**2*jnp.cos(th_val)**2 + r_val**2)**2 - jnp.sin(th_val)**2*(2*m_val*r_val/(a_val**2*jnp.cos(th_val)**2 + r_val**2) - 1)*(2*a_val**2*jnp.sin(th_val)**2*m_val*r_val/(a_val**2*jnp.cos(th_val)**2 + r_val**2) + a_val**2 + r_val**2))
    dpr_dλ = 1.0*pr_val**2*r_val*(a_val**2 - 2*m_val*r_val + r_val**2)/(a_val**2*jnp.cos(th_val)**2 + r_val**2)**2 - 0.5*pr_val**2*(-2*m_val + 2*r_val)/(a_val**2*jnp.cos(th_val)**2 + r_val**2) + 1.0*pth_val**2*r_val/(a_val**2*jnp.cos(th_val)**2 + r_val**2)**2 + 0.5*(-8*a_val*jnp.sin(th_val)**2*m_val*pphi_val*pt_val*r_val**2/(a_val**2*jnp.cos(th_val)**2 + r_val**2)**2 + 4*a_val*jnp.sin(th_val)**2*m_val*pphi_val*pt_val/(a_val**2*jnp.cos(th_val)**2 + r_val**2) + jnp.sin(th_val)**2*pt_val**2*(-4*a_val**2*jnp.sin(th_val)**2*m_val*r_val**2/(a_val**2*jnp.cos(th_val)**2 + r_val**2)**2 + 2*a_val**2*jnp.sin(th_val)**2*m_val/(a_val**2*jnp.cos(th_val)**2 + r_val**2) + 2*r_val) + pphi_val**2*(-4*m_val*r_val**2/(a_val**2*jnp.cos(th_val)**2 + r_val**2)**2 + 2*m_val/(a_val**2*jnp.cos(th_val)**2 + r_val**2)))/(4*a_val**2*jnp.sin(th_val)**4*m_val**2*r_val**2/(a_val**2*jnp.cos(th_val)**2 + r_val**2)**2 - jnp.sin(th_val)**2*(2*m_val*r_val/(a_val**2*jnp.cos(th_val)**2 + r_val**2) - 1)*(2*a_val**2*jnp.sin(th_val)**2*m_val*r_val/(a_val**2*jnp.cos(th_val)**2 + r_val**2) + a_val**2 + r_val**2)) + 0.5*(4*a_val*jnp.sin(th_val)**2*m_val*pphi_val*pt_val*r_val/(a_val**2*jnp.cos(th_val)**2 + r_val**2) + jnp.sin(th_val)**2*pt_val**2*(2*a_val**2*jnp.sin(th_val)**2*m_val*r_val/(a_val**2*jnp.cos(th_val)**2 + r_val**2) + a_val**2 + r_val**2) + pphi_val**2*(2*m_val*r_val/(a_val**2*jnp.cos(th_val)**2 + r_val**2) - 1))*(16*a_val**2*jnp.sin(th_val)**4*m_val**2*r_val**3/(a_val**2*jnp.cos(th_val)**2 + r_val**2)**3 - 8*a_val**2*jnp.sin(th_val)**4*m_val**2*r_val/(a_val**2*jnp.cos(th_val)**2 + r_val**2)**2 + jnp.sin(th_val)**2*(2*m_val*r_val/(a_val**2*jnp.cos(th_val)**2 + r_val**2) - 1)*(-4*a_val**2*jnp.sin(th_val)**2*m_val*r_val**2/(a_val**2*jnp.cos(th_val)**2 + r_val**2)**2 + 2*a_val**2*jnp.sin(th_val)**2*m_val/(a_val**2*jnp.cos(th_val)**2 + r_val**2) + 2*r_val) + jnp.sin(th_val)**2*(-4*m_val*r_val**2/(a_val**2*jnp.cos(th_val)**2 + r_val**2)**2 + 2*m_val/(a_val**2*jnp.cos(th_val)**2 + r_val**2))*(2*a_val**2*jnp.sin(th_val)**2*m_val*r_val/(a_val**2*jnp.cos(th_val)**2 + r_val**2) + a_val**2 + r_val**2))/(4*a_val**2*jnp.sin(th_val)**4*m_val**2*r_val**2/(a_val**2*jnp.cos(th_val)**2 + r_val**2)**2 - jnp.sin(th_val)**2*(2*m_val*r_val/(a_val**2*jnp.cos(th_val)**2 + r_val**2) - 1)*(2*a_val**2*jnp.sin(th_val)**2*m_val*r_val/(a_val**2*jnp.cos(th_val)**2 + r_val**2) + a_val**2 + r_val**2))**2
    dpth_dλ = -1.0*a_val**2*jnp.cos(th_val)*jnp.sin(th_val)*pr_val**2*(a_val**2 - 2*m_val*r_val + r_val**2)/(a_val**2*jnp.cos(th_val)**2 + r_val**2)**2 - 1.0*a_val**2*jnp.cos(th_val)*jnp.sin(th_val)*pth_val**2/(a_val**2*jnp.cos(th_val)**2 + r_val**2)**2 + 0.5*(8*a_val**3*jnp.cos(th_val)*jnp.sin(th_val)**3*m_val*pphi_val*pt_val*r_val/(a_val**2*jnp.cos(th_val)**2 + r_val**2)**2 + 4*a_val**2*jnp.cos(th_val)*jnp.sin(th_val)*m_val*pphi_val**2*r_val/(a_val**2*jnp.cos(th_val)**2 + r_val**2)**2 + 8*a_val*jnp.cos(th_val)*jnp.sin(th_val)*m_val*pphi_val*pt_val*r_val/(a_val**2*jnp.cos(th_val)**2 + r_val**2) + 2*jnp.cos(th_val)*jnp.sin(th_val)*pt_val**2*(2*a_val**2*jnp.sin(th_val)**2*m_val*r_val/(a_val**2*jnp.cos(th_val)**2 + r_val**2) + a_val**2 + r_val**2) + jnp.sin(th_val)**2*pt_val**2*(4*a_val**4*jnp.cos(th_val)*jnp.sin(th_val)**3*m_val*r_val/(a_val**2*jnp.cos(th_val)**2 + r_val**2)**2 + 4*a_val**2*jnp.cos(th_val)*jnp.sin(th_val)*m_val*r_val/(a_val**2*jnp.cos(th_val)**2 + r_val**2)))/(4*a_val**2*jnp.sin(th_val)**4*m_val**2*r_val**2/(a_val**2*jnp.cos(th_val)**2 + r_val**2)**2 - jnp.sin(th_val)**2*(2*m_val*r_val/(a_val**2*jnp.cos(th_val)**2 + r_val**2) - 1)*(2*a_val**2*jnp.sin(th_val)**2*m_val*r_val/(a_val**2*jnp.cos(th_val)**2 + r_val**2) + a_val**2 + r_val**2)) + 0.5*(4*a_val*jnp.sin(th_val)**2*m_val*pphi_val*pt_val*r_val/(a_val**2*jnp.cos(th_val)**2 + r_val**2) + jnp.sin(th_val)**2*pt_val**2*(2*a_val**2*jnp.sin(th_val)**2*m_val*r_val/(a_val**2*jnp.cos(th_val)**2 + r_val**2) + a_val**2 + r_val**2) + pphi_val**2*(2*m_val*r_val/(a_val**2*jnp.cos(th_val)**2 + r_val**2) - 1))*(-16*a_val**4*jnp.cos(th_val)*jnp.sin(th_val)**5*m_val**2*r_val**2/(a_val**2*jnp.cos(th_val)**2 + r_val**2)**3 - 16*a_val**2*jnp.cos(th_val)*jnp.sin(th_val)**3*m_val**2*r_val**2/(a_val**2*jnp.cos(th_val)**2 + r_val**2)**2 + 4*a_val**2*jnp.cos(th_val)*jnp.sin(th_val)**3*m_val*r_val*(2*a_val**2*jnp.sin(th_val)**2*m_val*r_val/(a_val**2*jnp.cos(th_val)**2 + r_val**2) + a_val**2 + r_val**2)/(a_val**2*jnp.cos(th_val)**2 + r_val**2)**2 + 2*jnp.cos(th_val)*jnp.sin(th_val)*(2*m_val*r_val/(a_val**2*jnp.cos(th_val)**2 + r_val**2) - 1)*(2*a_val**2*jnp.sin(th_val)**2*m_val*r_val/(a_val**2*jnp.cos(th_val)**2 + r_val**2) + a_val**2 + r_val**2) + jnp.sin(th_val)**2*(2*m_val*r_val/(a_val**2*jnp.cos(th_val)**2 + r_val**2) - 1)*(4*a_val**4*jnp.cos(th_val)*jnp.sin(th_val)**3*m_val*r_val/(a_val**2*jnp.cos(th_val)**2 + r_val**2)**2 + 4*a_val**2*jnp.cos(th_val)*jnp.sin(th_val)*m_val*r_val/(a_val**2*jnp.cos(th_val)**2 + r_val**2)))/(4*a_val**2*jnp.sin(th_val)**4*m_val**2*r_val**2/(a_val**2*jnp.cos(th_val)**2 + r_val**2)**2 - jnp.sin(th_val)**2*(2*m_val*r_val/(a_val**2*jnp.cos(th_val)**2 + r_val**2) - 1)*(2*a_val**2*jnp.sin(th_val)**2*m_val*r_val/(a_val**2*jnp.cos(th_val)**2 + r_val**2) + a_val**2 + r_val**2))**2
    dpt_dλ = 0.0  
    dpphi_dλ = 0.0  


    vt = dt_dλ
    vr = dr_dλ
    vtheta = dth_dλ
    vphi = dphi_dλ


    points_4_velocity = [t, t_var, r_val, th_val, phi_val, vt, vr, vtheta, vphi]


    intensity_increment = jax.lax.cond(
        disk_intersection(y),
        lambda _: intensity_at_point(points_4_velocity, E0, a_val, m_val)[0],
        lambda _: 0.0,
        operand=None
    )


    dI_dλ = intensity_increment


    dy_dt = jnp.array([
        dt_dλ, dr_dλ, dth_dλ, dphi_dλ,
        dpr_dλ, dpth_dλ, dpt_dλ, dpphi_dλ, dI_dλ
    ])
    return dy_dt



@jax.jit
def disk_intersection(y):
    r_val = y[1]
    th_val = y[2]
    phi_val = y[3]
    z = r_val * jnp.cos(th_val)
    rho = r_val*jnp.sin(th_val)
    in_rho = jnp.logical_and(7 < rho, rho < 12)
    in_z = jnp.logical_and(-1< z, z < 1)
    return jnp.logical_and(in_rho, in_z)

metric_params = {'m': 1.0, 'a': 0.998}
args_array = jnp.array([metric_params['m'], metric_params['a']])
E0 = 1.0  
args_array = jnp.array([metric_params['m'], metric_params['a'], E0])

ode_term = diffrax.ODETerm(eom_system)

def event_function(t, y, args):

    t_var, r_val, th_val, phi_val, pr_val, pth_val, pt_val, pphi_val, I_val = y
    m_val, a_val, E0 = args


    epsilon = 0.001
    condition_r = r_val - (1.064+ epsilon)
    return condition_r 





solver = diffrax.Tsit5()
#solver = diffrax.Kvaerno5()
#solver=diffrax.Dopri8()
#solver=diffrax.Heun()



def event_function_wrapper(t, y, args, **kwargs):
    return event_function(t, y, args)

event = diffrax.Event(cond_fn=event_function_wrapper)

jax.config.update("jax_enable_x64", True)
controller = diffrax.PIDController(rtol=1e-5, atol=1e-6,dtmin=0.01, force_dtmin=True)  # Built-in controller
#controller = diffrax.ConstantStepSize ()
#solver = diffrax.SemiImplicitEuler()
from functools import partial

def solve_single_trajectory(y0, args, ode_term, solver):

    
    solution = diffrax.diffeqsolve(
        terms=ode_term,
        solver=solver,
        t0=t0,
        t1=t1,
        dt0=dt0,
        y0=y0,
        args=args,
        max_steps=1500000,
        event=event,
        stepsize_controller=controller,
        saveat=diffrax.SaveAt(t1=True)
    )
    final_state = solution.ys[-1]
    final_time = solution.ts[-1]
    return final_state, final_time

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.cm as cm
from PIL import Image
import diffrax

# Assuming these functions and objects are already defined:
# - initial_vector_batch(...)
# - solve_trajectories (your vmapped diffrax integration) 
# - ode_term, solver, t0, t1, dt0, controller, event
# - Any other helper functions (e.g., disk_intersection, etc.)
# They are available in your module context.

def render_image(a_val,th0):
    """
    Render a low-resolution (resolution x resolution) black hole image.
    
    Parameters:
      a_val      : Black hole spin (a parameter in metric_params)
      resolution : Grid resolution along one dimension (default: 100)
      r0         : Initial radial coordinate
      th0        : Initial theta value (in radians)
      E          : Energy parameter (default: 1.0)
      inward     : Boolean for inward or outward trajectories
      
    Returns:
      An RGB image (as a NumPy array) scaled up for a pixel-art aesthetic.
    """
    E=1.0
    inward=True
    r0=100
    resolution=100
    th0=(th0*jnp.pi / 2)/90
    # Define metric parameters and other constants
    metric_params = {'m': 1.0, 'a': a_val}
    args_array = jnp.array([metric_params['m'], metric_params['a'], E])
    
    # Create a grid of impact parameters b and alpha (α)
    # (Adjust these limits as needed for your visualization)
    b_values = jnp.linspace(-20, 20, resolution)
    al_values = jnp.linspace(-20, 20, resolution)
    b_grid, al_grid = jnp.meshgrid(b_values, al_values, indexing='ij')
    
    # Flatten the grid to create a list of initial conditions
    b_list = b_grid.ravel()
    al_list = al_grid.ravel()
    
    # Use your initial_vector_batch to build initial conditions for each ray.
    y0s = initial_vector_batch(r0, th0, b_list, al_list, metric_params, E=E, inward=inward)
    
    # Build an argument array for each ray (tile args_array)
    argss = jnp.tile(args_array, (b_list.shape[0], 1))
    
    # Run the simulation for all rays using your vmapped integration
    # (solve_trajectories is assumed to be a jax.vmap-ed function on your solve_single_trajectory_jit)
    final_results = solve_trajectories(y0s, argss, ode_term, solver)
    final_states, final_times = final_results
    # Assume the intensity is stored as the last element in the state vector
    final_intensities = final_states[:, -1]
    intensities_np = np.array(final_intensities).reshape((resolution, resolution))

    # Set a threshold below which values are forced to zero
    threshold = 1e-7
    intensities_np = np.where(intensities_np < threshold, 0.0, intensities_np)

    # Normalize the intensity values (so they lie in [0, 1])
    norm = intensities_np - intensities_np.min()
    if norm.max() > 0:
        norm = norm / norm.max()
    else:
        norm = intensities_np  # Avoid division by zero
    

    # Apply the 'hot' colormap
    import matplotlib.cm as cm
    cmap = cm.get_cmap("hot")
    image_rgba = cmap(norm)

    # Force any pixel that is below threshold (i.e., norm is 0) to be pure black.
    # This is applied elementwise on the RGBA array.
    image_rgba[norm < threshold] = [0, 0, 0, 1]

    # Drop the alpha channel and convert to 8-bit RGB
    image_rgb = (image_rgba[..., :3] * 255).astype(np.uint8)
    # Transpose if needed (if your image orientation is off)
    image_rgb = image_rgb.transpose((1, 0, 2))

    # Upscale the image using nearest-neighbor interpolation for that pixel-art look.
    from PIL import Image
    scale_factor = 4  # adjust as needed
    img = Image.fromarray(image_rgb)
    img = img.resize((resolution * scale_factor, resolution * scale_factor), Image.NEAREST)
    return np.array(img)


solve_single_trajectory_jit = jax.jit(
    solve_single_trajectory,
    static_argnames=('ode_term', 'solver')
)
# vectorize using vmap
solve_trajectories = jax.vmap(
    solve_single_trajectory_jit,
    in_axes=(0, 0, None, None)
)

t0 = 0.0
t1 = 300
dt0 = 0.025



