import math
import os
import xml.etree.ElementTree as etxml
from copy import deepcopy

import casadi as cs
import numpy as np

from model_base_env.symbolic_systems import SymbolicModel

class CartPoleEnvModel():
    def __init__(self, dt=0.02):
        self.dt = dt
        

    def _setup_symbolic(self, prior_prop={}, **kwargs):
        '''Creates symbolic (CasADi) models for dynamics, observation, and cost.

        Args:
            prior_prop (dict): specify the prior inertial prop to use in the symbolic model.
        '''
        length = prior_prop.get('pole_length', self.EFFECTIVE_POLE_LENGTH)
        m = prior_prop.get('pole_mass', self.POLE_MASS)
        M = prior_prop.get('cart_mass', self.CART_MASS)
        Mm, ml = m + M, m * length
        g = self.GRAVITY_ACC
        dt = self.CTRL_TIMESTEP
        # Input variables.
        x = cs.MX.sym('x')
        x_dot = cs.MX.sym('x_dot')
        theta = cs.MX.sym('theta')
        theta_dot = cs.MX.sym('theta_dot')
        X = cs.vertcat(x, x_dot, theta, theta_dot) # State vector
        U = cs.MX.sym('U') 
        nx = 4 # Number of state variables
        nu = 1 # Number of input variables
        # Dynamics.
        temp_factor = (U + ml * theta_dot**2 * cs.sin(theta)) / Mm
        theta_dot_dot = ((g * cs.sin(theta) - cs.cos(theta) * temp_factor) / (length * (4.0 / 3.0 - m * cs.cos(theta)**2 / Mm)))
        X_dot = cs.vertcat(x_dot, temp_factor - ml * theta_dot_dot * cs.cos(theta) / Mm, theta_dot, theta_dot_dot)
        # Observation.
        Y = cs.vertcat(x, x_dot, theta, theta_dot)
        # Define cost (quadratic form).
        Q = cs.MX.sym('Q', nx, nx)
        R = cs.MX.sym('R', nu, nu)
        Xr = cs.MX.sym('Xr', nx, 1)
        Ur = cs.MX.sym('Ur', nu, 1)                                            
        cost_func = 0.5 * (X - Xr).T @ Q @ (X - Xr) + 0.5 * (U - Ur).T @ R @ (U - Ur)
        # Define dynamics and cost dictionaries.
        dynamics = {'dyn_eqn': X_dot, 'obs_eqn': Y, 'vars': {'X': X, 'U': U}}
        cost = {'cost_func': cost_func, 'vars': {'X': X, 'U': U, 'Xr': Xr, 'Ur': Ur, 'Q': Q, 'R': R}}
        # Additional params to cache
        params = {
            # prior inertial properties
            'pole_length': length,
            'pole_mass': m,
            'cart_mass': M,
            # equilibrium point for linearization
            'X_EQ': np.zeros(self.state_dim),
            'U_EQ': np.atleast_2d(self.U_GOAL)[0, :],
        }
        # Setup symbolic model.
        self.symbolic = SymbolicModel(dynamics=dynamics, cost=cost, dt=dt, params=params)

    def _setup_constraints(self):
        '''Creates a list of constraints as an attribute.'''
        self.constraints = None
        self.num_constraints = 0
        if self.CONSTRAINTS is not None:
            self.constraints = create_constraint_list(self.CONSTRAINTS, self.AVAILABLE_CONSTRAINTS, self)
            self.num_constraints = self.constraints.num_constraints

    def get_state_constraint_symbolic_models(self):
        '''Return only the constraints that act on the state.

        Returns:
            symbolic_models (list): A list of the symbolic models of the state constraints.
        '''
        return get_symbolic_constraint_models(self.state_constraints)

    def get_input_constraint_symbolic_models(self):
        '''Return only the constraints that act on the input.

        Returns:
            symbolic_models (list): A list of the symbolic models of the input constraints.
        '''
        return get_symbolic_constraint_models(self.input_constraints)

    def get_input_and_state_constraint_symbolic_models(self):
        '''Return only the constraints that act on both state and inputs simultaneously.

        Returns:
            symbolic_models (list): A list of the symbolic models of the joint state and input constraints.
        '''
        return get_symbolic_constraint_models(self.input_state_constraints)

def get_symbolic_constraint_models(constraint_list):
    '''Create list of symbolic models from list of constraints.

    Args:
        constraint_list (list): A list of constraints.

    Returns:
        symbolic_models (list): A list of the symbolic models of the constraints.
    '''
    symbolic_models = [con.get_symbolic_model() for con in constraint_list]
    return symbolic_models