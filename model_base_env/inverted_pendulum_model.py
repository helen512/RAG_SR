"""Symbolic model for the Gymnasium `InvertedPendulum-v4` task.

This module provides a lightweight symbolic dynamics model that mirrors the
continuous-control cartpole dynamics used by the MuJoCo inverted pendulum
environment.  The symbolic model is built using CasADi so it can be consumed by
control-barrier-function (CBF) filters or other model-based components without
depending on the upstream `safe-control-gym` package.

The state ordering matches Gymnasium's environment observation:

    X = [cart position, cart velocity, pole angle, pole angular velocity].T

The single control input corresponds to the horizontal cart force.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import casadi as cs
import numpy as np

from model_base_env.symbolic_systems import SymbolicModel


@dataclass
class InvertedPendulumParams:
    """Physical parameters for the inverted pendulum."""

    gravity: float = 9.81
    cart_mass: float = 1.0
    pole_mass: float = 0.1
    pole_length: float = 0.6  # full pole length (m)
    friction_cart: float = 0.0
    max_force: float = 3.0
    x_max: float = 0.8

    @property
    def effective_length(self) -> float:
        """Return half pole length for center-of-mass distance."""

        return self.pole_length / 2.0


def build_inverted_pendulum_symbolic(
    dt: float = 0.02,
    overrides: Optional[Dict[str, float]] = None,
) -> SymbolicModel:
    """Create a symbolic model for the inverted pendulum dynamics.

    Args:
        dt: Integration timestep used for the discrete dynamics model.
        overrides: Optional dictionary overriding base physical parameters.

    Returns:
        A :class:`SymbolicModel` instance exposing continuous and discrete
        dynamics along with quadratic cost helpers.
    """

    params = InvertedPendulumParams()
    if overrides:
        for key, value in overrides.items():
            if not hasattr(params, key):
                raise ValueError(f"Unknown inverted pendulum parameter: {key}")
            setattr(params, key, value)

    # Alias frequently used parameters.
    g = params.gravity
    m_p = params.pole_mass
    m_c = params.cart_mass
    total_mass = m_p + m_c
    length = params.effective_length

    # State and input symbols.
    x = cs.MX.sym("x")
    x_dot = cs.MX.sym("x_dot")
    theta = cs.MX.sym("theta")
    theta_dot = cs.MX.sym("theta_dot")
    X = cs.vertcat(x, x_dot, theta, theta_dot)

    u = cs.MX.sym("u")

    # Continuous-time dynamics derived from the classic cartpole system.
    polemass_length = m_p * length
    sin_theta = cs.sin(theta)
    cos_theta = cs.cos(theta)

    temp = (u + polemass_length * theta_dot**2 * sin_theta) / total_mass
    theta_acc = (g * sin_theta - cos_theta * temp) / (
        length * (4.0 / 3.0 - m_p * cos_theta**2 / total_mass)
    )
    x_acc = temp - polemass_length * theta_acc * cos_theta / total_mass

    X_dot = cs.vertcat(x_dot, x_acc, theta_dot, theta_acc)

    # Observation matches the full state.
    Y = X

    # Quadratic cost placeholders (matrices provided at evaluation time).
    nx = X.size1()
    nu = 1
    Q = cs.MX.sym("Q", nx, nx)
    R = cs.MX.sym("R", nu, nu)
    Xr = cs.MX.sym("Xr", nx, 1)
    Ur = cs.MX.sym("Ur", nu, 1)

    # cost = 0.5 * (X - Xr).T @ Q @ (X - Xr) + 0.5 * (u - Ur).T @ R @ (u - Ur)
    cost = 0

    dynamics = {"dyn_eqn": X_dot, "obs_eqn": Y, "vars": {"X": X, "U": u}}
    cost_dict = {
        "cost_func": cost,
        "vars": {"X": X, "U": u, "Xr": Xr, "Ur": Ur, "Q": Q, "R": R},
    }

    params_dict = {
        "gravity": params.gravity,
        "cart_mass": params.cart_mass,
        "pole_mass": params.pole_mass,
        "pole_length": params.pole_length,
        "max_force": params.max_force,
        "x_max": params.x_max,
        "X_EQ": np.zeros((nx, 1)),
        "U_EQ": np.zeros((nu, 1)),
    }

    return SymbolicModel(dynamics=dynamics, cost=cost_dict, dt=dt, params=params_dict)


