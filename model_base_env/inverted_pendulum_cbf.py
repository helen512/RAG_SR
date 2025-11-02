"""Control Barrier Function (CBF) safety filter for `InvertedPendulum-v4`.

The filter relies exclusively on the symbolic model defined in
`model_base_env.inverted_pendulum_model` and avoids any dependency on the
`safe-control-gym` package.  It enforces a single cart position constraint
``|x| <= x_max`` by solving a simple 1-D projection problem at each timestep.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import casadi as cs
import numpy as np

from model_base_env.base_safety_filter import BaseSafetyFilter
from model_base_env.inverted_pendulum_model import build_inverted_pendulum_symbolic


@dataclass
class CBFStatistics:
    total_actions: int = 0
    corrected_actions: int = 0
    failed_actions: int = 0
    cumulative_correction: float = 0.0
    max_correction: float = 0.0

    def record(self, delta: float) -> None:
        self.total_actions += 1
        if np.abs(delta) > 0:
            self.corrected_actions += 1
            self.cumulative_correction += np.abs(delta)
            self.max_correction = max(self.max_correction, np.abs(delta))

    @property
    def correction_rate(self) -> float:
        if self.total_actions == 0:
            return 0.0
        return self.corrected_actions / self.total_actions

    @property
    def average_correction(self) -> float:
        if self.corrected_actions == 0:
            return 0.0
        return self.cumulative_correction / self.corrected_actions


class InvertedPendulumCBF(BaseSafetyFilter):
    """CBF-QP filter specialised for the inverted pendulum."""

    def __init__(
        self,
        x_max: float = 1.5,
        alpha: float = 1.0,
        dt: float = 0.02,
        action_limit: Optional[float] = None,
        prior_overrides: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.model = build_inverted_pendulum_symbolic(dt=dt, overrides=prior_overrides)

        self.x_max = x_max
        self.alpha = alpha

        # Determine action limits either from overrides or symbolic model params.
        max_force = action_limit if action_limit is not None else float(self.model.max_force)
        self.action_low = -max_force
        self.action_high = max_force

        # Build CBF expressions using CasADi.
        self._setup_barrier_functions()
        self._setup_optimizer()

        # Statistics.
        self.stats = CBFStatistics()

    def _setup_barrier_functions(self) -> None:
        X = self.model.x_sym
        U = self.model.u_sym

        barrier_expr = 1.0 - (X[0] / self.x_max) ** 2
        self._barrier = cs.Function("h", [X], [barrier_expr], ["X"], ["h"])

        grad_h = cs.gradient(barrier_expr, X)
        lie_expr = cs.dot(grad_h, self.model.x_dot)
        self._lie = cs.Function("lie", [X, U], [lie_expr], ["X", "U"], ["Lfh"])  # h_dot

    def _setup_optimizer(self) -> None:
        nx, nu = self.model.nx, self.model.nu # Number of state and input variables

        opti = cs.Opti("conic")
        u_var = opti.variable(nu, 1) # Optimization variable: control input
        state_param = opti.parameter(nx, 1) # States
        u_desired_param = opti.parameter(nu, 1) # Desired control input

        barrier_val = self._barrier(X=state_param)["h"]
        lie_val = self._lie(X=state_param, U=u_var)["Lfh"]

        opti.subject_to(lie_val + self.alpha * barrier_val >= 0)
        opti.subject_to(u_var >= self.action_low)
        opti.subject_to(u_var <= self.action_high)

        cost = 0.5 * cs.sumsqr(u_var - u_desired_param)
        opti.minimize(cost)

        opts = {"printLevel": "low", "error_on_fail": False}
        opti.solver("qpoases", opts)

        self._opti = opti
        self._opti_vars = {
            "u": u_var,
            "state": state_param,
            "u_desired": u_desired_param,
        }

    def _evaluate_barrier(self, state: np.ndarray) -> float:
        x = np.asarray(state, dtype=np.float64).reshape(self.model.nx, 1)
        return float(self._barrier(X=x)["h"])

    def _evaluate_constraint(self, state: np.ndarray, action: np.ndarray) -> float:
        x = np.asarray(state, dtype=np.float64).reshape(self.model.nx, 1)
        u = np.atleast_1d(action).astype(np.float64).reshape(self.model.nu, 1)
        lie = float(self._lie(X=x, U=u)["Lfh"])
        return lie + self.alpha * self._evaluate_barrier(state)  # h_dot + alpha * h, require to be non-negative

    def _grid_search_fallback(self, state: np.ndarray, desired: float) -> float:
        search_grid = np.linspace(self.action_low, self.action_high, 81)
        feasible = []
        for candidate in search_grid:
            if self._evaluate_constraint(state, np.array([candidate])) >= 0.0:
                feasible.append(candidate)

        if feasible:
            feasible = np.array(feasible)
            idx = int(np.argmin(np.abs(feasible - desired)))
            return float(feasible[idx])

        return float(np.clip(desired, self.action_low, self.action_high))

    def _solve_qp(self, state: np.ndarray, u_rl: float) -> Tuple[float, bool]:
        opti = self._opti
        u_var = self._opti_vars["u"]
        state_param = self._opti_vars["state"]
        desired_param = self._opti_vars["u_desired"]

        state_vec = np.asarray(state, dtype=np.float64).reshape(self.model.nx, 1)
        desired_vec = np.array([[u_rl]], dtype=np.float64)

        opti.set_value(state_param, state_vec)
        opti.set_value(desired_param, desired_vec)
        opti.set_initial(u_var, desired_vec)

        try:
            sol = opti.solve()
            certified = float(sol.value(u_var))
            return certified, True
        except RuntimeError:
            certified = self._grid_search_fallback(state, u_rl)
            return certified, False

    def certify_action(
        self,
        current_state,
        uncertified_action,
        info=None,
    ) -> Tuple[np.ndarray, bool]:
        state = np.asarray(current_state, dtype=np.float64).reshape(self.model.nx)
        desired = float(np.atleast_1d(uncertified_action)[0])

        desired = float(np.clip(desired, self.action_low, self.action_high))
        certified, feasible = self._solve_qp(state, desired)

        self.stats.record(certified - desired)
        corrected = not np.isclose(certified, desired)
        if not feasible:
            corrected = True
            self.stats.failed_actions += 1
        action = np.array([certified], dtype=np.float32)
        return action, corrected

    # Helper accessors -------------------------------------------------
    def reset_stats(self) -> None:
        self.stats = CBFStatistics()

    def get_stats(self) -> dict:
        return {
            "total_actions": self.stats.total_actions,
            "corrected_actions": self.stats.corrected_actions,
            "correction_rate": self.stats.correction_rate,
            "avg_correction": self.stats.average_correction,
            "max_correction": self.stats.max_correction,
            "failed_actions": self.stats.failed_actions,
        }


