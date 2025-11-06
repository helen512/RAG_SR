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
        x_max: float = 0.8,
        alpha: float = 0.7,
        dt: float = 0.02,
        action_limit: Optional[float] = None,
        prior_overrides: Optional[dict] = None,
        kappa: float = 0.1,  # Decay rate for discrete-time CBF
        use_discrete_cbf: bool = True,  # Use discrete-time CBF by default
        slack_weight: float = 100.0,  # Penalty weight for slack variable
    ) -> None:
        super().__init__()
        self.model = build_inverted_pendulum_symbolic(dt=dt, overrides=prior_overrides)

        self.x_max = x_max
        self.alpha = alpha
        self.kappa = kappa
        self.use_discrete_cbf = use_discrete_cbf
        self.slack_weight = slack_weight

        # Determine action limits either from overrides or symbolic model params.
        max_force = action_limit if action_limit is not None else float(self.model.max_force)
        self.action_low = -max_force
        self.action_high = max_force

        # Build CBF expressions using CasADi.
        self._setup_barrier_functions()
        self._setup_optimizer()

        # Statistics.
        self.stats = CBFStatistics()
        
        # Logging for discrete violations
        self.enable_violation_logging = False
        self.violation_logs = []
        self._last_slack = 0.0

    def _setup_barrier_functions(self) -> None:
        X = self.model.x_sym
        U = self.model.u_sym

        # Barrier function: h(x) = 1 - (x/x_max)^2
        # Safe set: h(x) >= 0, i.e., |x| <= x_max
        barrier_expr = 1 - (X[0] / self.x_max)**2
        self._barrier = cs.Function("h", [X], [barrier_expr], ["X"], ["h"])

        if self.use_discrete_cbf:
            # Discrete-time CBF: h(x_next) - (1-kappa)*h(x) >= 0
            # Use the model integrator to predict next state
            # Note: fd_func returns a dict with key 'xf' for final state
            pass  # Setup done in optimizer
        else:
            # Continuous-time CBF: h_dot + alpha*h >= 0 (legacy)
            grad_h = cs.gradient(barrier_expr, X)
            lie_expr = cs.dot(grad_h, self.model.x_dot)
            self._lie = cs.Function("lie", [X, U], [lie_expr], ["X", "U"], ["Lfh"])  # h_dot

    def _setup_optimizer(self) -> None:
        nx, nu = self.model.nx, self.model.nu # Number of state and input variables

        opti = cs.Opti("conic")
        u_var = opti.variable(nu, 1) # Optimization variable: control input
        slack_var = opti.variable(1, 1) # Slack variable for soft constraint
        state_param = opti.parameter(nx, 1) # States
        u_desired_param = opti.parameter(nu, 1) # Desired control input

        constraint_expr = None
        if self.use_discrete_cbf:
            # Discrete-time CBF: h(x_{k+1}) - (1-kappa)*h(x_k) >= 0
            # Predict next state using model integrator
            x_next_dict = self.model.fd_func(x0=state_param, p=u_var)
            x_next = x_next_dict['xf']
            
            # Evaluate barrier at current and next states
            h_current = self._barrier(X=state_param)["h"]
            h_next = self._barrier(X=x_next)["h"]
            
            # Discrete-time CBF constraint with slack
            constraint_expr = h_next - (1 - self.kappa) * h_current + slack_var
        else:
            # Continuous-time CBF: h_dot + alpha*h >= 0 (legacy)
            barrier_val = self._barrier(X=state_param)["h"]
            lie_val = self._lie(X=state_param, U=u_var)["Lfh"]
            constraint_expr = lie_val + self.alpha * barrier_val + slack_var

        opti.subject_to(constraint_expr >= 0)
        opti.subject_to(slack_var >= 0)
        
        opti.subject_to(u_var >= self.action_low)
        opti.subject_to(u_var <= self.action_high)

        cost = 0.5 * cs.sumsqr(u_var - u_desired_param) + self.slack_weight * cs.sumsqr(slack_var)
        opti.minimize(cost)

        opts = {"printLevel": "low", "error_on_fail": False}
        opti.solver("qpoases", opts)

        self._opti = opti
        self._opti_vars = {
            "u": u_var,
            "slack": slack_var,
            "state": state_param,
            "u_desired": u_desired_param,
        }

    def _compute_affine_terms(self, state: np.ndarray) -> Tuple[float, float]:
        """Return (L_f h, L_g h) for the current state.
        
        Only valid for continuous-time CBF (legacy mode).
        """
        if self.use_discrete_cbf:
            raise RuntimeError("Affine terms not applicable for discrete-time CBF")
            
        x = np.asarray(state, dtype=np.float64).reshape(self.model.nx, 1)

        u_low = np.array([[self.action_low]], dtype=np.float64)
        u_high = np.array([[self.action_high]], dtype=np.float64)

        lie_low = float(self._lie(X=x, U=u_low)["Lfh"])
        lie_high = float(self._lie(X=x, U=u_high)["Lfh"])

        if np.isclose(self.action_high, self.action_low):
            lg_h = 0.0
        else:
            lg_h = (lie_high - lie_low) / (self.action_high - self.action_low)

        lf_h = lie_low - lg_h * self.action_low
        return lf_h, lg_h

    def _analytic_projection(self, state: np.ndarray, desired: float) -> Tuple[float, bool]:
        """Project the desired action using the affine CBF constraint.
        
        Only valid for continuous-time CBF (legacy mode).
        """
        if self.use_discrete_cbf:
            # For discrete-time CBF, constraint is not affine - use QP solver instead
            return self._solve_qp(state, desired)
            
        lf_h, lg_h = self._compute_affine_terms(state)
        h_val = self._evaluate_barrier(state)
        c = lf_h + self.alpha * h_val

        # Control authority negligible: action cannot influence constraint meaningfully
        if np.isclose(lg_h, 0.0, atol=1e-8):
            clipped = float(np.clip(desired, self.action_low, self.action_high))
            feasible = (c >= 0.0)
            return clipped, feasible

        threshold = -c / lg_h
        lower = self.action_low
        upper = self.action_high

        if lg_h > 0:
            lower = max(lower, threshold)
        else:
            upper = min(upper, threshold)

        if lower <= upper:
            projected = float(np.clip(desired, lower, upper))
            return projected, True

        # No feasible control inside bounds; choose boundary that maximizes constraint value
        candidates = [self.action_low, self.action_high]
        def constraint_margin(u: float) -> float:
            return lg_h * u + c
        best = max(candidates, key=constraint_margin)
        return float(np.clip(best, self.action_low, self.action_high)), False

    def _evaluate_barrier(self, state: np.ndarray) -> float:
        x = np.asarray(state, dtype=np.float64).reshape(self.model.nx, 1)
        return float(self._barrier(X=x)["h"])

    def _evaluate_constraint(self, state: np.ndarray, action: np.ndarray) -> float:
        """Evaluate the CBF constraint value for given state and action.
        
        Returns the constraint margin (should be >= 0 for safety).
        """
        x = np.asarray(state, dtype=np.float64).reshape(self.model.nx, 1)
        u = np.atleast_1d(action).astype(np.float64).reshape(self.model.nu, 1)
        
        if self.use_discrete_cbf:
            # Discrete-time CBF: h(x_next) - (1-kappa)*h(x_k)
            x_next_dict = self.model.fd_func(x0=x, p=u)
            x_next = x_next_dict['xf']
            h_current = self._evaluate_barrier(state)
            h_next = float(self._barrier(X=x_next)["h"])
            return h_next - (1 - self.kappa) * h_current
        else:
            # Continuous-time CBF: h_dot + alpha*h
            lie = float(self._lie(X=x, U=u)["Lfh"])
            return lie + self.alpha * self._evaluate_barrier(state)

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
        slack_var = self._opti_vars["slack"]
        state_param = self._opti_vars["state"]
        desired_param = self._opti_vars["u_desired"]

        state_vec = np.asarray(state, dtype=np.float64).reshape(self.model.nx, 1)
        desired_vec = np.array([[u_rl]], dtype=np.float64)

        opti.set_value(state_param, state_vec)
        opti.set_value(desired_param, desired_vec)
        opti.set_initial(u_var, desired_vec)
        opti.set_initial(slack_var, 0.0)

        try:
            sol = opti.solve()
            certified = float(sol.value(u_var))
            self._last_slack = float(sol.value(slack_var))
            return certified, True
        except RuntimeError:
            self._last_slack = np.nan
            certified = self._grid_search_fallback(state, u_rl)
            return certified, False

    def certify_action(
        self,
        current_state,
        uncertified_action,
        info=None,
    ) -> Tuple[np.ndarray, bool]:
        self._last_slack = 0.0
        state = np.asarray(current_state, dtype=np.float64).reshape(self.model.nx)
        desired = float(np.atleast_1d(uncertified_action)[0])

        desired = float(np.clip(desired, self.action_low, self.action_high))
        certified, feasible = self._analytic_projection(state, desired)

        if not feasible:
            certified = self._grid_search_fallback(state, desired)
            feasible = self._evaluate_constraint(state, np.array([certified])) >= 0.0

        self.stats.record(certified - desired)
        corrected = not np.isclose(certified, desired)
        slack_used = self._last_slack
        if isinstance(slack_used, (float, int)):
            if not np.isnan(slack_used) and slack_used > 1e-6:
                corrected = True
        if not feasible:
            corrected = True
            self.stats.failed_actions += 1
        action = np.array([certified], dtype=np.float32)
        return action, corrected

    # Helper methods for prediction and logging ------------------------
    def predict_next_barrier(self, state: np.ndarray, action: np.ndarray) -> float:
        """Predict h(x_{k+1}) using the model integrator."""
        x = np.asarray(state, dtype=np.float64).reshape(self.model.nx, 1)
        u = np.atleast_1d(action).astype(np.float64).reshape(self.model.nu, 1)
        x_next_dict = self.model.fd_func(x0=x, p=u)
        x_next = x_next_dict['xf']
        return float(self._barrier(X=x_next)["h"])
    
    def log_step(self, state: np.ndarray, action: np.ndarray, 
                 next_state: np.ndarray, step: int) -> dict:
        """Log CBF values and check for discrete violations.
        
        Returns a dict with:
        - h_k: barrier value at current state
        - constraint_k: constraint value at current state (should be >= 0)
        - h_next_predicted: predicted barrier value at next state
        - h_next_actual: actual barrier value at next state (from env)
        - discrete_violation: True if constraint satisfied but h_next < 0
        """
        h_k = self._evaluate_barrier(state)
        constraint_k = self._evaluate_constraint(state, action)
        h_next_pred = self.predict_next_barrier(state, action)
        h_next_actual = self._evaluate_barrier(next_state)
        slack_val = self._last_slack if isinstance(self._last_slack, (float, int)) else float(self._last_slack)
        if np.isnan(slack_val):
            slack_val = 0.0
        
        # Flag discrete violation: constraint satisfied but actual h_next < 0
        discrete_violation = (constraint_k >= 0) and (h_next_actual < 0)
        
        log_entry = {
            "step": step,
            "h_k": h_k,
            "constraint_k": constraint_k,
            "h_next_predicted": h_next_pred,
            "h_next_actual": h_next_actual,
            "discrete_violation": discrete_violation,
            "state_x": float(state[0]) if len(state) > 0 else 0.0,
            "state_x_dot": float(state[1]) if len(state) > 1 else 0.0,
            "action": float(action[0]) if len(np.atleast_1d(action)) > 0 else 0.0,
            "slack": slack_val,
        }
        
        if self.enable_violation_logging:
            self.violation_logs.append(log_entry)
            
        return log_entry
    
    def enable_logging(self, enable: bool = True) -> None:
        """Enable or disable violation logging."""
        self.enable_violation_logging = enable
        if enable:
            self.violation_logs = []
    
    def get_violation_logs(self) -> list:
        """Return all logged violations."""
        return self.violation_logs
    
    def get_violation_summary(self) -> dict:
        """Return a summary of logged violations."""
        if not self.violation_logs:
            return {
                "total_steps": 0,
                "violations": 0,
                "violation_rate": 0.0,
            }
        
        violations = sum(1 for log in self.violation_logs if log["discrete_violation"])
        return {
            "total_steps": len(self.violation_logs),
            "violations": violations,
            "violation_rate": violations / len(self.violation_logs),
            "logs": self.violation_logs,
        }

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


