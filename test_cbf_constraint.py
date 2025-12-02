"""
Test script to verify CBF constraint calculation behavior
"""
import numpy as np
import sys
sys.path.append('/home/dmy/gymtest/safety-starter-agents')

# Import the functions we need to test
from reacher_safe_rl_multi_seed import hocbf_A_b_theta2_constraint, reacher_f_g

def test_cbf_constraint_symmetry():
    """Test that CBF constraint behaves symmetrically for +/- theta1"""
    print("Testing CBF Constraint Symmetry")
    print("="*60)
    
    theta2_max = 2.4
    c1, c2 = 4.0, 60.0
    
    # Test symmetric states
    test_cases = [
        # [theta0, theta1, theta0_dot, theta1_dot]
        [0.0,  1.5, 0.0, 0.0],  # Positive theta1, at rest
        [0.0, -1.5, 0.0, 0.0],  # Negative theta1, at rest
        [0.0,  2.2, 0.0, 0.5],  # Near limit, moving away
        [0.0, -2.2, 0.0, -0.5], # Near limit (negative), moving away
    ]
    
    u_test = np.array([0.5, 0.5])
    
    for x in test_cases:
        x = np.array(x)
        A, b = hocbf_A_b_theta2_constraint(x, theta2_max, c1, c2)
        
        # Compute constraint value
        constraint = (A @ u_test)[0] - b[0]
        
        # Compute barrier function value
        h = theta2_max**2 - x[1]**2
        hdot = -2 * x[1] * x[3]
        
        print(f"\nState: theta1={x[1]:.3f}, theta1_dot={x[3]:.3f}")
        print(f"  h={h:.3f}, hdot={hdot:.3f}")
        print(f"  A={A[0]}")
        print(f"  b={b[0]:.3f}")
        print(f"  A·u={((A @ u_test)[0]):.3f}")
        print(f"  Constraint (A·u - b)={constraint:.3f}")
        print(f"  Status: {'SAFE' if constraint >= 0 else 'UNSAFE'}")

def test_constraint_with_actions():
    """Test how different actions affect the constraint"""
    print("\n" + "="*60)
    print("Testing CBF Constraint with Different Actions")
    print("="*60)
    
    theta2_max = 2.4
    c1, c2 = 4.0, 60.0
    
    # State near the positive limit
    x = np.array([0.0, 2.0, 0.0, 0.1])  # theta1=2.0, moving positive
    
    actions = [
        np.array([0.0,  1.0]),  # Push towards limit
        np.array([0.0,  0.0]),  # No action
        np.array([0.0, -1.0]),  # Push away from limit
    ]
    
    print(f"\nState: theta1={x[1]:.3f}, theta1_dot={x[3]:.3f}")
    h = theta2_max**2 - x[1]**2
    print(f"Barrier h={h:.3f} (distance from limit: {theta2_max - abs(x[1]):.3f})")
    
    A, b = hocbf_A_b_theta2_constraint(x, theta2_max, c1, c2)
    print(f"A={A[0]}, b={b[0]:.3f}")
    
    for u in actions:
        constraint = (A @ u)[0] - b[0]
        print(f"\n  Action u={u}:")
        print(f"    A·u={((A @ u)[0]):.3f}")
        print(f"    Constraint (A·u - b)={constraint:.3f}")
        print(f"    Status: {'SAFE' if constraint >= 0 else 'UNSAFE'}")

def test_constraint_calculation_manual():
    """Manually verify constraint calculation matches the formula"""
    print("\n" + "="*60)
    print("Manual Verification of Constraint Calculation")
    print("="*60)
    
    # Use values from the terminal output
    theta2_max = 2.4
    c1, c2 = 4.0, 60.0
    
    # State
    theta1 = -1.479
    theta1_dot = 0.0  # Assume at rest for simplicity
    x = np.array([0.0, theta1, 0.0, theta1_dot])
    
    # Action
    u_nom = np.array([0.64134336, 0.2887864])
    
    # Get A and b from function
    A, b = hocbf_A_b_theta2_constraint(x, theta2_max, c1, c2)
    
    # Compute h, hdot manually
    h = theta2_max**2 - theta1**2
    hdot = -2 * theta1 * theta1_dot
    
    print(f"State: theta1={theta1:.3f}, theta1_dot={theta1_dot:.3f}")
    print(f"Action: u={u_nom}")
    print(f"\nBarrier function:")
    print(f"  h = {theta2_max}^2 - {theta1}^2 = {h:.6f}")
    print(f"  hdot = -2*{theta1}*{theta1_dot} = {hdot:.6f}")
    
    print(f"\nCBF constraint A·u >= b:")
    print(f"  A = {A[0]}")
    print(f"  b = {b[0]:.6f}")
    print(f"  A·u = {((A @ u_nom)[0]):.6f}")
    print(f"  Constraint (A·u - b) = {((A @ u_nom)[0] - b[0]):.6f}")
    
    # Manual calculation
    f, g = reacher_f_g(x)
    f3 = f[3]
    print(f"\nManual calculation:")
    print(f"  f3 = {f3:.6f}")
    print(f"  Expected b = 2*{theta1_dot}^2 + 2*{theta1}*{f3} + 2*{c1}*{theta1}*{theta1_dot} - {c2}*{h}")
    print(f"             = {2*theta1_dot**2 + 2*theta1*f3 + 2*c1*theta1*theta1_dot - c2*h:.6f}")
    print(f"  Actual b   = {b[0]:.6f}")
    print(f"  Match: {np.isclose(b[0], 2*theta1_dot**2 + 2*theta1*f3 + 2*c1*theta1*theta1_dot - c2*h)}")

if __name__ == "__main__":
    test_cbf_constraint_symmetry()
    test_constraint_with_actions()
    test_constraint_calculation_manual()
    print("\n" + "="*60)
    print("Testing Complete!")
    print("="*60)




