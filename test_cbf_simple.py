"""
Standalone test to verify CBF constraint calculation
"""
import numpy as np

# Copy the dynamics function
def reacher_f_g(x):
    """Compute f and g for Reacher dynamics"""
    theta1, theta2, theta1_dot, theta2_dot = x
    c2 = np.cos(theta2)
    s2 = np.sin(theta2)
    
    ALPHA = 6.86512e-4
    BETA  = 2.24100e-4
    DELTA = 1.69004e-4

    def accel_theta(u):
        u1 = u[0]
        u2 = u[1]
        
        # Mass matrix (with armature)
        m11 = 1.0 + ALPHA + 2.0 * BETA * c2
        m12 =        DELTA +       BETA * c2
        m22 = 1.0 +  DELTA
        
        # Coriolis / centrifugal (links)
        h1 = -2.0 * BETA * s2 * theta1_dot * theta2_dot - BETA * s2 * theta2_dot * theta2_dot
        h2 =  BETA * s2 * theta1_dot * theta1_dot
        
        # Damping
        d1 = theta1_dot
        d2 = theta2_dot
        
        # RHS = tau - h - damping
        r1 = 200.0 * u1 - h1 - d1
        r2 = 200.0 * u2 - h2 - d2
        
        # Solve 2x2 system explicitly
        det = m11 * m22 - m12 * m12
        ddq1 = ( r1 * m22 - r2 * m12 ) / det
        ddq2 = ( m11 * r2 - m12 * r1 ) / det
        
        return ddq1, ddq2
    
    # Compute f and g vectors
    theta0_ddot_0, theta1_ddot_0 = accel_theta(np.array([0.0, 0.0]))
    theta0_ddot_u1, theta1_ddot_u1 = accel_theta(np.array([1.0, 0.0]))
    theta0_ddot_u2, theta1_ddot_u2 = accel_theta(np.array([0.0, 1.0]))
    
    # State vector: [theta0, theta1, theta0_dot, theta1_dot]
    f = np.array([theta1_dot, theta2_dot, theta0_ddot_0, theta1_ddot_0])
    g = np.array([[0.0, 0.0, theta0_ddot_u1 - theta0_ddot_0, theta1_ddot_u1 - theta1_ddot_0], 
                  [0.0, 0.0, theta0_ddot_u2 - theta0_ddot_0, theta1_ddot_u2 - theta1_ddot_0]])
    
    return f, g

# Copy the HOCBF constraint function
def hocbf_A_b_theta2_constraint(x, theta2_max, c1=0.05, c2=0.5):
    """Returns A, b for HOCBF constraint"""
    f, g = reacher_f_g(x)
    theta1, theta1_dot = x[1], x[3]
    f3, g31, g32 = f[3], g[0,3], g[1,3]
    
    # Barrier function: h = theta2_max^2 - theta1^2
    h = theta2_max**2 - theta1**2
    hdot = -2 * theta1 * theta1_dot
    
    # HOCBF constraint: A·u >= b
    A = -2 * theta1*np.array([[g31, g32]])
    b = np.array([2 * theta1_dot**2 + 2 * theta1* f3 - c1*hdot - c2*h], dtype=float)
    
    return A, b

print("="*70)
print("Testing CBF Constraint Calculation")
print("="*70)

# Test case from terminal output
theta2_max = 2.4
c1, c2 = 4.0, 60.0

# State: theta1=-1.479
theta1 = -1.479
theta1_dot = 0.1  # Small velocity
x = np.array([0.0, theta1, 0.0, theta1_dot])

# Action from terminal
u_nom = np.array([0.64134336, 0.2887864])

# Get constraint
A, b = hocbf_A_b_theta2_constraint(x, theta2_max, c1, c2)
constraint = (A @ u_nom)[0] - b[0]

# Compute barrier values
h = theta2_max**2 - theta1**2
hdot = -2 * theta1 * theta1_dot

print(f"\nTest Case 1: Near limit, moderate velocity")
print(f"State: theta1={theta1:.3f}, theta1_dot={theta1_dot:.3f}")
print(f"Action: u={u_nom}")
print(f"\nBarrier function:")
print(f"  h = {theta2_max}² - {theta1}² = {h:.6f}")
print(f"  hdot = -2*{theta1}*{theta1_dot} = {hdot:.6f}")
print(f"\nCBF Constraint (A·u >= b):")
print(f"  A = [{A[0,0]:.6f}, {A[0,1]:.6f}]")
print(f"  b = {b[0]:.6f}")
print(f"  A·u = {((A @ u_nom)[0]):.6f}")
print(f"  Constraint (A·u - b) = {constraint:.6f}")
print(f"  Status: {'✓ SAFE' if constraint >= 0 else '✗ UNSAFE'}")

# Test case 2: Near limit with action towards limit
print("\n" + "="*70)
print("Test Case 2: Near positive limit, action pushing towards limit")
print("="*70)

theta1 = 2.0
theta1_dot = 0.2
x = np.array([0.0, theta1, 0.0, theta1_dot])
u_dangerous = np.array([0.0, 1.0])  # Large positive action on joint 1

A, b = hocbf_A_b_theta2_constraint(x, theta2_max, c1, c2)
constraint = (A @ u_dangerous)[0] - b[0]
h = theta2_max**2 - theta1**2
hdot = -2 * theta1 * theta1_dot

print(f"State: theta1={theta1:.3f}, theta1_dot={theta1_dot:.3f}")
print(f"Action: u={u_dangerous}")
print(f"\nBarrier: h={h:.6f}, hdot={hdot:.6f}")
print(f"Constraint: A·u - b = {constraint:.6f}")
print(f"Status: {'✓ SAFE' if constraint >= 0 else '✗ UNSAFE (needs correction)'}")

# Test case 3: Near limit with action away from limit
print("\n" + "="*70)
print("Test Case 3: Near positive limit, action pushing away")
print("="*70)

u_safe = np.array([0.0, -1.0])  # Large negative action
constraint = (A @ u_safe)[0] - b[0]

print(f"State: theta1={theta1:.3f}, theta1_dot={theta1_dot:.3f}")
print(f"Action: u={u_safe}")
print(f"Constraint: A·u - b = {constraint:.6f}")
print(f"Status: {'✓ SAFE' if constraint >= 0 else '✗ UNSAFE'}")

# Test the symmetry
print("\n" + "="*70)
print("Test Case 4: Symmetric case (negative theta1)")
print("="*70)

theta1 = -2.0
theta1_dot = -0.2
x = np.array([0.0, theta1, 0.0, theta1_dot])

A, b = hocbf_A_b_theta2_constraint(x, theta2_max, c1, c2)
h = theta2_max**2 - theta1**2

print(f"State: theta1={theta1:.3f}, theta1_dot={theta1_dot:.3f}")
print(f"Barrier: h={h:.6f}")

# Action pushing towards negative limit
u1 = np.array([0.0, -1.0])
c1 = (A @ u1)[0] - b[0]
print(f"Action towards limit: u={u1}, constraint={c1:.6f}, {'✓' if c1>=0 else '✗'}")

# Action pushing away (towards zero)
u2 = np.array([0.0, 1.0])
c2 = (A @ u2)[0] - b[0]
print(f"Action away from limit: u={u2}, constraint={c2:.6f}, {'✓' if c2>=0 else '✗'}")

print("\n" + "="*70)
print("Summary of Findings:")
print("="*70)
print("""
The CBF constraint value (A·u - b) depends on:
1. Current state (theta1, theta1_dot)
2. Proposed action (u)
3. System dynamics (f, g)

Key observations:
- The constraint is A·u >= b, so A·u - b >= 0 means SAFE
- A changes sign based on theta1 (negative for theta1>0, positive for theta1<0)
- This ensures symmetric behavior around theta1=0
- The constraint predicts FUTURE violations, not just current state safety
- Large |b| values (due to -c2*h term) are normal when h is large (far from limit)

The constraint correctly evaluates whether an action will maintain the barrier
function condition: ḧ + c1·ḣ + c2·h >= 0
""")




