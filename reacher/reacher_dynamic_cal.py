import sympy as sp

# ============================
# 1. Symbols and generalized coords
# ============================

t = sp.symbols('t')  # time

# generalized coordinates as functions of time
q1 = sp.Function('q1')(t)
q2 = sp.Function('q2')(t)
dq1 = sp.diff(q1, t)
dq2 = sp.diff(q2, t)

# parameters (lengths, COM distances, masses, inertias)
l1, c1, c2 = sp.symbols('l1 c1 c2', real=True)
m1, m2 = sp.symbols('m1 m2', positive=True)
I1, I2 = sp.symbols('I1 I2', positive=True)

# ============================
# 2. Kinematics: COM positions & velocities
# ============================

# Link 1 COM
x1 = c1 * sp.cos(q1)
y1 = c1 * sp.sin(q1)

# Link 2 COM
x2 = l1 * sp.cos(q1) + c2 * sp.cos(q1 + q2)
y2 = l1 * sp.sin(q1) + c2 * sp.sin(q1 + q2)

# velocities via time differentiation
vx1 = sp.diff(x1, t)
vy1 = sp.diff(y1, t)
vx2 = sp.diff(x2, t)
vy2 = sp.diff(y2, t)

v1_sq = sp.simplify(vx1**2 + vy1**2)
v2_sq = sp.simplify(vx2**2 + vy2**2)

# angular velocities (planar z-axis rotations)
omega1 = dq1
omega2 = dq1 + dq2

# ============================
# 3. Kinetic energy and Lagrangian
# ============================

T = sp.simplify(
    sp.Rational(1, 2) * m1 * v1_sq
  + sp.Rational(1, 2) * I1 * omega1**2
  + sp.Rational(1, 2) * m2 * v2_sq
  + sp.Rational(1, 2) * I2 * omega2**2
)

# Horizontal plane Reacher: no gravity term → V = 0
L = T

# ============================
# 4. Euler–Lagrange equations
# ============================

q  = [q1, q2]
dq = [dq1, dq2]

tau1, tau2 = sp.symbols('tau1 tau2')
tau = sp.Matrix([tau1, tau2])  # generalized torques

EL = []
for qi, dqi in zip(q, dq):
    dL_ddqi   = sp.diff(L, dqi)
    dt_dL_ddqi = sp.diff(dL_ddqi, t)
    dL_dqi    = sp.diff(L, qi)
    EL.append(sp.simplify(dt_dL_ddqi - dL_dqi))

EL = sp.Matrix(EL)   # EL = tau

# ============================
# 5. Replace q̈ with symbols and extract M(q), h(q, q̇)
# ============================

ddq1, ddq2 = sp.symbols('ddq1 ddq2')
subs_ddq = {
    sp.diff(q1, t, 2): ddq1,
    sp.diff(q2, t, 2): ddq2,
}

# make equations linear in ddq1, ddq2
EL_lin = sp.simplify(EL.subs(subs_ddq))  # EL_lin(q, dq, ddq) = tau

ddq = sp.Matrix([ddq1, ddq2])

# Bring tau to left: resid = M(q)*ddq + h(q,dq) - tau
resid = sp.simplify(EL_lin - tau)

# Mass matrix M: coefficients of ddq
M = sp.Matrix([[sp.diff(resid[i], ddq[j]) for j in range(2)] for i in range(2)])
M = sp.simplify(M)

# Nonlinear term h(q, dq): leftover after removing M*ddq
h = sp.simplify(resid - M*ddq)

print("Lagrangian L(q, dq) =")
sp.pprint(L)
print("\nMass matrix M(q) =")
sp.pprint(M)
print("\nNonlinear term h(q, dq) (Coriolis/centrifugal) =")
sp.pprint(h)
