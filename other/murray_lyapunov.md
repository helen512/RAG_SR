![](_page_0_Figure_1.jpeg)

Figure 4.12: Solution curves for a stable limit cycle. The phase portrait on the left shows that the trajectory for the system rapidly converges to the stable limit cycle. The starting points for the trajectories are marked by circles in the phase portrait. The time domain plots on the right show that the states do not converge to the solution but instead maintain a constant phase error.

In this section we will describe techniques for determining the stability of solutions for a nonlinear system  $(4.13)$ . We will generally be interested in stability of equilibrium points, and it will be convenient to assume that  $x_e = 0$  is the equilibrium point of interest. (If not, rewrite the equations in a new set of coordinates  $z = x - x_e$ .

## Lyapunov Functions

A Lyapunov function  $V: \mathbb{R}^n \to \mathbb{R}$  is an energy-like function that can be used to determine the stability of a system. Roughly speaking, if we can find a nonnegative function that always decreases along trajectories of the system, we can conclude that the minimum of the function is a stable equilibrium point (locally).

To describe this more formally, we start with a few definitions. We say that a continuous function V is *positive definite* if  $V(x) > 0$  for all  $x \neq 0$  and  $V(0) = 0$ . Similarly, a function is *negative definite* if  $V(x) < 0$  for all  $x \neq 0$  and  $V(0) = 0$ . We say that a function V is *positive semidefinite* if  $V(x) \ge 0$  for all x, but  $V(x)$ can be zero at points other than just  $x = 0$ .

To illustrate the difference between a positive definite function and a positive semidefinite function, suppose that  $x \in \mathbb{R}^2$  and let

$$V_1(x) = x_1^2, \qquad V_2(x) = x_1^2 + x_2^2.$$

Both  $V_1$  and  $V_2$  are always nonnegative. However, it is possible for  $V_1$  to be zero even if  $x \neq 0$ . Specifically, if we set  $x = (0, c)$ , where  $c \in \mathbb{R}$  is any nonzero number, then  $V_1(x) = 0$ . On the other hand,  $V_2(x) = 0$  if and only if  $x = (0, 0)$ . Thus  $V_1$  is positive semidefinite and  $V_2$  is positive definite.

We can now characterize the stability of an equilibrium point  $x_e = 0$  for the system  $(4.13)$ .

**Theorem 4.2** (Lyapunov stability theorem). Let  $V$  be a nonnegative function on

![](_page_1_Figure_1.jpeg)

Figure 4.13: Geometric illustration of Lyapunov's stability theorem. The closed contours represent the level sets of the Lyapunov function  $V(x) = c$ . If  $dx/dt$  points inward to these sets at all points along the contour, then the trajectories of the system will always cause  $V(x)$ to decrease along the trajectory.

 $\mathbb{R}^n$  and let  $\dot{V}$  represent the time derivative of V along trajectories of the system dynamics (4.13):

$$\dot{V} = \frac{\partial V}{\partial x}\frac{dx}{dt} = \frac{\partial V}{\partial x}F(x).$$

Let  $B_r = B_r(0)$  be a ball of radius r around the origin. If there exists  $r > 0$  such that V is positive definite and  $\dot{V}$  is negative semidefinite for all  $x \in B_r$ , then  $x = 0$ is locally stable in the sense of Lyapunov. If V is positive definite and  $\dot{V}$  is negative definite in  $B_r$ , then  $x = 0$  is locally asymptotically stable.

If  $V$  satisfies one of the conditions above, we say that  $V$  is a (local) Lyapunov *function* for the system. These results have a nice geometric interpretation. The level curves for a positive definite function are the curves defined by  $V(x) = c$ ,  $c > 0$ , and for each c this gives a closed contour, as shown in Figure 4.13. The condition that  $V(x)$  is negative simply means that the vector field points toward lower-level contours. This means that the trajectories move to smaller and smaller values of V and if  $\dot{V}$  is negative definite then x must approach 0.

# **Example 4.9 Scalar nonlinear system**

Consider the scalar nonlinear system

$$\frac{dx}{dt} = \frac{2}{1+x} - x.$$

This system has equilibrium points at  $x = 1$  and  $x = -2$ . We consider the equilibrium point at  $x = 1$  and rewrite the dynamics using  $z = x - 1$ :

$$\frac{dz}{dt} = \frac{2}{2+z} - z - 1,$$

which has an equilibrium point at  $z = 0$ . Now consider the candidate Lyapunov function

$$V(z) = \frac{1}{2}z^2,$$

which is globally positive definite. The derivative of  $V$  along trajectories of the system is given by

$$\dot{V}(z) = z\dot{z} = \frac{2z}{2+z} - z^2 - z.$$

If we restrict our analysis to an interval  $B_r$ , where  $r < 2$ , then  $2 + z > 0$  and we can multiply through by  $2 + z$  to obtain

$$2z - (z2 + z)(2 + z) = -z3 - 3z2 = -z2(z + 3) < 0, \qquad z \in B_r, r < 2.$$

It follows that  $V(z) < 0$  for all  $z \in B_r$ ,  $z \neq 0$ , and hence the equilibrium point  $x_e = 1$  is locally asymptotically stable.

A slightly more complicated situation occurs if  $\dot{V}$  is negative semidefinite. In this case it is possible that  $V(x) = 0$  when  $x \neq 0$ , and hence x could stop decreasing in value. The following example illustrates this case.

### Example 4.10 Hanging pendulum

A normalized model for a hanging pendulum is

$$\frac{dx_1}{dt} = x_2, \qquad \frac{dx_2}{dt} = -\sin x_1$$

where  $x_1$  is the angle between the pendulum and the vertical, with positive  $x_1$ corresponding to counterclockwise rotation. The equation has an equilibrium  $x_1 =$  $x_2 = 0$ , which corresponds to the pendulum hanging straight down. To explore the stability of this equilibrium we choose the total energy as a Lyapunov function:

$$V(x) = 1 - \cos x_1 + \frac{1}{2}x_2^2 \approx \frac{1}{2}x_1^2 + \frac{1}{2}x_2^2.$$

The Taylor series approximation shows that the function is positive definite for small x. The time derivative of  $V(x)$  is

$$V = \dot{x}_1 \sin x_1 + \dot{x}_2 x_2 = x_2 \sin x_1 - x_2 \sin x_1 = 0.$$

Since this function is positive semidefinite, it follows from Lyapunov's theorem that the equilibrium is stable but not necessarily asymptotically stable. When perturbed, the pendulum actually moves in a trajectory that corresponds to constant energy.  $\nabla$ 

Lyapunov functions are not always easy to find, and they are not unique. In many cases energy functions can be used as a starting point, as was done in Example 4.10. It turns out that Lyapunov functions can always be found for any stable system (under certain conditions), and hence one knows that if a system is stable, a Lyapunov function exists (and vice versa). Recent results using sum-of-squares methods have provided systematic approaches for finding Lyapunov systems [PPP02]. Sum-ofsquares techniques can be applied to a broad variety of systems, including systems whose dynamics are described by polynomial equations, as well as hybrid systems, which can have different models for different regions of state space.

For a linear dynamical system of the form

$$\frac{dx}{dt} = Ax$$

it is possible to construct Lyapunov functions in a systematic manner. To do so, we consider quadratic functions of the form

$$V(x) = x^T P x,$$

where  $P \in \mathbb{R}^{n \times n}$  is a symmetric matrix  $(P = P^T)$ . The condition that V be positive definite is equivalent to the condition that  $P$  be a *positive definite matrix*:

$$x^T P x > 0$$
, for all  $x \neq 0$ ,

which we write as  $P > 0$ . It can be shown that if P is symmetric, then P is positive definite if and only if all of its eigenvalues are real and positive.

Given a candidate Lyapunov function  $V(x) = x^T P x$ , we can now compute its derivative along flows of the system:

$$\dot{V} = \frac{\partial V}{\partial x}\frac{dx}{dt} = x^T(A^T P + PA)x =: -x^T Qx.$$

The requirement that  $V$  be negative definite (for asymptotic stability) becomes a condition that the matrix  $Q$  be positive definite. Thus, to find a Lyapunov function for a linear system it is sufficient to choose a  $Q > 0$  and solve the Lyapunov *equation*:

$$A^T P + P A = -Q.\t\t(4.14)$$

This is a linear equation in the entries of  $P$ , and hence it can be solved using linear algebra. It can be shown that the equation always has a solution if all of the eigenvalues of the matrix  $A$  are in the left half-plane. Moreover, the solution  $P$  is positive definite if  $Q$  is positive definite. It is thus always possible to find a quadratic Lyapunov function for a stable linear system. We will defer a proof of this until Chapter 5, where more tools for analysis of linear systems will be developed.

Knowing that we have a direct method to find Lyapunov functions for linear systems, we can now investigate the stability of nonlinear systems. Consider the system

$$\frac{dx}{dt} = F(x) =: Ax + \tilde{F}(x),\tag{4.15}$$

where  $F(0) = 0$  and  $\tilde{F}(x)$  contains terms that are second order and higher in the elements of x. The function  $Ax$  is an approximation of  $F(x)$  near the origin, and we can determine the Lyapunov function for the linear approximation and investigate if it is also a Lyapunov function for the full nonlinear system. The following example illustrates the approach.

## Example 4.11 Genetic switch

Consider the dynamics of a set of repressors connected together in a cycle, as shown in Figure 4.14a. The normalized dynamics for this system were given in Exercise 2.9:

$$\frac{dz_1}{d\tau} = \frac{\mu}{1 + z_2^n} - z_1, \qquad \frac{dz_2}{d\tau} = \frac{\mu}{1 + z_1^n} - z_2, \tag{4.16}$$

where  $z_1$  and  $z_2$  are scaled versions of the protein concentrations, *n* and  $\mu$  are

![](_page_4_Figure_1.jpeg)

Figure 4.14: Stability of a genetic switch. The circuit diagram in (a) represents two proteins that are each repressing the production of the other. The inputs  $u_1$  and  $u_2$  interfere with this repression, allowing the circuit dynamics to be modified. The equilibrium points for this circuit can be determined by the intersection of the two curves shown in (b).

parameters that describe the interconnection between the genes and we have set the external inputs  $u_1$  and  $u_2$  to zero.

The equilibrium points for the system are found by equating the time derivatives to zero. We define

$$f(u) = \frac{\mu}{1+u^n}, \qquad f'(u) = \frac{df}{du} = \frac{-\mu n u^{n-1}}{(1+u^n)^2},$$

and the equilibrium points are defined as the solutions of the equations

$$z_1 = f(z_2), \qquad z_2 = f(z_1).$$

If we plot the curves  $(z_1, f(z_1))$  and  $(f(z_2), z_2)$  on a graph, then these equations will have a solution when the curves intersect, as shown in Figure 4.14b. Because of the shape of the curves, it can be shown that there will always be three solutions: one at  $z_{1e} = z_{2e}$ , one with  $z_{1e} < z_{2e}$  and one with  $z_{1e} > z_{2e}$ . If  $\mu \gg 1$ , then we can show that the solutions are given approximately by

$$z_{1e} \approx \mu, \quad z_{2e} \approx \frac{1}{\mu^{n-1}}; \qquad z_{1e} = z_{2e}; \qquad z_{1e} \approx \frac{1}{\mu^{n-1}}, \quad z_{2e} \approx \mu. \tag{4.17}$$

To check the stability of the system, we write  $f(u)$  in terms of its Taylor series expansion about  $u_e$ :

$$f(u) = f(u_e) + f'(u_e) \cdot (u - u_e) + f''(u_e) \cdot (u - u_e)^2 + \text{higher-order terms},$$

where  $f'$  represents the first derivative of the function, and  $f''$  the second. Using these approximations, the dynamics can then be written as

$$\frac{dw}{dt} = \begin{bmatrix} -1 & f'(z_{2e}) \\ f'(z_{1e}) & -1 \end{bmatrix} w + \tilde{F}(w),$$

where  $w = z - z_e$  is the shifted state and  $\tilde{F}(w)$  represents quadratic and higher-order

terms.

We now use equation (4.14) to search for a Lyapunov function. Choosing  $Q = I$ and letting  $P \in \mathbb{R}^{2 \times 2}$  have elements  $p_{ij}$ , we search for a solution of the equation

$$\begin{bmatrix} -1 & f_2' \\ f_1' & -1 \end{bmatrix} \begin{bmatrix} p_{11} & p_{12} \\ p_{12} & p_{22} \end{bmatrix} + \begin{bmatrix} p_{11} & p_{12} \\ p_{12} & p_{22} \end{bmatrix} \begin{bmatrix} -1 & f_1' \\ f_2' & -1 \end{bmatrix} = \begin{bmatrix} -1 & 0 \\ 0 & -1 \end{bmatrix},$$

where  $f'_1 = f'(z_{1e})$  and  $f'_2 = f'(z_{2e})$ . Note that we have set  $p_{21} = p_{12}$  to force P to be symmetric. Multiplying out the matrices, we obtain

$$\begin{bmatrix} -2p_{11} + 2f'_2 p_{12} & p_{11}f'_1 - 2p_{12} + p_{22}f'_2 \ p_{11}f'_1 - 2p_{12} + p_{22}f'_2 & -2p_{22} + 2f'_1 p_{12} \end{bmatrix} = \begin{bmatrix} -1 & 0 \ 0 & -1 \end{bmatrix}$$

which is a set of *linear* equations for the unknowns  $p_{ij}$ . We can solve these linear equations to obtain

$$p_{11} = -\frac{f_1'^2 - f_2'f_1' + 2}{4(f_1'f_2' - 1)}, \qquad p_{12} = -\frac{f_1' + f_2'}{4(f_1'f_2' - 1)}, \qquad p_{22} = -\frac{f_2'^2 - f_1'f_2' + 2}{4(f_1'f_2' - 1)}.$$

To check that  $V(w) = w^T P w$  is a Lyapunov function, we must verify that  $V(w)$  is positive definite function or equivalently that  $P > 0$ . Since P is a 2  $\times$  2 symmetric matrix, it has two real eigenvalues  $\lambda_1$  and  $\lambda_2$  that satisfy

$$\lambda_1 + \lambda_2 = \text{trace}(P), \qquad \lambda_1 \cdot \lambda_2 = \text{det}(P).$$

In order for P to be positive definite we must have that  $\lambda_1$  and  $\lambda_2$  are positive, and we thus require that

$$\text{trace}(P) = \frac{f_1'^2 - 2f_2'f_1' + f_2'^2 + 4}{4 - 4f_1'f_2'} > 0, \quad \text{det}(P) = \frac{f_1'^2 - 2f_2'f_1' + f_2'^2 + 4}{16 - 16f_1'f_2'} > 0.$$

We see that  $trace(P) = 4 det(P)$  and the numerator of the expressions is just  $(f_1 - f_2)^2 + 4 > 0$ , so it suffices to check the sign of  $1 - f_1' f_2'$ . In particular, for  $P$  to be positive definite, we require that

$$f'(z_{1e})f'(z_{2e}) < 1.$$

We can now make use of the expressions for  $f'$  defined earlier and evaluate at the approximate locations of the equilibrium points derived in equation  $(4.17)$ . For the equilibrium points where  $z_{1e} \neq z_{2e}$ , we can show that

$$f'(z_{1e})f'(z_{2e}) \approx f'(\mu)f'(\frac{1}{\mu^{n-1}}) = \frac{-\mu n\mu^{n-1}}{(1+\mu^n)^2} \cdot \frac{-\mu n\mu^{-(n-1)^2}}{1+\mu^{-n(n-1)}} \approx n^2\mu^{-n^2+n}.$$

Using  $n = 2$  and  $\mu \approx 200$  from Exercise 2.9, we see that  $f'(z_{1e}) f'(z_{2e}) \ll 1$  and hence  $P$  is a positive definite. This implies that  $V$  is a positive definite function and hence a potential Lyapunov function for the system.

To determine if the system (4.16) is stable, we now compute  $\dot{V}$  at the equilibrium

![](_page_6_Figure_1.jpeg)

Figure 4.15: Dynamics of a genetic switch. The phase portrait on the left shows that the switch has three equilibrium points, corresponding to protein A having a concentration greater than, equal to or less than protein B. The equilibrium point with equal protein concentrations is unstable, but the other equilibrium points are stable. The simulation on the right shows the time response of the system starting from two different initial conditions. The initial portion of the curve corresponds to initial concentrations  $z(0) = (1, 5)$  and converges to the equilibrium where  $z_{1e} < z_{2e}$ . At time  $t = 10$ , the concentrations are perturbed by  $+2$  in  $z_1$  and  $-2$  in  $z_2$ , moving the state into the region of the state space whose solutions converge to the equilibrium point where  $z_{2e} < z_{1e}$ .

point. By construction,

$$\begin{split} \dot{V} &= w^T (PA + A^T P) w + \tilde{F}^T (w) P w + w^T P \tilde{F} (w) \\ &= -w^T w + \tilde{F}^T (w) P w + w^T P \tilde{F} (w). \end{split}$$

Since all terms in  $\tilde{F}$  are quadratic or higher order in  $w$ , it follows that  $\tilde{F}^{T}(w)Pw$ and  $w^T P \tilde{F}(w)$  consist of terms that are at least third order in w. Therefore if w is sufficiently close to zero, then the cubic and higher-order terms will be smaller than the quadratic terms. Hence, sufficiently close to  $w = 0$ ,  $\dot{V}$  is negative definite, allowing us to conclude that these equilibrium points are both stable.

Figure 4.15 shows the phase portrait and time traces for a system with  $\mu = 4$ , illustrating the bistable nature of the system. When the initial condition starts with a concentration of protein B greater than that of A, the solution converges to the equilibrium point at (approximately)  $(1/\mu^{n-1}, \mu)$ . If A is greater than B, then it goes to  $(\mu, 1/\mu^{n-1})$ . The equilibrium point with  $z_{1e} = z_{2e}$  is unstable.  $\nabla$ 

More generally, we can investigate what the linear approximation tells about the stability of a solution to a nonlinear equation. The following theorem gives a partial answer for the case of stability of an equilibrium point.

**Theorem 4.3.** Consider the dynamical system (4.15) with  $F(0) = 0$  and  $\tilde{F}$  such that  $\lim \|\tilde{F}(x)\|/\|x\| \to 0$  as  $\|x\| \to 0$ . If the real parts of all eigenvalues of A are strictly less than zero, then  $x_e = 0$  is a locally asymptotically stable equilibrium  $point of equation (4.15).$ 

This theorem implies that asymptotic stability of the linear approximation implies *local* asymptotic stability of the original nonlinear system. The theorem is very important for control because it implies that stabilization of a linear approximation of a nonlinear system results in a stable equilibrium for the nonlinear system. The proof of this theorem follows the technique used in Example 4.11. A formal proof can be found in [Kha01].

# Krasovski–Lasalle Invariance Principle

For general nonlinear systems, especially those in symbolic form, it can be difficult to find a positive definite function  $V$  whose derivative is strictly negative definite. The Krasovski–Lasalle theorem enables us to conclude the asymptotic stability of an equilibrium point under less restrictive conditions, namely, in the case where  $\dot{V}$ is negative semidefinite, which is often easier to construct. However, it applies only to time-invariant or periodic systems. This section makes use of some additional concepts from dynamical systems; see Hahn [Hah67] or Khalil [Kha01] for a more detailed description.

We will deal with the time-invariant case and begin by introducing a few more definitions. We denote the solution trajectories of the time-invariant system

$$\frac{dx}{dt} = F(x) \tag{4.18}$$

as  $x(t : a)$ , which is the solution of equation (4.18) at time t starting from a at  $t_0 = 0$ . The  $\omega$  limit set of a trajectory  $x(t; a)$  is the set of all points  $z \in \mathbb{R}^n$  such that there exists a strictly increasing sequence of times  $t_n$  such that  $x(t_n; a) \to z$ as  $n \to \infty$ . A set  $M \subset \mathbb{R}^n$  is said to be an *invariant set* if for all  $b \in M$ , we have  $x(t; b) \in M$  for all  $t > 0$ . It can be proved that the  $\omega$  limit set of every trajectory is closed and invariant. We may now state the Krasovski-Lasalle principle.

**Theorem 4.4** (Krasovski–Lasalle principle). Let  $V : \mathbb{R}^n \to \mathbb{R}$  be a locally positive definite function such that on the compact set  $\Omega_r = \{x \in \mathbb{R}^n : V(x) < r\}$  we have  $V(x) \leq 0$ . Define

$$S = \{x \in \Omega_r : V(x) = 0\}.$$

As  $t \to \infty$ , the trajectory tends to the largest invariant set inside S; i.e., its  $\omega$  limit set is contained inside the largest invariant set in S. In particular, if S contains no invariant sets other than  $x = 0$ , then 0 is asymptotically stable.

Proofs are given in [Kra63] and [LaS60].

Lyapunov functions can often be used to design stabilizing controllers, as is illustrated by the following example, which also illustrates how the Krasovski-Lasalle principle can be applied.

## **Example 4.12 Inverted pendulum**

Following the analysis in Example 2.7, an inverted pendulum can be described by the following normalized model:

$$\frac{dx_1}{dt} = x_2, \qquad \frac{dx_2}{dt} = \sin x_1 + u \cos x_1, \tag{4.19}$$

![](_page_8_Figure_1.jpeg)

**Figure 4.16:** Stabilized inverted pendulum. A control law applies a force  $u$  at the bottom of the pendulum to stabilize the inverted position (a). The phase portrait (b) shows that the equilibrium point corresponding to the vertical position is stabilized. The shaded region indicates the set of initial conditions that converge to the origin. The ellipse corresponds to a level set of a Lyapunov function  $V(x)$  for which  $V(x) > 0$  and  $\dot{V}(x) < 0$  for all points inside the ellipse. This can be used as an estimate of the region of attraction of the equilibrium point. The actual dynamics of the system evolve on a manifold (c).

where  $x_1$  is the angular deviation from the upright position and u is the (scaled) acceleration of the pivot, as shown in Figure 4.16a. The system has an equilibrium at  $x_1 = x_2 = 0$ , which corresponds to the pendulum standing upright. This equilibrium is unstable.

To find a stabilizing controller we consider the following candidate for a Lyapunov function:

$$V(x) = (\cos x_1 - 1) + a(1 - \cos^2 x_1) + \frac{1}{2}x_2^2 \approx (a - \frac{1}{2})x_1^2 + \frac{1}{2}x_2^2.$$

The Taylor series expansion shows that the function is positive definite near the origin if  $a > 0.5$ . The time derivative of  $V(x)$  is

$$V = -\dot{x}_1 \sin x_1 + 2a\dot{x}_1 \sin x_1 \cos x_1 + \dot{x}_2 x_2 = x_2(u + 2a \sin x_1) \cos x_1.$$

Choosing the feedback law

$$u = -2a\sin x_1 - x_2\cos x_1$$

gives

$$\dot{V} = -x_2^2 \cos^2 x_1.$$

It follows from Lyapunov's theorem that the equilibrium is locally stable. However, since the function is only negative semidefinite, we cannot conclude asymptotic stability using Theorem 4.2. However, note that  $V = 0$  implies that  $x_2 = 0$  or  $x_1 = \pi/2 \pm n\pi$ .

If we restrict our analysis to a small neighborhood of the origin  $\Omega_r$ ,  $r \ll \pi/2$ , then we can define

$$S = \{(x_1, x_2) \in \Omega_r : x_2 = 0\}$$

and we can compute the largest invariant set inside  $S$ . For a trajectory to remain in this set we must have  $x_2 = 0$  for all t and hence  $\dot{x}_2(t) = 0$  as well. Using the dynamics of the system (4.19), we see that  $x_2(t) = 0$  and  $\dot{x}_2(t) = 0$  implies  $x_1(t) = 0$ 0 as well. Hence the largest invariant set inside S is  $(x_1, x_2) = 0$ , and we can use the Krasovski–Lasalle principle to conclude that the origin is locally asymptotically stable. A phase portrait of the closed loop system is shown in Figure 4.16b.

In the analysis and the phase portrait, we have treated the angle of the pendulum  $\theta = x_1$  as a real number. In fact,  $\theta$  is an angle with  $\theta = 2\pi$  equivalent to  $\theta = 0$ . Hence the dynamics of the system actually evolves on a *manifold* (smooth surface) as shown in Figure 4.16c. Analysis of nonlinear dynamical systems on manifolds is more complicated, but uses many of the same basic ideas presented here.  $\nabla$ 

# 4.5 Parametric and Nonlocal Behavior

Most of the tools that we have explored are focused on the local behavior of a fixed system near an equilibrium point. In this section we briefly introduce some concepts regarding the global behavior of nonlinear systems and the dependence of a system's behavior on parameters in the system model.

## **Regions of Attraction**

To get some insight into the behavior of a nonlinear system we can start by finding the equilibrium points. We can then proceed to analyze the local behavior around the equilibria. The behavior of a system near an equilibrium point is called the *local* behavior of the system.

The solutions of the system can be very different far away from an equilibrium point. This is seen, for example, in the stabilized pendulum in Example 4.12. The inverted equilibrium point is stable, with small oscillations that eventually converge to the origin. But far away from this equilibrium point there are trajectories that converge to other equilibrium points or even cases in which the pendulum swings around the top multiple times, giving very long oscillations that are topologically different from those near the origin.

To better understand the dynamics of the system, we can examine the set of all initial conditions that converge to a given asymptotically stable equilibrium point. This set is called the *region of attraction* for the equilibrium point. An example is shown by the shaded region of the phase portrait in Figure 4.16b. In general, computing regions of attraction is difficult. However, even if we cannot determine the region of attraction, we can often obtain patches around the stable equilibria that are attracting. This gives partial information about the behavior of the system.

One method for approximating the region of attraction is through the use of Lyapunov functions. Suppose that  $V$  is a local Lyapunov function for a system around an equilibrium point  $x_0$ . Let  $\Omega_r$  be a set on which  $V(x)$  has a value less than r,

$$\Omega_r = \{x \in \mathbb{R}^n : V(x) \le r\},\$$