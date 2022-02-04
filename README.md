# Documentation



This project is part of my Bachelor thesis. I implemented solvers for bilevel optimization problems with a mixed-integer quadratic leader problem and a quadratic follower problem. They are implementations from the algorithms proposed by Thomas Kleinert, Verkonika Grimm and Martin Schmidt in the paper "Outer approximation techniques for the global opimization of mixed-integer quadratic bilevel problems".

Concretely, the algorithms solve problems of the form

$$
\begin{split}
\min_{x,\bar{y}} \; & q_u(x,\bar{y}) = \frac{1}{2} x^\top H_u x + c_u^\top x
+ \frac{1}{2} \bar{y}^\top G_u \bar{y} + d_u^\top \bar{y}\\
s.t. \; & A x + B \bar{y} \geq a,\\
& x_i \in \mathbb{Z} \cap [x_i^-, x_i^+] \; \forall i \in I := \{1, ..., |I| \},\\
& x_i \in \mathbb{R} \; \forall i \in R := \{|I| + 1, ..., n_x \},\\
& \bar{y} \in \argmin_y \{q_l (y) = \frac{1}{2}y^\top G_l y + d_l^\top y : C x_I + Dy \geq b, y \in \mathbb{R}^{n_y}\}
\end{split}
$$

with positive semidefinite matrices $H_u \in \R^{n_x \times n_x}$, $G_u \in \R^{n_y \times n_y}$, $G_l \in \R^{n_y \times n_y}$ and vectors $c_u \in \R^{n_x}$, $d_u, d_l \in \R^{n_y}$, $a \in \R^{m_u}$, $b \in \R^{m_l}$ and matrices $A \in \R^{m_u \times n_x}$, $B \in \R^{m_u \times n_y}$, $C \in \R^{m_l \times n_x}$ and $D \in \R^{m_l \times n_y}$.

# Ideas

- Maybe reduced problems stay feasible if only number of int variables is reduced

# Remarks

- p0282-0.500000 solvable with big-M of 1e6
- Reduction of n_I only makes problem infeasible too. Maybe their removal adds too much slack for lower level constraints.

# TODO

- MT-K-F: Kelley cuts are added for every non-improving integer solution too.

## Errors to fix

- harp2-0.500000 with ST-K-C-S took longer than 900s but had a measured running time of 810 s
- Sometimes Multitree produces negative Gap, e.g. for lseu-0.900000. This causes ST-K-C-S to deem problem infeasible. Especially severe for remark_1.
    - I observed constraint violations in remark_1 subproblem of maginitude 1e-3, while there is no violation for remark_2.
- Unsolved Issue : p0033-0.500000 has better obj val in MT than in other approaches. If corresponding integer solution is set as x_I_param in debugging of other functions, they also have the same (better obj val)
