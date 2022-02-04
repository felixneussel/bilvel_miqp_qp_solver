# Documentation



This project is part of my Bachelor thesis. I implemented solvers for bilevel optimization problems with a mixed-integer quadratic leader problem and a quadratic follower problem. They are implementations from the algorithms proposed by Thomas Kleinert, Verkonika Grimm and Martin Schmidt in the paper "Outer approximation techniques for the global opimization of mixed-integer quadratic bilevel problems".

Concretely, the algorithms solve problems of the form

<!-- $$
\begin{split}
\min_{x,\bar{y}} \; & q_u(x,\bar{y}) = \frac{1}{2} x^\top H_u x + c_u^\top x
+ \frac{1}{2} \bar{y}^\top G_u \bar{y} + d_u^\top \bar{y}\\
s.t. \; & A x + B \bar{y} \geq a,\\
& x_i \in \mathbb{Z} \cap [x_i^-, x_i^+] \; \forall i \in I := \{1, ..., |I| \},\\
& x_i \in \mathbb{R} \; \forall i \in R := \{|I| + 1, ..., n_x \},\\
& \bar{y} \in \argmin_y \{q_l (y) = \frac{1}{2}y^\top G_l y + d_l^\top y : C x_I + Dy \geq b, y \in \mathbb{R}^{n_y}\}
\end{split}
$$ --> 

<div align="center"><img style="background: white;" src="https://render.githubusercontent.com/render/math?math=%5Cbegin%7Bsplit%7D%0A%5Cmin_%7Bx%2C%5Cbar%7By%7D%7D%20%5C%3B%20%26%20q_u(x%2C%5Cbar%7By%7D)%20%3D%20%5Cfrac%7B1%7D%7B2%7D%20x%5E%5Ctop%20H_u%20x%20%2B%20c_u%5E%5Ctop%20x%0A%2B%20%5Cfrac%7B1%7D%7B2%7D%20%5Cbar%7By%7D%5E%5Ctop%20G_u%20%5Cbar%7By%7D%20%2B%20d_u%5E%5Ctop%20%5Cbar%7By%7D%5C%5C%0As.t.%20%5C%3B%20%26%20A%20x%20%2B%20B%20%5Cbar%7By%7D%20%5Cgeq%20a%2C%5C%5C%0A%26%20x_i%20%5Cin%20%5Cmathbb%7BZ%7D%20%5Ccap%20%5Bx_i%5E-%2C%20x_i%5E%2B%5D%20%5C%3B%20%5Cforall%20i%20%5Cin%20I%20%3A%3D%20%5C%7B1%2C%20...%2C%20%7CI%7C%20%5C%7D%2C%5C%5C%0A%26%20x_i%20%5Cin%20%5Cmathbb%7BR%7D%20%5C%3B%20%5Cforall%20i%20%5Cin%20R%20%3A%3D%20%5C%7B%7CI%7C%20%2B%201%2C%20...%2C%20n_x%20%5C%7D%2C%5C%5C%0A%26%20%5Cbar%7By%7D%20%5Cin%20%5Cargmin_y%20%5C%7Bq_l%20(y)%20%3D%20%5Cfrac%7B1%7D%7B2%7Dy%5E%5Ctop%20G_l%20y%20%2B%20d_l%5E%5Ctop%20y%20%3A%20C%20x_I%20%2B%20Dy%20%5Cgeq%20b%2C%20y%20%5Cin%20%5Cmathbb%7BR%7D%5E%7Bn_y%7D%5C%7D%0A%5Cend%7Bsplit%7D"></div>

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
