# Documentation



This work-in-process project is part of my Bachelor thesis. I implemented solvers for bilevel optimization problems with a mixed-integer quadratic leader problem and a quadratic follower problem. They are implementations from the algorithms proposed by Thomas Kleinert, Verkonika Grimm and Martin Schmidt in the paper "Outer approximation techniques for the global opimization of mixed-integer quadratic bilevel problems".

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

with positive semidefinite matrices <!-- $H_u \in \R^{n_x \times n_x}$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=H_u%20%5Cin%20%5CR%5E%7Bn_x%20%5Ctimes%20n_x%7D">, <!-- $G_u \in \R^{n_y \times n_y}$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=G_u%20%5Cin%20%5CR%5E%7Bn_y%20%5Ctimes%20n_y%7D">, <!-- $G_l \in \R^{n_y \times n_y}$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=G_l%20%5Cin%20%5CR%5E%7Bn_y%20%5Ctimes%20n_y%7D"> and vectors <!-- $c_u \in \R^{n_x}$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=c_u%20%5Cin%20%5CR%5E%7Bn_x%7D">, <!-- $d_u, d_l \in \R^{n_y}$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=d_u%2C%20d_l%20%5Cin%20%5CR%5E%7Bn_y%7D">, <!-- $a \in \R^{m_u}$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=a%20%5Cin%20%5CR%5E%7Bm_u%7D">, <!-- $b \in \R^{m_l}$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=b%20%5Cin%20%5CR%5E%7Bm_l%7D"> and matrices <!-- $A \in \R^{m_u \times n_x}$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=A%20%5Cin%20%5CR%5E%7Bm_u%20%5Ctimes%20n_x%7D">, <!-- $B \in \R^{m_u \times n_y}$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=B%20%5Cin%20%5CR%5E%7Bm_u%20%5Ctimes%20n_y%7D">, <!-- $C \in \R^{m_l \times n_x}$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=C%20%5Cin%20%5CR%5E%7Bm_l%20%5Ctimes%20n_x%7D"> and <!-- $D \in \R^{m_l \times n_y}$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=D%20%5Cin%20%5CR%5E%7Bm_l%20%5Ctimes%20n_y%7D">.

The different solution approaches are implemented in [Solvers/multitree.py](Solvers/multitree.py).

In [main.py](main.py), there is a demonstration how to use the solver.
The specific algorithm to be used can be set with the algorithm parameter.
These are the different algorithms:

|  Algorithm | Description  |
|---|---|
|  MT | Multi-tree approach  |
| MT-K  |  Like MT with addidtional Kelley-type cutting planes |
|  MT-K-F | Like MT-K but the master problem is terminated with the first improving integer solution  |
|  MT-K-F-W |  Like MT-K-F but the master problem is warmstarted with the current best solution |
| ST  |  Single-tree approach |
| ST-K  |  Like ST with additional Kelley-type cutting planes |
|  ST-K-C |  Like ST-K but a bilevel feasible solution is used to derive an initial strong-duality cut |
| ST-K-C-S  |  Like ST-K-C but the initial master problem is warmstarted with a bilevel feasible solution |

Numerical studies have shown that MT-K-F-W and ST-K-C-S are the best performing algorithm.


