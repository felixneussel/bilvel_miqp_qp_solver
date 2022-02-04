# bilvel_miqp_qp_solver

Watch out: Major Error in the Solver that doesn't ues OOP (Everything thats not in the folder 'Solver_OOP'). ![$G_l$] and $G_u$ as well as $d_l$ and $d_u$ are treated as if they were the same. Please only use the Files in the Foler 'Solver_OOP'.

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
