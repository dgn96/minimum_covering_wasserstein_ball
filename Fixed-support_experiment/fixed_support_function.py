import numpy as np
import cvxpy as cp

def fixed_support_min_wasserstein_ball(xi, p_k,):
	# Define number of distributions
	K = xi.shape[0]

	# Define number of atoms per sample
	N = xi.shape[1]

	# Dec Var
	T = cp.Variable((N, N))
	epsilon = cp.Variable((1, 1))
	p = cp.Variable((N))

	# Obj
	obj = cp.Minimize(epsilon)

	# Cons
	cons = [T >= 0]
	cons.append(cp.trace(T@c) <= epsilon)
	cons.append(T@np.ones(N) == p)
	cons.append(T.T@np.ones(N) == pi)

	# Solve program
	problem = cp.Problem(obj, cons)
	problem.solve()