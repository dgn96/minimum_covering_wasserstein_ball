import numpy as np
import cvxpy as cp

def fixed_support_min_wasserstein_ball(p_k, c):
	"""
	To say something here
	
	"""

	# Define number of distributions
	K = len(p_k)

	# Define number of atoms per sample
	N = len(p_k[0])

	# Dec Var
	T = [cp.Variable((N, N)) for k in range(K)]
	epsilon = cp.Variable((1, 1))
	p = cp.Variable((N))

	# Obj
	obj = cp.Minimize(epsilon)

	# Cons
	for k in range(K):
    		cons = [T[k] >= 0]
    		#cons.append(sum(sum(cp.multiply(T[k], c))) <= epsilon)
    		cons.append(cp.trace(c.T@T[k]) <= epsilon)
    		cons.append(T[k]@np.ones(N) == p)
    		cons.append(T[k].T@np.ones(N) == p_k[k])

	# Solve program
	problem = cp.Problem(obj, cons)
	problem.solve()

	return p.value
