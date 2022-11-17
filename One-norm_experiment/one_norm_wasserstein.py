import numpy as np
import cvxpy as cp
from itertools import product

def one_norm_wasserstein(xi, p):
	# Define number of distributions
	K = xi.shape[0]

	# Define number of atoms per sample
	N = xi.shape[1]

	# Dec Var
	mu = cp.Variable((K, N))
	lam = cp.Variable((K, 1))

	# Obj
	obj = cp.Maximize(-sum(sum(cp.multiply(mu, p))))

	# Cons
	cons = [lam >= 0]
	cons = [sum(lam) == 1]
	for alpha in product(*[range(len(xi[k])) for k in range(K)]):
    		for xi_0 in product(*[xi[k] for k in range(K)]):
        		cons.append(sum(mu[:, alpha[:]] + lam*cp.norm(xi_0 - xi[:, alpha[:]], 1)) >= 0)


	# Solve program
	problem = cp.Problem(obj, cons)
	problem.solve()