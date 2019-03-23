import numpy as np
from numpy import linalg as LA

m, n = 4, 5
A = np.random.rand(m, n)

U, S, V = LA.svd(A)

# checking if U, V are orthogonal and S is a diagonal matrix with
# nonnegative decreasing elements
print(A, '\n')
print(U, '\n')
print(S, '\n')
print(V, '\n')
