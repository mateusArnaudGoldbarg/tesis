import numpy as np
from scipy.sparse import random
from time import time

# Generate a large sparse matrix with 8-bit integer values
n = 10
sparsity = 0.70
density = 1-sparsity


A = random(n, n, density, format='csr', dtype=np.int8)
B = random(n, n, density, format='csr', dtype=np.int8)

print(A.toarray())

# Sparse matrix multiplication with MKL
t1 = time()
C_sparse = A.dot(B)
t2 = time()

# Dense matrix multiplication with numpy
t3 = time()
C_dense = np.dot(A.toarray(), B.toarray())
t4 = time()

print(type(A.toarray()))

print("Sparse multiplication time:", t2 - t1)
print("Dense multiplication time:", t4 - t3)
