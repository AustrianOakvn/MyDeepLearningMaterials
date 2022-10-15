import numpy as np

X = np.array([4, 3])
# L_0 norm is the number of element that is not zero


l_0_norm = np.linalg.norm(X, ord=0)
print(l_0_norm)

# L_1 is the Mahhatan distance

l_1_norm = np.linalg.norm(X, ord=1)
print(l_1_norm)

# L_2 norm is the Euclidean distance
l_2_norm = np.linalg.norm(X, ord=2)
print(l_2_norm)