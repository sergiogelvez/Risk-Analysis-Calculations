import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


B = np.array([[0 , 1 , 2 , 0 , 2 , 0 , 1 , 0 , 0 , 0 , 0 , 3 , 1 , 1 , 2 , 2 , 2 , 2 , 2 , 2 , 2 , 2 , 2 , 2 , 0 , 0],\
[2 , 0 , 2 , 1 , 1 , 0 , 0 , 0 , 0 , 1 , 0 , 0 , 0 , 0 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 3 , 1 , 1],\
[1 , 1 , 0 , 0 , 1 , 0 , 0 , 0 , 0 , 0 , 0 , 1 , 1 , 1 , 2 , 2 , 2 , 2 , 2 , 2 , 2 , 2 , 2 , 2 , 1 , 0],\
[0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 1 , 0 , 2 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 0],\
[1 , 0 , 1 , 0 , 0 , 0 , 0 , 0 , 0 , 1 , 0 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 0],\
[0 , 1 , 0 , 0 , 4 , 0 , 0 , 0 , 0 , 1 , 0 , 0 , 2 , 2 , 2 , 2 , 2 , 2 , 2 , 2 , 2 , 2 , 2 , 2 , 0 , 0],\
[1 , 1 , 1 , 2 , 0 , 0 , 0 , 1 , 3 , 2 , 1 , 1 , 3 , 3 , 3 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 2 , 3 , 1],\
[2 , 1 , 2 , 0 , 0 , 0 , 1 , 0 , 2 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 2 , 2],\
[0 , 1 , 1 , 2 , 0 , 0 , 0 , 0 , 0 , 1 , 0 , 0 , 3 , 2 , 4 , 3 , 3 , 3 , 3 , 3 , 3 , 3 , 3 , 1 , 4 , 2],\
[1 , 1 , 1 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0],\
[1 , 3 , 1 , 1 , 0 , 0 , 2 , 2 , 2 , 2 , 0 , 2 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1],\
[3 , 4 , 1 , 0 , 1 , 0 , 0 , 2 , 1 , 0 , 0 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 4 , 1 , 1 , 1 , 2 , 1 , 0],\
[1 , 1 , 1 , 0 , 2 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 3 , 1 , 1 , 1 , 1 , 1 , 3 , 0 , 1 , 0 , 0 , 0],\
[1 , 1 , 0 , 0 , 2 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 0 , 0],\
[1 , 1 , 0 , 0 , 2 , 0 , 0 , 0 , 1 , 0 , 0 , 0 , 2 , 2 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 0 , 0],\
[0 , 0 , 0 , 0 , 1 , 0 , 0 , 0 , 1 , 0 , 0 , 0 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 0],\
[0 , 0 , 0 , 0 , 1 , 0 , 0 , 0 , 1 , 0 , 0 , 0 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 0 , 0],\
[0 , 0 , 0 , 0 , 1 , 0 , 0 , 0 , 1 , 0 , 0 , 0 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 0 , 0 , 0],\
[0 , 0 , 0 , 0 , 1 , 0 , 0 , 0 , 1 , 0 , 0 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 0 , 0 , 0],\
[3 , 4 , 1 , 0 , 1 , 0 , 0 , 0 , 4 , 0 , 0 , 0 , 0 , 0 , 0 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 0 , 0 , 0],\
[0 , 0 , 0 , 2 , 1 , 0 , 0 , 0 , 1 , 0 , 0 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 0 , 0 , 0],\
[0 , 0 , 0 , 0 , 1 , 0 , 0 , 0 , 1 , 0 , 0 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 0 , 0],\
[0 , 0 , 0 , 0 , 1 , 0 , 0 , 0 , 1 , 0 , 0 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 0 , 0],\
[0 , 0 , 0 , 0 , 1 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 0 , 0],\
[0 , 0 , 0 , 0 , 1 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 3 , 3 , 3 , 3 , 3 , 3 , 3 , 3 , 3 , 3 , 3 , 0 , 0 , 0],\
[1 , 2 , 0 , 2 , 1 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 1 , 0 , 0]])

print(B)
rows, columns = B.shape 

# normalisation of direct influence matrix

print(rows, columns)
sums = np.zeros(rows)

for i in range(rows):
    sums[i] = sum(B[i, :])

max_sums_row = max(sums)
print(max_sums_row)

C = B / max_sums_row

print(C)
    
# C is the normalised matrix

I = np.identity(rows)
print(I) 

D = C @ np.linalg.inv(I - C)
print(D)

impact_degree = np.zeros(rows)
impact_risk_factor = np.zeros(columns)

# calculate impact dregree and impact risk for factors

for i in range(rows) :
    impact_degree[i] = sum(D[i, :])
    impact_risk_factor[i] = sum(D[:, i])

print(impact_degree)
print(impact_risk_factor)

# calculate centrality and causality

centrality = impact_degree + impact_risk_factor
causality = impact_degree - impact_risk_factor

print(centrality)
print(causality)

# Causal diagram
plt.plot(centrality, causality, "*r")
plt.show()

# Calculate overall influence matrix

H = np.identity(rows) + D
print(H)

# threslhold for reachability
l = 0.06

# arr[arr > threshold] = 0
print(H.shape)
K = np.empty_like(H, dtype=np.int32)

for i in range(rows):
    for j in range(columns):
        if H[i, j] >= l :
            K[i, j] = 1
        else :
            K[i, j] = 0

print("K")
print(K)

R = np.zeros(rows)
S = np.zeros(rows)

'''
for i in range(rows):
    for j in range(columns):
        if K[i, j] != 0 :
            
'''