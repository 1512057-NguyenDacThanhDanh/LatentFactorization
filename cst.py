import csv
import pandas as pd
import threading
import numpy as np
import math

from numpy import linalg as LA
'''
dataset = pd.read_csv('./Matrix4.csv', sep=',').values

A = dataset[:,1:]
A = A.astype(float)
#m, n = 2, 3
#A = np.random.rand(m, n)
print(A)  


U, S, V = LA.svd(A)

print()
print ('U: \n', U)
print()
print ('S: \n', S)
print()

print ('V: \n', V)
print()
print(U.dot(S).dot(V))

#new_a = U.dot(S.dot(V.T))
#print(U.dot(U.T) - np.eye(8))
'''
