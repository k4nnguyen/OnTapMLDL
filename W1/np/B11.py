"""
Tìm các vector riêng và trị riêng của các vector, vector lớn nhất và trị riêng của nó
"""
import numpy as np
A = np.array([[4, 2], [2, 3]])

eigen_vector, eigen_values = np.linalg.eig(A) # Do np.linalg.eig(A) trả về 2 giá trị là vectors và values

print(f"Vector riêng của A: {eigen_vector}")
print(f"Trị riêng của A: {eigen_values}")

max_vals = np.max(eigen_values)
idx = np.argmax(eigen_values)
print(f"Trị riêng lớn nhất: {max_vals}")
print(f"Vector ứng với trị riêng lớn nhất: {eigen_vector[idx]}")