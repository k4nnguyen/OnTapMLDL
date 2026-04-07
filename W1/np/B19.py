"""
Code Random Forest
"""
import numpy as np
n_samples = 10
original_indices = np.arange(n_samples)
tree1_data = np.random.choice(original_indices, size=n_samples, replace=True)
print(f"Dữ liệu cho Cây 1: {tree1_data}")
tree2_data = np.random.choice(original_indices, size=n_samples, replace=True)
print(f"Dữ liệu cho Cây 2: {tree2_data}")