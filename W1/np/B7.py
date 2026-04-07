"""
Cho một ma trận 2D có chứa các giá trị np.nan. Hãy điền các giá trị NaN này bằng giá trị trung bình của cột chứa nó.
"""
import numpy as np
X = np.array([
    [1, 2, np.nan],
    [4, np.nan, 6],
    [7, 8, 9]
])
mean_value = np.nanmean(X)
idx = np.where(np.isnan(X))
print(idx)
X[idx] = mean_value
print(X)