"""
Cho một mảng 2D bất kỳ (kích thước M X N). 
Hãy viết code để chuẩn hóa các giá trị của mảng này theo từng cột về khoảng [0, 1] mà không dùng vòng lặp.
"""
import numpy as np 
X = np.random.randint(1,100, size = (5,3))
print(X)
X_min = np.min(X, axis=0)
X_max = np.max(X, axis=0)

X_scaled = (X - X_min) / (X_max - X_min)

print(X_scaled)