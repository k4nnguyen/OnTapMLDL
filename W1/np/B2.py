"""
Cho một mảng 1D chứa các nhãn phân loại dưới dạng số nguyên. 
Hãy chuyển đổi mảng này thành một ma trận one-hot (one-hot encoded matrix) chỉ sử dụng các hàm cơ bản của NumPy.
"""
import numpy as np
labels = np.array([0, 2, 1, 2, 0, 3])
result = np.zeros((len(labels),np.max(labels)+1))
result[np.arange(len(labels)),labels] = 1

print(result)