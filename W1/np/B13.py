"""
Viết lại Hàm ReLU(x) với x > 0  -> x; x <= 0 -> 0.
(Chỉ cần trả về mảng bit với 1 nếu > 0 và 0 nếu ngược lại)
"""
import numpy as np
X = np.array(
    [1,-3,5,1,2,0,-9,-23]
)

def relu_grad(x):
    return np.where(x>0,1,0)

print(relu_grad(X))