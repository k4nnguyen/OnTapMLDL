"""
Viết lại hàm z trong mạng neuron và hàm sigmoid : 1/(1+e^(-z))
"""
import numpy as np
x = np.array([1.5, 2.0, -1.0, 0.5])
W = np.random.randn(3, 4)
b = np.array([0.1, 0.2, -0.1])
z = W@x + b
sigmoid = 1 / (1+ np.exp(-z))
print(f"z = {z}\nHàm sigmoid: {sigmoid}")