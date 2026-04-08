"""
Cho mảng đầu vào x, và trọng số w,b.
Viết code để tính Feed Forward và Backpropagation

Các mạng neuron sẽ có các lớp như: Input layer, Hidden layer và Output layer.
Input layer nhận các đầu vào, tính toán ở các lớp Hidden layer (Tính tổng tuyến tính và hàm kích hoạt),
và đưa kết quả ở Output layer. 
Tính tổng tuyến tính: z = wx + b, và đưa qua hàm kích hoạt như sigmoid, ReLU, sin, cos, ...

Thực hiện Feed forward để có thể dự đoán y^, và lưu lại các giá trị trung gian z ở các node
Thực hiện Back Propagation để có thể điều chỉnh trọng số w,b từ việc tính Loss Function và sử dụng Chain rule.
Giả sử với hàm activation đơn giản như sin(x), ta có chain rule: dy/dw = dy/dz * dz/dw
với dy/dz = dz(sin(z)) = cos(z)
    dz/dw = dw(wx+b) = x
-> dy/dw = cos(z) * x
"""
import numpy as np
x = np.array([0, np.pi/4, np.pi/2, np.pi]) 
w = 2.0
b = 0.5

# Tính feed forward
z = w*x+b 
print(f"Các giá trị sau khi tính tổng tuyến tính: {z}")

# Cho z đi qua activation function
y_pred = np.sin(z)
print(f"Output: {y_pred}")

# BackPropagation
dy_dz = np.cos(z)
dz_dw = x 

dy_dw = dy_dz * dz_dw

print(f"Gradient của trọng số w là: {dy_dw}")
