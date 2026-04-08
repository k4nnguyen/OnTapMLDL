"""
Tự viết lại thuật toán Gradient Descent (Linear Regression) 
y = wx + b
Ta cần phải đưa hai tham số w và b sao cho y^ gần với y nhất:

Ta có hàm Loss (mục tiêu để biết có bị khác biệt nhiều không? Loss càng nhỏ càng tốt)
L = (1/n) * Sigma(yi^ - yi)^2
Sau đó, ta sẽ muốn biết khi thay đổi các tham số như w hay b thì sẽ ảnh hưởng đến hàm Loss như nào (tăng / giảm)
Với L như trên, có thể thay vào: L = (1/n) * Sigma(w*xi + b - yi)^2 
dw = @L / @W (Đạo hàm L / Đạo hàm W): (2/n) * Sigma(w*xi + b - yi) * xi <=> (2/n) * Sigma(xi*(yi^ - yi))
db = @L / @b (Đạo hàm L / Đạo hàm b): (2/n) * Sigma(w*xi + b - yi) <=> (2/n) * Sigma(yi^ - yi)
"""
import numpy as np

# Mục tiêu tìm ra y = 3x+5
x = np.array([1, 2, 3, 4, 5]).reshape(-1,1)
y = np.array([8, 11, 14, 17, 20]).reshape(-1,1)

# Tham số để máy học
lr = 0.01 
epochs = 101
n = len(x)

x = np.concatenate((x,np.ones((n,1))), axis=1) # Shape (n,2)
theta = np.zeros((2,1))
for i in range(epochs):
    y_hat =  x@theta  # Với y^ là y dự đoán, shape = (n,1)
    
    loss = y_hat - y  # Lấy y_pred - y sẽ ra phần loss của từng giá trị
    d_theta = (2/n) * x.T @ loss  # Tính theta, bao gồm cả W lẫn b
    
    theta = theta - lr * d_theta
    
    if(i%5 == 0):
        print(f"Epoch {i}:")
        print(f"y_hat: {y_hat}\nd_theta: {d_theta}\ntheta: {theta}\n\n")
        
        