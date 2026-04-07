"""
Tự viết lại thuật toán Logistic Regression, thuật toán được áp dụng trong bài toán phân loại (Classification)

Ta vẫn tính tổng tuyến tính: z = wx + b, tuy vậy giờ sẽ được đi qua một hàm kích hoạt sigmoid để biến khoảng giá trị thành (0,1)
với y^ hay y_pred = sigmoid(z) = sigmoid(1/(1+e^(-z)))
Về Loss Function: Binary Cross-Entropy

Giải thích tại sao không dùng MSE cho bài toán Logistic:
- Logistic sinh ra để dự đoán xác suất trong khoảng (0,1)
- Vì vậy khi dùng MSE sẽ thực hiện: Loss = 1/2(y^ - y)^2, dL/dw = dL/dy^ * dy^/dz * dz/dw
dL / dy^ = y^ - y
dy^ / dz = y^(1-y^) (Tính chất hàm sigmoid)
dz / dw = x
--> dL/dw = x * y^ * (1-y^) * (y^-y) 

Tuy vậy sẽ dễ bị dính Vanishing Gradient (triệt tiêu đạo hàm), ở phần y^ * (1-y^).
Tưởng tượng chỉ cần y^ = 0.01, thì khi nhân vào sẽ thành 0.0099 (xấp xỉ 0) -> Các phần còn lại nhân vào sẽ gần bằng 0
Điều này khiến máy sẽ rất khó/không thể học đối với Loss Function như MSE.

Hướng giải quyết: Sử dụng Binary Cross-Entropy (BCE)
Với Loss Function: -1/n * Sigma(i=1->n) (yi*log(yi^) + (1-yi)*log(1-yi^))
Áp dụng Chain rule với BCE:
dL/dw = dL/dy^ * dy^/dz * dz/dw
dL/dy^ = (-y/y^) + [(1-y)/(1-y^)] = (y^-y) / y^(1-y^)
dy/dz = y^(1-y^)
dz/dw = x
--> dL/dw = x * y^(1-y^) * ((y^-y)/y^(1-y^)) = x*(y^-y) -> (1/n)*Sigma(xi*(yi^ - yi))
Như ta thấy, việc chuyển thành loss function là BCE giúp triệt tiêu phần (1-y^), tránh bị Vanishing gradient.

Bài toán: Xác suất để một học sinh qua học phần với X là số giờ học, sử dụng Logistic Regression
"""
import numpy as np
X = np.array([1,2,3,4,5,6,7]).reshape(-1,1) # Số giờ học (Shape (n,1))
y = np.array([0,0,0,1,1,1,1]).reshape(-1,1) # Đích hướng tới (0 là trượt, 1 là đạt)

n = len(X)
X = np.concatenate((X,np.ones((n,1))),axis = 1) # shape (n,2)
theta = np.zeros((2,1)) 

lr = 0.01
epochs = 5001

def sigmoid(z):
    return 1/(1+np.exp(-z))

for i in range(epochs):
    z = X @ theta   # Shape (n,1)
    y_pred = sigmoid(z) # y dự đoán
    loss = y_pred - y 
    d_theta = (1/n) * X.T @ loss
    theta = theta - lr * d_theta
    
    if (i % 5 == 0):
        loss = -np.mean(y*np.log(y_pred) + (1-y) * np.log(1-y_pred))
        print(f"Epochs {i}: Loss: {loss:.4f}")
    
print(f"Trọng số của w và b: {theta}")
predictions = sigmoid(X @ theta)
print(f"Xác suất qua môn của từng giờ học tương ứng:\n {np.round(predictions,2)}")