"""
Code lại MLP (Fully connected layer / Dense layer)
(Loss sử dụng Cross Entropy) 
Giải thích lại phần Backpropagation:
Trước hết thì Feed forward là lúc máy chưa học được tham số gì và dự đoán ngay, backpropagation là giai đoạn truy ngược lại,
để kiểm tra lỗi và để máy học tham số

Feed forward trải qua giai đoạn: z1 = XW1 + b1 -> a1 = activation(z) --> z2 = a1W2 + b2 -> ... -> (Lặp lại quá trình) 
--> zn = a(n-1)W(n) -> output = softmax(zn) (Sử dụng softmax để biến thành các xác suất có tổng = 1)  

Backpropagation:
Nhắc lại hàm Loss Cross entropy: L = -sigma(yi*log(yi^)); softmax(z) = e^zi / sigma(e^zj)

1. Kết hợp softmax và Cross entropy (Tìm (dL/dz2))
Ta có dz2 = (dL / yi^) * (dyi^/ dz2)
- (dL / yi^) = -yi / yi^
Xét (dyi^ / dz2) có: 
yi^ = softmax(zi) = e^(zi) / sigma(e^zj) -> Đặt là u / v
Lại có: d(u/v) = (u'v - uv') /  v^2 
+) Với i = j thì v' = e^zi ; u' = e^zi
-> d(u/v) =  (e^zi * sigma - e^zi * e^zi) / sigma^2 =  (e^zi / sigma) - (e^zi / sigma) * (e^zi / sigma)
-> d(u/v) = yi^ - yi^ * yi^ = yi^(1-yi^)
+) Với i != j thì u' = 0
-> d(u/v) = -(e^zi * e^zj) / sigma^2 = -yi^ * yj^

=> (dL/dz2) = (-yi / yi^) * (yi^(1-yi^))  + (yj / yj^)*sigma{i!=j}(yi^ * yj)
= -yi(1-yi^) + yi^* sigma{i!=j}(yj) = -yi + yi*yi^ + yi^*sigma{i!=j}(yj) = -yi + yi^(yi + sigma{i!=j}(yj))  
= -yi + yi^*sigma(yj)
Mà yj là vector one hot -> sigma(yj) = 1 -> (dL/dz2) = yi^ - yi (kí hiệu dz2)
-> dz2 = out - y_true

2. Tìm (dL / dW2)
(dL / dW2) = (dL / dz2) * (dz2 / dW2). Với z2 = a1 * W2 + b2
Ta có W2ij nối từ nơ ron i -> j, chỉ ảnh hưởng đến z2j
Tức là z2j = (a11 * W21j) + (a12 *W22j) + ... + (a1i * W2ij) + ... + b2j
Nên (dz2j / dW2ij) = a1i => (dL / dW2) = a1i * dz2 (kí hiệu dW2)
-> dW2 = np.dot(a1.T, dz2)

3. Tìm (dL / da1i)
(dL / da1i) = sigma(dL / dz2j) * (dz2j / da1i) (Vì a đóng góp tất cả vào các neuron output z2j)
Mà z2j =  sigma{i} (a1i * W2ij + b2j) -> d(z2j) / d(a1i) = W2ij
=> (dL / da1i) = sigma{j}(dz2j * W2ij)
-> da1 = np.dot(dz2,W2.T)

4. Tìm (dL / dz1) 
Mà có a1 = relu(z1)
Nên (dL / dz1) = (dL / da1) * (da1 / dz1)
=> (dL / dz1) = da1 * relu_derivative(z1)
-> dz1 = da1 * relu_derivative(z1)  
"""
import numpy as np

# Hàm softmax và relu
def softmax(z):
    zi = z - np.max(z) # Trừ đi phần max của mảng z, tránh bị tràn số
    return np.exp(zi) / (np.sum(np.exp(zi)))

def relu(x):
    return np.maximum(0,x)

def relu_derivative(z):
    return (z>0).astype(float) # Trả về 1.0 nếu z>0


# Feed Forward (Máy chỉ đoán chứ chưa học)
X = np.array([0.5, -0.2]).reshape(1,-1) # Dữ liệu đầu vào
y_true = np.array([1.0, 0.0, 0.0]).reshape(1,-1) # Dữ liệu output nên có
lr = 0.1
epochs = 100

# Hidden layer
W1 = np.random.randn(2,3) * 0.1 # Do lớp đầu vào (input) có shape là (2,)
b1 = np.zeros((3,))

# Output layer
W2 = np.random.randn(3,3) * 0.1 # Do lớp ẩn có output là 3 nên đầu vào cho output phải có shape ((3,x))  
b2 = np.zeros((3,))

# Thực hiện Feed Forward 
z1 = np.dot(X,W1) + b1
a1 = relu(z1)

z2 = np.dot(a1,W2) + b2 
out = softmax(z2)

print(f"Kết quả output lúc chưa học được gì: {out}") # Kết quả máy đoán

# Backpropagation
# Tính lỗi ở lớp Output
dz2 = out - y_true  # (1,3)

# Tính đạo hàm của a và dz2 để tìm ra lỗi 
dW2 = np.dot(a1.T,dz2)
db2 = np.sum(dz2,axis = 0, keepdims=True)

# Tính lỗi truyền về lớp Hidden
da1 = np.dot(dz2, W2.T)
dz1 = da1 * relu_derivative(z1)

# Tương tự ở phần cuối
dW1 = np.dot(X.T,dz1)
db1 = np.sum(dz1, axis = 0, keepdims=True)

# Update trọng số 
W2 = W2 - lr * dW2
b2 = b2 - lr * db2
W1 = W1 - lr * dW1
b1 = b1 - lr * db1
print("Đã cập nhật trọng số thành công!")

# Thử dự đoán lại (Feed Forward)
z1_new = np.dot(X,W1) + b1
a1_new = relu(z1_new)

z2_new = np.dot(a1_new,W2) + b2 
out_new = softmax(z2_new)
print(f"Kết quả output lúc sau khi học: {out_new}") # Kết quả máy đoán

# Có thể thêm epochs để train cho thấy rõ kết quả:
for i in range(epochs):
    z1 = np.dot(X, W1) + b1
    a1 = relu(z1)
    z2 = np.dot(a1, W2) + b2 
    out = softmax(z2)
    
    dz2 = out - y_true  
    dW2 = np.dot(a1.T,dz2)
    db2 = np.sum(dz2,axis = 0, keepdims=True)
    da1 = np.dot(dz2, W2.T)
    dz1 = da1 * relu_derivative(z1)
    dW1 = np.dot(X.T,dz1)
    db1 = np.sum(dz1, axis = 0, keepdims=True)

    # Update trọng số 
    W2 = W2 - lr * dW2
    b2 = b2 - lr * db2
    W1 = W1 - lr * dW1
    b1 = b1 - lr * db1
    
    if (i+1) % 20 == 0:
        loss = -np.sum(y_true * np.log(out)) # Công thức tính Loss
        print(f"Epoch {i+1} | Loss: {loss:.4f} | Dự đoán: {np.round(out, 3)}")