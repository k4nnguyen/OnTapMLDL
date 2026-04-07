"""
Tự viết code tính Mean ($\mu$) và Variance sigma^2 bằng công thức,
không dùng hàm có sẵn của NumPy (dùng np.sum, phép tính cơ bản...).
"""
import numpy as np
data = np.array([10, 12, 23, 23, 16, 23, 21, 16])
sum = np.sum(data)
n = len(data)
mean = sum / n 
var = 0
var = np.sum((data-mean)**2)/n
std = np.sqrt(var)
print(f"Trung bình là: {mean}\nPhương sai là: {var}")

z_arr = (data - mean) / std 
print(f"Mảng sau khi chuẩn hóa Z-Score là: {z_arr}")