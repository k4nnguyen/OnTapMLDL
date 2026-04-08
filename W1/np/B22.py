"""
Hàm Softmax:
softmax(z) = e^zi / sigma(e^zj)
Sử dụng e^ cho hàm softmax vì cần phải biến các số thành dạng xác suất, có tổng = 1
Ngoài ra e^ cũng giúp xử lý các số âm. Softmax cũng được kết hợp với Cross-Entropy

Tuy vậy code phải để ý, vì nếu zi quá lớn -> tràn số (zi ~ 1000)
"""
import numpy as np
def softmax(z):
    zi = z - np.max(z) # Trừ đi phần max của mảng z, tránh bị tràn số
    return np.exp(zi) / (np.sum(np.exp(zi)))

z = np.array([-5,5,10, -25, -5,])

print(f"Phần softmax của từng phần tử trong z: {softmax(z)}")
print(f"Tổng của từng phần tử sau khi softmax: {np.sum(softmax(z))}")