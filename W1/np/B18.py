"""
Code hàm tính Gini và Entropy 
Giả sử với 6 khách, muốn dự đoán xem ai sẽ mua hàng (y_goc)

"""
import numpy as np
def calculate_gini(x):
    if(len(x) == 0):
        return 0.0
    _, cnt = np.unique(x,return_counts=True) # Hàm này trả về 2 tham số: mảng đã sắp xếp unique và tần suất từng số
    
    probabilities = cnt / len(x)
    gini = 1.0 - np.sum(probabilities**2)   # 1 - sigma(pi^2)
    return gini

def calculate_entropy(x):
    if(len(x) == 0):
        return 0.0
    _, cnt = np.unique(x,return_counts=True) 
    probabilities = cnt / len(x)
    entropy = -np.sum(probabilities * np.log2(probabilities+1e-9))  # -sigma(pi * log2(pi)) (Cộng với 1e-9 tránh bị lỗi log(0))
    return entropy

nhom_tinh_khiet = np.array([1, 1, 1, 1, 1])
print(f"Gini nhóm tinh khiết: {calculate_gini(nhom_tinh_khiet)}") # Kết quả: 0.0

# Nhóm lộn xộn (50% nhãn 0, 50% nhãn 1)
nhom_lon_xon = np.array([0, 0, 1, 1])
print(f"Gini nhóm lộn xộn: {calculate_gini(nhom_lon_xon)}") # Kết quả: 0.5

def information_gain(node_goc, node_trai, node_phai):
    p_trai = len(node_trai) / len(node_goc)
    p_phai = len(node_phai) / len(node_goc)
    
    gain = calculate_gini(node_goc) - (p_trai * calculate_gini(node_trai) + p_phai * calculate_gini(node_phai))
    return gain     # gain càng cao càng tốt

y_goc = np.array([0, 0, 0, 1, 1, 1]) # Gini = 0.5 (Lộn xộn)

# Giả sử khi câu hỏi ở node_goc là true -> sẽ hỏi tiếp câu hỏi ở y_trai, và ngược lại sẽ hỏi câu hỏi y_phai
# Nhưng sau khi phân tách thì với y_trai vẫn chưa tách sao cho gini = 0, và y_phai cũng vậy nên phải tiến hành rẽ nhánh tiếp
y_trai_1 = np.array([0, 0, 1]) 
y_phai_1 = np.array([0, 1, 1])
print(f"Gain của 1: {information_gain(y_goc,y_trai_1,y_phai_1)}") # Gain 0.05

# Lần này câu hỏi giúp chia ra làm 2 nhánh và nó đều tinh khiêt (gini = 0)
y_trai_2 = np.array([1, 1, 1])
y_phai_2 = np.array([0, 0, 0])
print(f"Gain của 2: {information_gain(y_goc,y_trai_2,y_phai_2)}") # Lần này gain là 0.5 (Cao nhất)