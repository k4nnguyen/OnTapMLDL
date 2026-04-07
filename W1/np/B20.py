"""
Code lại KNN
"""
import numpy as np
from collections import Counter

def euler_distance(A,B):
    return np.sqrt(np.sum((A-B)**2)) # Khoảng cách euclid

def knn(X,y,new_point,k = 3):
    list_distance = []      # List các khoảng cách và nhãn
    for i in range(len(X)):
        cur_point = X[i]
        cur_label = y[i]
        
        d = euler_distance(new_point,cur_point)     # Tính khoảng cách euler giữa điểm mới và điểm hiện tại
        list_distance.append((d,cur_label))         # Thêm vào list khoảng cách và label hiện tại
    list_distance.sort(key=lambda x: x[0])          # Sort tăng dần theo khoảng cách
    knn = list_distance[:k]
    neighbor_label = [label for distance,label in knn]
    result = Counter(neighbor_label).most_common(1)[0][0]   # Tìm ra nhãn xuất hiện nhiều nhất trong K hàng xóm
    
    return result

X_train = np.array([
    [0.5, 0.2],   # Thuộc Nhãn 0
    [0.6, 0.4],   # Thuộc Nhãn 0
    [-0.5, -0.8], # Thuộc Nhãn 1
    [-0.7, -0.5]  # Thuộc Nhãn 1
])
y_train = np.array([0, 0, 1, 1])

X_test = np.array([0.4,0.3])
y_pred = knn(X_train,y_train,X_test,k=3)

print(f"Điểm mới sẽ có nhãn: {y_pred}")        