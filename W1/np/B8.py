"""
Sắp xếp mảng 2D theo một cột chỉ định (Sorting by Column)
Yêu cầu: Cho một ma trận 2D chứa dữ liệu của các mẫu vật (mỗi hàng là một mẫu, mỗi cột là một đặc trưng). 
Hãy sắp xếp toàn bộ ma trận này theo thứ tự tăng dần dựa trên giá trị của cột thứ 2 (index 1), 
trong khi vẫn giữ nguyên sự gắn kết của các hàng. 
"""
import numpy as np
data = np.array([[9, 2, 3],
                 [4, 5, 6],
                 [7, 0, 5]])
indices_data = data[:,1] # Cắt phần cột giữa để sort
sorted_indices = np.argsort(indices_data)
result = data[sorted_indices] # Sắp xếp lại theo sorted_indices
print(f"Ma trận sau khi được sắp xếp:\n {result}")