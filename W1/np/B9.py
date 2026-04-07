"""
Cho một mảng 1D chứa điểm số (từ 0 đến 10). Hãy tạo ra một mảng mới chứa nhãn phân loại:
Điểm < 5: Nhãn 0 (Kém)
Điểm từ 5 đến < 8: Nhãn 1 (Khá)
Điểm >= 8: Nhãn 2 (Giỏi)
"""
import numpy as np
scores = np.array([3.5, 8.0, 5.5, 9.2, 1.0, 6.5])
conditions = [
    (scores < 5),
    (scores >= 5) & (scores < 8),
    (scores >= 8)
]

choices = ["Kém", "Khá", "Giỏi"]
#print(conditions)
print(np.select(conditions,choices,default="Unknown"))