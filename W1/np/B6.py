"""
Cho một mảng 1D. Hãy tìm tất cả các "đỉnh cục bộ" (local maxima). 
Một phần tử được gọi là đỉnh nếu nó lớn hơn cả phần tử liền trước và liền sau nó. 
Gợi ý: Hãy dùng kỹ thuật cắt mảng (slicing) để tạo ra 3 mảng lệch nhau (trước, giữa, sau) và so sánh chúng.
"""
import numpy as np
arr = np.array([1, 3, 7, 1, 2, 6, 0, 1])
left = arr[:-2]
center = arr[1:-1]
right = arr[2:]
res = (center > left) & (center > right)
print(f"Các điểm cực đại cục bộ là: {center[res]}")