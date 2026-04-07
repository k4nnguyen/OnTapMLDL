"""
 Cho một mảng 1D đại diện cho một chuỗi dữ liệu (ví dụ: chứng khoán, nhiệt độ).
 Hãy tính trung bình động (moving average) với cửa sổ trượt (window size) là w
"""
import numpy as np
arr = np.array([1, 2, 3, 7, 9, 10, 21, 25])
w = 3

window = np.ones(w) / w
print(window)

res = np.convolve(arr,window,mode='valid')
print(res)