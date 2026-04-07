import numpy as np

arr = np.array([
    
    [
        [1,2,3,4],
        [1,2,3,4]
    ],
    [
        [1,2,3,4],
        [1,2,3,4]
    ],
])

print(f"Số chiều của ma trận: {arr.ndim}") # n Dimension
print(f"Shape của ma trận: {arr.shape}")
print(f"Số mảng con trong ma trận ngoài cùng: {len(arr)}") # Length
print(f"Số phần tử trong ma trận: {arr.size}")
print(f"Kiểu của ma trận: {type(arr)}") 
print(f"Kiểu dữ liệu của ma trận: {arr.dtype}") # Data type
print(np.zeros(shape=(2,3,1)))

f = np.array([0,1,2,3,4,5,6])
print(f[[1,4,2,3]])