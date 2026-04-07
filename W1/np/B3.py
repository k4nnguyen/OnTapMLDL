"""
Cho một tập hợp các điểm tọa độ trong không gian 2D được lưu trong một mảng NumPy. 
Hãy tính ma trận khoảng cách Euclidean giữa tất cả các cặp điểm với nhau bằng kỹ thuật broadcasting. 
Không sử dụng các hàm có sẵn để tính khoảng cách của scipy hay sklearn.
"""
import numpy as np
points = np.array([[0, 0], [1, 1], [3, 4], [6, 8]]) # shape = (4,2)
X = points[:,np.newaxis,:] 
print(X) # shape = (4, 1, 2)
print("\n")
Y = points[np.newaxis, :, :]
print(Y) # shape = (1,4,2)
print("\n\n")
tmp = X-Y
print(tmp) # shape = (4,4,2) (Vì (4,1,2) - (1,4,2) -> Kéo dãn thành (4,4,2))
distances = np.sqrt(np.sum(tmp**2,axis=-1)) # axis = -1 kiếm trục sâu nhất (Ở đây = 2)

print(f"Ma trận khoảng cách:\n{distances}")
"""
X sau khi giãn ra:
[
    [
        [0 0]
        [0 0]
        [0 0]
        [0 0]
    ]

    [
        [1 1]
        [1 1]
        [1 1]
        [1 1]
    ]

    [
        [3 4]
        [3 4]
        [3 4]
        [3 4]
    ]

    [
        [6 8]
        [6 8]
        [6 8]
        [6 8]
    ]
]

Y sau khi giãn ra:
[
    [
        [0 0]
        [1 1]
        [3 4]
        [6 8]
    ]
    [
        [0 0]
        [1 1]
        [3 4]
        [6 8]
    ]
    [
        [0 0]
        [1 1]
        [3 4]
        [6 8]
    ]
    [
        [0 0]
        [1 1]
        [3 4]
        [6 8]
    ]
]
    """
