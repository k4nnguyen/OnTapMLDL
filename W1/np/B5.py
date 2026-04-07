"""
Viết code chuyển mảng ảnh màu này thành ảnh xám 2D kích thước (H, W) bằng cách áp dụng công thức chuẩn: 
Grayscale = 0.2989 * R + 0.5870 * G + 0.1140 * B. 
Hãy thử dùng phép nhân ma trận (np.dot hoặc toán tử @) thay vì nhân thủ công từng kênh.
"""
import numpy as np
import matplotlib.pyplot as plt
image = np.random.randint(0, 256, size=(100, 100, 3))
convert_matrix = np.array([0.2989,0.5870,0.1140])

converted_image = image @ convert_matrix

print(f"Ma trận ảnh trước khi chuyển đổi: {image}")
plt.imshow(image)
plt.title("Trước khi chuyển xám")
plt.show()

print("\n\n")
print(f"Ma trận ảnh sau khi chuyển đổi: {converted_image}")
plt.imshow(converted_image,cmap='gray')
plt.title("Sau khi chuyển xám")
plt.show()