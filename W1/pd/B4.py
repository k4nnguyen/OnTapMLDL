"""
Giả sử bạn có dữ liệu bán kem trong 10 ngày. Bạn nghi ngờ rằng nhiệt độ càng cao thì bán càng được nhiều kem,
và ngày cuối tuần thì khách mua đông hơn. Hãy vẽ đồ thị để kiểm chứng!
"""
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 

data = {
    'Ngay': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Nhiet_Do': [28, 30, 32, 35, 36, 25, 22, 31, 34, 38], # Nhiệt độ (độ C)
    'Doanh_Thu': [15, 18, 22, 28, 30, 10, 8, 20, 25, 35], # Doanh thu (Triệu VNĐ)
    'Cuoi_Tuan': ['Khong', 'Khong', 'Khong', 'Khong', 'Khong', 'Co', 'Co', 'Khong', 'Khong', 'Khong'] 
}
df = pd.DataFrame(data)

# Đồ thị của plt
plt.figure(figsize=(8,4))
plt.xlabel("Ngay")
plt.ylabel("Doanh thu")
plt.plot(df['Ngay'], df['Doanh_Thu'],marker='o') # Biểu đồ dạng đường nối các điểm 
plt.grid(True) # Bật lưới
plt.show()

# Đồ thị của sns
plt.figure(figsize=(8,5))
sns.scatterplot(x='Ngay',y='Doanh_Thu',data=data,hue=df['Cuoi_Tuan'])
plt.title("Tương quan Nhiệt độ và Doanh thu bán kem")
plt.show()

plt.figure(figsize=(6, 4))
sns.histplot(df['Nhiet_Do'],bins=5,kde=True)
plt.show()