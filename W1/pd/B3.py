"""
Cho bộ dữ liệu về Số_Giờ_Học và Điểm_Thi của 20 học sinh. Trong đó có 1 học sinh là thiên tài (Học 1 giờ nhưng được 10 điểm) 
và 1 học sinh ngủ gật (Học 10 giờ nhưng được 1 điểm) - đây là các Outliers.
"""
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 

# Dữ liệu mô phỏng
data = {
    'So_Gio_Hoc': [2, 3, 4, 3, 5, 6, 7, 6, 8, 9, 2, 4, 5, 7, 8, 9, 3, 8, 1, 20],
    'Diem_Thi':   [3, 4, 4, 5, 6, 6, 8, 7, 8, 9, 2, 5, 5, 7, 9, 10, 3, 7, 10, 1] 
    # Lưu ý: Cặp (1, 10) và (10, 1) là Outliers
}
df = pd.DataFrame(data)
#print(df.describe())

plt.figure(figsize=(8,5))
plt.scatter('So_Gio_Hoc','Diem_Thi',data=data) # Scatter những điểm ở đồ thị
plt.title('Tương quan giữa số giờ học và điểm thi') # Title ở trên đầu
plt.xlabel('Số giờ học') # Label cho trục x và y
plt.ylabel('Điểm thi')
plt.show()

plt.figure(figsize = (6,4))
sns.boxplot(data=data)
plt.title("Boxplot của Điểm thi")
plt.show()