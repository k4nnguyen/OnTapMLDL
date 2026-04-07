"""
Xử lý thử dữ liệu với pandas 
"""
import pandas as pd 
import numpy as np

data = {
    'ID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Tuoi': [23, 25, np.nan, 21, 22, 22, 24, np.nan, 23, 25],
    'Thu_Nhap': [10, 12, 11, np.nan, 13, 15, 1000, 14, 11, 12], # Có ông thu nhập 1000 là Outlier
    'Gioi_Tinh': ['Nam', 'Nu', 'Nam', 'Nu', np.nan, 'Nam', 'Nu', 'Nam', 'Nu', 'Nam'],
    'Da_Mua_Hang': [1, 0, 1, 1, 0, 1, 0, np.nan, 1, 0] # Cột mục tiêu (Target)
}

df = pd.DataFrame(data)
print("Dữ liệu gốc bị bẩn:\n", df)


res = df.copy()
res.dropna(subset=['Da_Mua_Hang'],inplace=True) # Có thể bỏ
res['Tuoi'] = res['Tuoi'].fillna(round(res['Tuoi'].mean())) # Do tuổi không có outlier
res['Thu_Nhap'] = res['Thu_Nhap'].fillna(res['Thu_Nhap'].median()) # Thu nhập có 1000 là outlier
res['Gioi_Tinh'] = res['Gioi_Tinh'].fillna(res['Gioi_Tinh'].mode()[0]) # Giới tính lấy giới tính đông nhất là đc
print("Dữ liệu sau khi xử lý:\n",res)