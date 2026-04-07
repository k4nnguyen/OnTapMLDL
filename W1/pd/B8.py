"""
Bạn có dữ liệu về tuổi, thu nhập và thành phố của khách hàng. Hãy dự đoán xem họ có "Mua hàng" (1) hay "Không" (0).
"""
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

data = {
    'Tuoi': [25, 35, 45, 20, 55, 52, 23, 40, 60, 48],
    'Thu_Nhap': [20, 50, 80, 15, 100, 90, 25, 70, 120, 85], # Triệu/tháng
    'Thanh_Pho': ['HN', 'HCM', 'DN', 'HN', 'HCM', 'DN', 'HN', 'HCM', 'DN', 'HCM'],
    'Mua_Hang': [0, 1, 1, 0, 1, 1, 0, 1, 1, 1]
}
df = pd.DataFrame(data)

le = LabelEncoder() # Label Encoder để biến data thành dạng từ 0 -> n
df['Thanh_Pho'] = le.fit_transform(df['Thanh_Pho'])

#print(df['Thanh_Pho'])

X = df.drop('Mua_Hang', axis = 1)
y = df['Mua_Hang']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

sc = StandardScaler() # Vẫn phải dùng standard scaler để chuẩn hóa về (0,1)
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train_scaled,y_train)

y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test,y_pred)


print(f"Độ chính xác của KNN là: {acc}")