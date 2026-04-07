"""
Sử dụng sklearn để dự đoán bệnh ung thư với nhiều đặc trưng, chuẩn hóa dữ liệu
"""
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd 

# Load data
cancer_data = load_breast_cancer()

# Kiểm tra data
# print(f"Các key của bộ data Ung thư:\n {cancer_data.keys()}") # In ra các phần Key để hiểu dữ liệu
X = pd.DataFrame(cancer_data.data,columns=cancer_data.feature_names) # (569,30)
y = pd.DataFrame(cancer_data.target) # (569,1)

# Chia tập dữ liệu để huấn luyện
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=42)

# Vì các đặc trưng khác nhau trải dài khác nhau, nên cần chuẩn hóa về dạng (0,1), dựa trên tập Train
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Khai báo model và huấn luyện
model = LogisticRegression()
model.fit(X_train_scaled,y_train)

y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred) 
cm = confusion_matrix(y_test, y_pred)

# Dự đoán và Confusion Matrix
print(f"Độ chính xác (Accuracy): {acc}\n")
print(f"Ma trận nhầm lẫn:\n{cm}")