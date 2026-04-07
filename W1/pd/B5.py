"""
Sử dụng thư viện sklearn có sẵn Logistic Regression và Train Test Split để dự đoán bệnh 
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
data = {
    'Duong_Huyet': [80, 120, 150, 90, 180, 110, 200, 85, 130, 160],
    'BMI': [22, 26, 30, 24, 35, 27, 40, 21, 29, 32],
    'Benh': [0, 0, 1, 0, 1, 0, 1, 0, 1, 1] # 0: Không bệnh, 1: Có bệnh
}
df = pd.DataFrame(data)

# Gán X là input và y là output để máy học
X = df[['Duong_Huyet','BMI']]
y = df['Benh']

# Chia train test split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=30)
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

print(f"Nhãn thực tế: {y_test}")
print(f"Máy dự đoán: {y_pred}")

acc = accuracy_score(y_test,y_pred)
print(f"Độ chính xác: {acc}")

cm = confusion_matrix(y_test, y_pred)
print(f"Ma trận nhầm lẫn:\n{cm}")