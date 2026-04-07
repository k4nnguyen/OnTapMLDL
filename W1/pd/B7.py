"""
Bạn có dữ liệu hóa học của 178 chai rượu vang thuộc 3 loại khác nhau (được đánh nhãn 0, 1, 2). 
Mỗi chai rượu được đo lường bằng 13 chỉ số hóa học (Ví dụ: Nồng độ cồn, Axit Malic, Tro, Magie, v.v.).
Lưu ý quan trọng: Trong 13 chỉ số này, có những chất chỉ chiếm 0.1 gram, nhưng có chất (như Proline) lại lên tới 1000 gram
"""
import pandas as pd 
from sklearn.datasets import load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# Load data và xem thử
dataset = load_wine()
#print(dataset.keys())
X = pd.DataFrame(data=dataset.data, columns=dataset.feature_names)
y = pd.Series(data=dataset.target) # Đổi thành series để kh bị scikit learn warning
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
print(X.describe()) # Từ việc xem phần std của proline và alcohol thấy lệch khá nặng


# Chuẩn hóa input
scaler = StandardScaler()
X_train_scaler = scaler.fit_transform(X_train)
x_test_scaler = scaler.transform(X_test)

# Sử dụng Logistic Regression để dự đoán output
model = LogisticRegression()
model.fit(X_train_scaler,y_train)
y_pred = model.predict(x_test_scaler)

# Tính toán metrics
acc = accuracy_score(y_test,y_pred)
cm = confusion_matrix(y_test,y_pred)

print(f"Accuracy của mô hình: {acc}")
print(f"Confusion matrix:\n {cm}") 