"""
Dùng MLPClassifier của Sklearn để huấn luyện một mạng nơ-ron nhận diện các chữ số viết tay từ 0 đến 9.
Bộ dữ liệu này gồm các ảnh đen trắng kích thước 8 x 8 pixel.
"""
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

digits = load_digits()
X = digits.data
y = digits.target 

# plt.gray()
# plt.matshow(digits.images[0])
# plt.show()

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)

mlp = MLPClassifier(hidden_layer_sizes=(64,32), activation='relu', solver='adam', max_iter=5, random_state=42)
mlp.fit(X_train_scaled,y_train)

y_pred = mlp.predict(X_test_scaled)
cm = confusion_matrix(y_test,y_pred)

print(classification_report(y_test,y_pred))
print(cm)