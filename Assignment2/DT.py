from matplotlib import pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

digits = load_digits()

X_train, X_test, y_train, y_test = train_test_split(digits.data,digits.target,stratify=digits.target, random_state =2021)
scaler = StandardScaler()
scaler.fit(X_train)
X_train_sc = scaler.transform(X_train)
X_test_sc = scaler.transform(X_test)

y_train_hat = list()
y_test_hat= list()
for k in range(1,50):
    DTclf = DecisionTreeClassifier(min_samples_leaf = k)
    DTclf.fit(X_train_sc, y_train)
    y_train_hat.append(accuracy_score(y_train, DTclf.predict(X_train_sc)))
    y_test_hat.append(accuracy_score(y_test, DTclf.predict(X_test_sc)))
    
plt.plot(range(1,50), y_train_hat)
plt.plot(range(1,50), y_test_hat)
plt.title('Decision Tree')
plt.xlabel('min_samples_leaf')
plt.ylabel('accuracy_score')
plt.legend(['train accuracy', 'test accuracy'])
plt.show()

#k =3
print(y_test_hat)