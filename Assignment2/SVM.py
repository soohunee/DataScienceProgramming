from matplotlib import pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

digits = load_digits()

X_train, X_test, y_train, y_test = train_test_split(digits.data,digits.target,stratify=digits.target, random_state =2021)
scaler = StandardScaler()
scaler.fit(X_train)
X_train_sc = scaler.transform(X_train)
X_test_sc = scaler.transform(X_test)

SVC_train_hat = list()
SVC_test_hat= list()

C_settings = [0.01, 1, 100]
gamma_settings = [0.01, 0.1, 1]

for C in C_settings:
    for gamma in gamma_settings:
        SVCclf = SVC(C=C, kernel = 'rbf', gamma=gamma)
        SVCclf.fit(X_train_sc, y_train)
        SVC_train_hat.append(accuracy_score(y_train, SVCclf.predict(X_train_sc)))
        SVC_test_hat.append(accuracy_score(y_test, SVCclf.predict(X_test_sc)))
#C = 100, gamma = 0.01

print(SVC_test_hat)