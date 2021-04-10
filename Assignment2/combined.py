import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
digits = load_digits()

X_train, X_test, y_train, y_test = train_test_split(digits.data,digits.target,stratify=digits.target, random_state =2021)
scaler = StandardScaler()
scaler.fit(X_train)
X_train_sc = scaler.transform(X_train)
X_test_sc = scaler.transform(X_test)

values = dict()

k = 3
KNclf = KNeighborsClassifier(n_neighbors=k)
KNclf.fit(X_train_sc, y_train)
values[accuracy_score(y_test, KNclf.predict(X_test_sc))] = 'KN'

k = 3
DTclf = DecisionTreeClassifier(min_samples_leaf = k)
DTclf.fit(X_train_sc, y_train)
values[accuracy_score(y_test, DTclf.predict(X_test_sc))] = 'DT'

k = 21
RFclf = RandomForestClassifier(n_estimators=k)
RFclf.fit(X_train_sc, y_train)
values[accuracy_score(y_test, RFclf.predict(X_test_sc))] = 'RF'

C = 100
gamma = 0.01
SVCclf = SVC(C=C, kernel = 'rbf', gamma=gamma)
SVCclf.fit(X_train_sc, y_train)
values[accuracy_score(y_test, SVCclf.predict(X_test_sc))] = 'SVC'

k = 143
MLPclf = MLPClassifier(max_iter=k, random_state=2021)
MLPclf.fit(X_train_sc, y_train)
values[accuracy_score(y_test, MLPclf.predict(X_test_sc))] = 'MLP'

plt.bar(list(values.values()), list(values.keys()), width=0.6)
plt.title('All together')
plt.text(values[max(values.keys())],max(values.keys()),
         'max : ' + str(format(max(values.keys()), ".2f")),
         verticalalignment='bottom' , horizontalalignment='center')
plt.show()