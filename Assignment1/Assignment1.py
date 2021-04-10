from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

X, y = load_wine(return_X_y=True)
print(X.shape, y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=2021)
for k in range(1, 11):
    clf = KNeighborsClassifier(p=1, n_neighbors=k, weights='uniform')
    clf.fit(X_train, y_train)

    y_train_hat = clf.predict(X_train)
    y_test_hat = clf.predict(X_test)
    print('M/u k={} train : '.format(k), accuracy_score(y_train, y_train_hat), '/ test : ', accuracy_score(y_test, y_test_hat))
for k in range(1, 11):
    clf = KNeighborsClassifier(p=2, n_neighbors=k, weights='uniform')
    clf.fit(X_train, y_train)

    y_train_hat = clf.predict(X_train)
    y_test_hat = clf.predict(X_test)
    print('E/u k={} train : '.format(k), accuracy_score(y_train, y_train_hat), '/ test : ', accuracy_score(y_test, y_test_hat))
for k in range(1, 11):
    clf = KNeighborsClassifier(p=1, n_neighbors=k, weights='distance')
    clf.fit(X_train, y_train)

    y_train_hat = clf.predict(X_train)
    y_test_hat = clf.predict(X_test)
    print('M/d k={} train : '.format(k), accuracy_score(y_train, y_train_hat), '/ test : ', accuracy_score(y_test, y_test_hat))
for k in range(1, 11):
    clf = KNeighborsClassifier(p=2, n_neighbors=k, weights='distance')
    clf.fit(X_train, y_train)

    y_train_hat = clf.predict(X_train)
    y_test_hat = clf.predict(X_test)
    print('E/d k={} train : '.format(k), accuracy_score(y_train, y_train_hat), '/ test : ', accuracy_score(y_test, y_test_hat))