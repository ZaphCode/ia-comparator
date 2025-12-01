from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

"""
Entrena y evalúa el clasificador K-Nearest Neighbors.
Regresa: modelo, accuracy, reporte de clasificación y matriz de confusión.
"""
def run_knn(X_train, X_test, y_train, y_test, n_neighbors=5):
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    return model, acc, report, cm

"""
Entrena y evalúa el clasificador Gaussian Naive Bayes.
"""
def run_naive_bayes(X_train, X_test, y_train, y_test):
    model = GaussianNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    return model, acc, report, cm


"""
Entrena y evalúa el clasificador SVM (Máquinas de Soporte Vectorial).
"""
def run_svm(X_train, X_test, y_train, y_test, C=1.0, kernel='rbf'):
    model = SVC(C=C, kernel=kernel)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    return model, acc, report, cm