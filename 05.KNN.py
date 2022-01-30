PROGRAM .NO.5(KNN)
Write a program to implement k-Nearest Neighbor algorithm to classify the iris data set. Print
both correct and wrong predictions. Java/Python ML library classes can be used for this
problem. """"


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn import datasets
iris = datasets.load_iris()
iris_data = iris.data
iris_labels = iris.target
x_train, x_test, y_train, y_test = train_test_split(iris_data,
iris_labels, test_size=0.20)
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
print('Confusion matrix is as follows')
print(confusion_matrix(y_test, y_pred))
print('Accuracy Metrics')
print(classification_report(y_test, y_pred))


########################################################################################################################
# OUTPUT:
# Ignore single quotes at beginning and end
########################################################################################################################


Confusion matrix is as follows
[[ 9  0  0]
 [ 0 11  1]
 [ 0  0  9]]
Accuracy Metrics
              precision    recall  f1-score   support

           0       1.00      1.00      1.00         9
           1       1.00      0.92      0.96        12
           2       0.90      1.00      0.95         9

    accuracy                           0.97        30
   macro avg       0.97      0.97      0.97        30
weighted avg       0.97      0.97      0.97        30
