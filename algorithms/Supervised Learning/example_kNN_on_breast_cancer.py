"""
Using breast cancer data to illustrate overfit and underfit in the 
kNeighbors model. 

The plots of Accuracy vs. n_neighbors shows that for any n_neighbors such that
1 <= n_neighbors <= len(dataset), there exists a "sweet spot" where training
accuracy and test accuracy produce the greatest overall accuracy. 

NOTES:
As n_neighbors approaches len(dataset), kNN produces nearly the same result
every time it is run. (underfit)

As n_neighbors approaches 1, kNN becomes too complex of a model. (overfit)
"""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt


cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state=66)

training_accuracy = []
test_accuracy = []
neighbors_settings = range(1,11)

for n_neighbors in neighbors_settings:
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_train, y_train)
    training_accuracy.append(clf.score(X_train, y_train))
    test_accuracy.append(clf.score(X_test, y_test))

plt.plot(neighbors_settings, training_accuracy, label = 'training accuracy')
plt.plot(neighbors_settings, test_accuracy, label = 'test accuracy')
plt.ylabel('Accuracy')
plt.xlabel('n_neighbors')
plt.legend()
plt.show()