from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt
import mglearn

X, y = mglearn.datasets.make_forge() #synthetic two class classification dataset

#train_test_split to apportion 25% of data for error 
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

#for neighbors during predictions
clf = KNeighborsClassifier(n_neighbors=3)

#fitting classifier to test data
clf.fit(X_train, y_train) 

print("Test set prediction: {}".format(clf.predict(X_test)))
print("Test set accuracy: {:.2f}".format(clf.score(X_test, y_test)))


# vizualizing decision boundaries
fig, axes = plt.subplots(1, 3, figsize=(10,3))

for n_neighbors, ax in zip([1,3,9], axes):
    clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X,y)
    mglearn.plots.plot_2d_separator(clf, X, fill=True, eps=0.5, ax=ax, alpha=.4)
    mglearn.discrete_scatter(X[:,0], X[:, 1], y, ax=ax)
    ax.set_title("{} neighbor(s)".format(n_neighbors))
    ax.set_xlabel("feature 0")
    ax.set_ylabel("feature 1")
    # add chart notes here
axes[0].legend(loc=3)

# decision boundaries created  by the nearest neighbors model for different values of n_neighbors
plt.show()
