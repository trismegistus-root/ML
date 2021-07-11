"""
Example for a synthetic two-class classification dataset
"""

import mglearn
from matplotlib import pyplot as plt
#generating set
X,y = mglearn.datasets.make_forge()
#plotting
mglearn.discrete_scatter(X[:,0], X[:, 1], y)
plt.legend(['Class 0', 'Class 1'], loc=4)
plt.xlabel('First feature')
plt.ylabel('Second feature')
print('X.shape: {}'.format(X.shape))
plt.show()

