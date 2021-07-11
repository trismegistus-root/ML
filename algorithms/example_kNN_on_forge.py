"""
The  k-NN  algorithm  is  arguably  the  simplest  machine  learning  
algorithm.  Buildingthe  model  consists  only  of  storing  the  
training  dataset.  To  make  a  prediction  for  anew data point, 
the algorithm finds the closest data points in the training dataset—its
“nearest neighbors.”
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
plt.title("Original Plot")
plt.show()
plt.title("n_neighbors = 1")
mglearn.plots.plot_knn_classification(n_neighbors=1)
plt.show()
plt.title("n_neighbors = 3")
mglearn.plots.plot_knn_classification(n_neighbors=3)
plt.show()