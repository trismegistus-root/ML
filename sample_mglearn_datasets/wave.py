"""
To illustrate regression algorithms, 
this is the synthetic 'wave' dataset.
The wave dataset has a single input feature 
and a continuous target variable (or response)
that we want to model. 
x-axis: single feature
y-axis: regression target (output)
"""
import mglearn
from matplotlib import pyplot as plt

X,y = mglearn.datasets.make_wave(n_samples=40)
plt.plot(X,y,'o')
plt.ylim(-3,3)
plt.xlabel("Feature")
plt.ylabel("Target")
plt.show()

