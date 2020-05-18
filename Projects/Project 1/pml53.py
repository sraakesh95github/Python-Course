# function used by various programs to plot decision regions
# author: Allee updated by sdm

import numpy as np                            # needed for math stuff
from matplotlib.colors import ListedColormap  # for choosing colors
import matplotlib.pyplot as plt               # to create the plot

################################################################################
# Function to plot decision regions.                                           #
# Inputs:                                                                      #
#    X - feature values of each sample, e.g. coordinates on cartesian plane    #
#    y - the classification of each sample - a one-dimensional array           #
#    classifier - the machine learning classifier to use, e.g. perceptron      #
#    test_idx - typically the range of samples that were the test set          #
#               the default value is none; if present, highlight them          #
#    resolution - the resolution of the meshgrid                               #
# Output:                                                                      #
#    None                                                                      #
#                                                                              #
# NOTE: this will support up to 5 classes described by 2 features.             #
################################################################################

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    # we will support up to 5 classes...
    markers = ('v', 'x', 'o', '^', 's')                      # markers to use
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')   # colors to use
    cmap = ListedColormap(colors[:len(np.unique(y))])        # the color map https://docs.scipy.org/doc/numpy/reference/generated/numpy.unique.html
    
    # plot the decision surface
    # x1* will be the range +/- 1 of the first feature
    # x2* will be the range +/- 1 of the first feature
    x1_min, x1_max = X[:,0].min() - 1, X[:,0].max() + 1      # all rows, col 0
    x2_min, x2_max = X[:,1].min() - 1, X[:,1].max() + 1      # all rows, col 1

    # now create the meshgrid (see p14.py for examples)
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), #https://docs.scipy.org/doc/numpy/reference/generated/numpy.meshgrid.html?highlight=meshgrid#numpy.meshgrid
                           np.arange(x2_min, x2_max, resolution))
    #print(xx1)
    #print(xx2)

    # ravel flattens the array. The default, used here, is to flatten by taking
    # all of the first row, concanentating the second row, etc., for all rows
    # So we will predict the classification for every point in the grid

    Z = classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
    #https://docs.scipy.org/doc/numpy/reference/generated/numpy.ravel.html
    #https://scikit-learn.org/stable/tutorial/statistical_inference/supervised_learning.html
    #file:///C:/SR%20files/Subjects/Python/Class%20notes/Machine%20Learning/test_basics.py

    # reshape will take the resulting predictions and put them into a matrix
    # with the same shape as the mesh
    Z = Z.reshape(xx1.shape) #https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.shape.html

    # using Z, create a contour plot so we can see the regions for each class
    plt.contourf(xx1,xx2,Z,alpha=0.4,cmap=cmap)
    #https://matplotlib.org/3.1.3/api/_as_gen/matplotlib.pyplot.contourf.html#matplotlib.pyplot.contourf
    #Alpha is used to adjust the transparency
    #

    plt.xlim(xx1.min(), xx1.max())                # set x-axis ranges
    plt.ylim(xx2.min(), xx2.max())                # set y-axis ranges

    # plot all samples
    # NOTE: X[y==c1,0] returns all the column 0 values of X where the
    #       corresponding row of y equals c1. That is, only those rows of
    #       X are included that have been assigned to class c1.
    # So, for each of the unique classifications, plot them!
    # (In this case, idx and c1 are always the same, however this code
    #  will allow for non-integer classifications.)

    for idx, c1 in enumerate(np.unique(y)):
        plt.scatter(x=X[y==c1,0], y=X[y==c1,1], alpha=0.8, c=colors[idx],
                    marker=markers[idx], label=c1)
        
    #highlight test samples with black circles
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]   # test set is at the end
        plt.scatter(X_test[:,0],X_test[:,1],c='',edgecolor='black',alpha=1.0,
                    linewidth=1, marker='o',s=55,label='test set')

plt.show()
