import matplotlib.pyplot as plt  # so we can add to plot
import numpy as np  # needed for arrays
from sklearn.model_selection import train_test_split  # splits database
from sklearn.preprocessing import StandardScaler  # standardize data
from sklearn.linear_model import *  # the algorithm
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score  # grade the results
import pandas
from matplotlib.colors import ListedColormap
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import *
from sklearn.neighbors import KNeighborsClassifier

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    # we will support up to 5 classes...
    markers = ('v', 'x', 'o', '^', 's')  # markers to use
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')  # colors to use
    cmap = ListedColormap(colors[:len(np.unique(y))])  # the color map https://docs.scipy.org/doc/numpy/reference/generated/numpy.unique.html

    # plot the decision surface
    # x1* will be the range +/- 1 of the first feature
    # x2* will be the range +/- 1 of the first feature
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1  # all rows, col 0
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1  # all rows, col 1

    # now create the meshgrid (see p14.py for examples)
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           # https://docs.scipy.org/doc/numpy/reference/generated/numpy.meshgrid.html?highlight=meshgrid#numpy.meshgrid
                           np.arange(x2_min, x2_max, resolution))
    # print(xx1)
    # print(xx2)

    # ravel flattens the array. The default, used here, is to flatten by taking
    # all of the first row, concanentating the second row, etc., for all rows
    # So we will predict the classification for every point in the grid

    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.ravel.html
    # https://scikit-learn.org/stable/tutorial/statistical_inference/supervised_learning.html
    # file:///C:/SR%20files/Subjects/Python/Class%20notes/Machine%20Learning/test_basics.py

    # reshape will take the resulting predictions and put them into a matrix
    # with the same shape as the mesh
    Z = Z.reshape(xx1.shape)  # https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.shape.html

    # using Z, create a contour plot so we can see the regions for each class
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    # https://matplotlib.org/3.1.3/api/_as_gen/matplotlib.pyplot.contourf.html#matplotlib.pyplot.contourf
    # Alpha is used to adjust the transparency
    #

    plt.xlim(xx1.min(), xx1.max())  # set x-axis ranges
    plt.ylim(xx2.min(), xx2.max())  # set y-axis ranges

    # plot all samples
    # NOTE: X[y==c1,0] returns all the column 0 values of X where the
    # corresponding row of y equals c1. That is, only those rows of
    # X are included that have been assigned to class c1.
    # So, for each of the unique classifications, plot them!
    # (In this case, idx and c1 are always the same, however this code
    #  will allow for non-integer classifications.)

    for idx, c1 in enumerate(np.unique(y)):
        plt.scatter(x=X[y == c1, 0], y=X[y == c1, 1], alpha=0.8, c=colors[idx],
                    marker=markers[idx], label=c1)

    # highlight test samples with black circles
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]  # test set is at the end
        plt.scatter(X_test[:, 0], X_test[:, 1], c='', edgecolor='black', alpha=1.0,
                    linewidth=1, marker='o', s=55, label='test set')

    plt.xlabel('Skewness [standardized]')
    plt.ylabel('Curtosis [standardized]')
    plt.legend(loc='upper left')
    plt.show()

#Read the dataset for counterfeit notes
df = pandas.read_csv('data_banknote_authentication.txt', names=['variance', 'skewness', 'curtosis', 'entropy', 'class'])
df_numpy = df.to_numpy()
X = df_numpy[:, [0, 1, 2, 3]]
y = df_numpy[:, [4]].ravel()

# split the problem into train and test
# this will yield 70% training and 30% test
# random_state allows the split to be reproduced
# stratify=y not used in this case
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# scale X by removing the mean and setting the variance to 1 on all features.
# the formula is z=(x-u)/s where u is the mean and s is the standard deviation.
# (mean and standard deviation may be overridden with options...)

sc = StandardScaler()  # create the standard scalar
sc.fit(X_train)  # compute the required transformation
X_train_std = sc.transform(X_train)  # apply to the training data
X_test_std = sc.transform(X_test)  # and SAME transformation of test data!!!

# perceptron linear
# epoch is one forward and backward pass of all training samples (also an iteration)
# eta0 is rate of convergence
# max_iter, tol, if it is too low it is never achieved
# and continues to iterate to max_iter when above tol
# fit_intercept, fit the intercept or assume it is 0
# slowing it down is very effective, eta is the learning rate

def train_data(classifier_test, classifier_name):
    classifier_test.fit(X_train_std, y_train)  # do the training
    print("\n" + classifier_name)
    #print('Number in test ', len(y_test))
    y_pred = classifier_test.predict(X_test_std)  # now try with the test data

    # Note that this only counts the samples where the predicted value was wrong
    #print('Misclassified samples: %d' % (y_test != y_pred).sum())  # how'd we do?
    print('Test Data Accuracy: %.2f' % (accuracy_score(y_test, y_pred) * 100))

    # vstack puts first array above the second in a vertical stack
    # hstack puts first array to left of the second in a horizontal stack
    # NOTE the double parens!
    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))
    #print('Number in combined ', len(y_combined))

    # we did the stack so we can see how the combination of test and train data did
    y_combined_pred = classifier_test.predict(X_combined_std)
    #print('Misclassified combined samples: %d' % (y_combined != y_combined_pred).sum())
    print('Combined Accuracy: %.2f' % ((accuracy_score(y_combined, y_combined_pred)) * 100))

# now visualize the results
#plot_decision_regions(X=X_combined_std, y=y_combined, classifier=classifier,
#                      test_idx=range(len(X_train), len(X)))

print("Accuracies in Percentage\n")

classifier = Perceptron(max_iter=20, tol=1e-3, eta0=0.001, fit_intercept=True, random_state=0, verbose=False)
train_data(classifier, "Perceptron")

classifier = LogisticRegression(C = 10, solver='liblinear', multi_class='ovr', random_state = 0)
train_data(classifier, "Logistic Regression")

classifier = SVC(kernel = 'linear', C=1.0, random_state = 0)
train_data(classifier, "Support Vector Classifier")

classifier = DecisionTreeClassifier(criterion = 'entropy', max_depth = 5, random_state = 0)
train_data(classifier, "Decision Tree Classifier")

classifier = RandomForestClassifier(criterion='entropy', n_estimators=10, random_state=1, n_jobs=2)
train_data(classifier, "Random Forest Classifier")

classifier = KNeighborsClassifier(n_neighbors=10,p=2,metric='minkowski')
train_data(classifier, "K Nearest Neighbour Classifier")

#export_graphviz(classifier,out_file='tree.dot',
#                feature_names=['Variance', 'Curtosis'])
