# Perceptron example of Iris data set
# author: Allee, Hartin updated by sdm

from pml53 import plot_decision_regions  # plotting function
import matplotlib.pyplot as plt  # so we can add to plot
from sklearn import datasets  # read the data sets
import numpy as np  # needed for arrays
from sklearn.model_selection import train_test_split  # splits database
from sklearn.preprocessing import StandardScaler  # standardize data
from sklearn.linear_model import Perceptron  # the algorithm
from sklearn.metrics import accuracy_score  # grade the results
import pandas

# iris = datasets.load_iris()                  # load the data set
# X = iris.data[:,[2,3]]                       # separate the features we want
# y = iris.target                              # extract the classifications #Iris is a dictionary containing features and labels. Here target is the key
# print(iris)
# print(y)

df = pandas.read_csv('data_banknote_authentication.txt', names=['variance', 'skewness', 'curtosis', 'entropy', 'class'])
df_numpy = df.to_numpy()
X = df_numpy[:, [1, 2]]
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

ppn = Perceptron(max_iter=4, tol=1e-3, eta0=0.001, fit_intercept=True, random_state=0, verbose=True)
ppn.fit(X_train_std, y_train)  # do the training

print('Number in test ', len(y_test))
y_pred = ppn.predict(X_test_std)  # now try with the test data

# Note that this only counts the samples where the predicted value was wrong
print('Misclassified samples: %d' % (y_test != y_pred).sum())  # how'd we do?
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

# vstack puts first array above the second in a vertical stack
# hstack puts first array to left of the second in a horizontal stack
# NOTE the double parens!
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
print('Number in combined ', len(y_combined))

# we did the stack so we can see how the combination of test and train data did
y_combined_pred = ppn.predict(X_combined_std)
print('Misclassified combined samples: %d' % (y_combined != y_combined_pred).sum())
print('Combined Accuracy: %.2f' % accuracy_score(y_combined, y_combined_pred))

# now visualize the results
plot_decision_regions(X=X_combined_std, y=y_combined, classifier=ppn,
                      test_idx=range(len(X_train), len(X)))
plt.xlabel('Skewness [standardized]')
plt.ylabel('Curtosis [standardized]')
plt.legend(loc='upper left')
plt.show()
