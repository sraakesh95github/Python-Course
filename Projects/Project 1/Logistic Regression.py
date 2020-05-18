# Logistic Regression example of Iris data set
# author: d updated by sdm

from pml53 import plot_decision_regions  # plotting function
import matplotlib.pyplot as plt  # so we can add to plot
from sklearn import datasets  # read the data sets
import numpy as np  # needed for arrays
from sklearn.model_selection import train_test_split  # splits database
from sklearn.preprocessing import StandardScaler  # standardize data
from sklearn.linear_model import LogisticRegression  # the algorithm
import pandas

plt.cla()
iris = datasets.load_iris()  # load the data set
#X = iris.data[:, [1, 2]]  # separate the features we want
#y = iris.target  # extract the classifications

df = pandas.read_csv('data_banknote_authentication.txt', names = ['variance', 'skewness', 'curtosis', 'entropy', 'class'])
data = df.to_numpy()
#print(temp[:,[1,2]])
X = data[:,[1,2]]
y = data[:,[4]].ravel()

# split the problem into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

sc = StandardScaler()  # create the standard scalar
sc.fit(X_train)  # compute the required transformation
X_train_std = sc.transform(X_train)  # apply to the training data
X_test_std = sc.transform(X_test)  # and SAME transformation of test data!!!

# create logistic regression component.
# C is the inverse of the regularization strength. Smaller -> stronger!
#    C is used to penalize extreme parameter weights.
# solver is the particular algorithm to use
# multi_class determines how loss is computed - ovr -> binary problem for each label

lr = LogisticRegression(C=10, solver='liblinear', multi_class='ovr', random_state=0)
lr.fit(X_train_std, y_train)  # apply the algorithm to training data

# combine the train and test data
#rint(np.shape(X_train_std))
#rint(np.shape(X_test_std))
#print(np.shape(y_train))
#print(np.shape(y_test))
#print(y_test)
X_combined_std = np.vstack((X_train_std, X_test_std))
#print(np.shape(X_combined_std))
y_combined = np.hstack((y_train, y_test))
#print(np.shape(y_combined))

# plot the results
plot_decision_regions(X=X_combined_std, y=y_combined, classifier=lr, test_idx=range(105, 150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()



