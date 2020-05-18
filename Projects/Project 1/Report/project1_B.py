import matplotlib.pyplot as plt  # so we can add to plot
import numpy as np  # needed for arrays
from sklearn.model_selection import train_test_split  # splits database
from sklearn.preprocessing import StandardScaler  # standardize data
from sklearn.linear_model import *  # the algorithm
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score  # grade the results
import pandas
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import *
from sklearn.neighbors import KNeighborsClassifier

#Declare the accuracy arrays
test_data_accuracy_per = np.zeros(6)
combined_accuracy_per = np.zeros(6)
test_data_accuracy_num = np.zeros(6)
combined_accuracy_num = np.zeros(6)

#Read the dataset for counterfeit notes
df = pandas.read_csv('data_banknote_authentication.txt', names=['variance', 'skewness', 'curtosis', 'entropy', 'class'])
df_numpy = df.to_numpy()
X = df_numpy[:, [0, 1, 2, 3]]
y = df_numpy[:, [4]].ravel()
num_samples = df.values.shape[0]

# split the problem into train and test
# this will yield 70% training and 30% test
# random_state allows the split to be reproduced
# stratify=y not used in this case
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=30)

# epoch is one forward and backward pass of all training samples (also an iteration)
# eta0 is rate of convergence
# max_iter, tol, if it is too low it is never achieved
# and continues to iterate to max_iter when above tol
# fit_intercept, fit the intercept or assume it is 0
# slowing it down is very effective, eta is the learning rate
# The normalize variable is used to mention explicitly if the normalization of data
#is to be performed or not based on the standard scaler concept. This highly depends
#upon the distribution of the data
def train_data(classifier_test, normalize = False):

    # scale X by removing the mean and setting the variance to 1 on all features.
    # the formula is z=(x-u)/s where u is the mean and s is the standard deviation.
    # (mean and standard deviation may be overridden with options...)

    if(normalize == True):
        sc = StandardScaler()  # create the standard scalar
        sc.fit(X_train)  # compute the required transformation
        X_train_std = sc.transform(X_train)  # apply to the training data
        X_test_std = sc.transform(X_test)  # and SAME transformation of test data!!!
    else:
        X_train_std = X_train
        X_test_std = X_test

    #Performs a model fit based upon the classifier used for testing
    classifier_test.fit(X_train_std, y_train)  # do the training

    #Predicts using the same model for the test data
    y_pred = classifier_test.predict(X_test_std)  # now try with the test data

    #Gets the accuracy on the test data
    test_acc = accuracy_score(y_test, y_pred)

    test_acc_num = round(test_acc * num_samples * 0.3,0)
    test_acc_per = round(test_acc * 100,2)

    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))
    y_combined_pred = classifier_test.predict(X_combined_std)

    #Gets the accuracy on the combined data
    com_acc = accuracy_score(y_combined, y_combined_pred)
    com_acc_num = round(com_acc * num_samples,2)
    com_acc_per = round(com_acc * 100,2)

    #Return the test data accuracy values and combined accuracy values
    return test_acc_num, test_acc_per, com_acc_num, com_acc_per

#Call the various classification algorithms for performing the classification.
#This section assigns the classifier and the classification is done within the
#function train_data

#Perceptron Classifier
classifier = Perceptron(max_iter=20, tol=1e-3, eta0=0.001, fit_intercept=True, random_state=0, verbose=False)
test_data_accuracy_num[0], test_data_accuracy_per[0], combined_accuracy_num[0], combined_accuracy_per[0] = train_data(classifier, True)

#Logistic regression classifier
classifier = LogisticRegression(C = 10, solver='liblinear', multi_class='ovr', random_state = 40)
test_data_accuracy_num[1], test_data_accuracy_per[1], combined_accuracy_num[1], combined_accuracy_per[1] = train_data(classifier, True)

#Support vector classifier
classifier = SVC(kernel = 'linear', C=1.0, random_state = 0)
test_data_accuracy_num[2], test_data_accuracy_per[2], combined_accuracy_num[2], combined_accuracy_per[2] = train_data(classifier, True)

#Decision tree classifier
classifier = DecisionTreeClassifier(criterion = 'entropy', max_depth = 5, random_state = 0)
test_data_accuracy_num[3], test_data_accuracy_per[3], combined_accuracy_num[3], combined_accuracy_per[3] = train_data(classifier, True)

#Random Forest classifier
classifier = RandomForestClassifier(criterion='entropy', n_estimators=10, random_state=1, n_jobs=2)
test_data_accuracy_num[4], test_data_accuracy_per[4], combined_accuracy_num[4], combined_accuracy_per[4] = train_data(classifier, True)

#K Nerest neighbors classifier
classifier = KNeighborsClassifier(n_neighbors=5,p=2,metric='minkowski')
test_data_accuracy_num[5], test_data_accuracy_per[5], combined_accuracy_num[5], combined_accuracy_per[5] = train_data(classifier)

print("\nAccuracy of Prediction\n")

#Create a dataframe for the table of accuracies
pandas.set_option('display.max_columns', None)
representation_mat = list(zip(combined_accuracy_per, combined_accuracy_num, test_data_accuracy_per, test_data_accuracy_num))
df = pandas.DataFrame(representation_mat, index = ['Perceptron', 'Logistic Regression', 'Support Vector', 'Decision Tree', 'Random Forest', 'K Nearest Neighbor'], columns = ['Combined Acc (in %)', 'Combined Acc(in #Samples)', 'Test Acc(in %)', 'Test Acc (in #Samples)'])

#Display the accuracy table
print(df)