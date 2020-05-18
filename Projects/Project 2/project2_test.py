# import the required packages for processing the required data
import pandas as pd
from warnings import filterwarnings
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import numpy as np
# To calculate the time required for the completion of the execution
import time
import sys
from sklearn.metrics import confusion_matrix

# Initialize the parameters for learning
HIDDEN_LAYER_SIZES = (10, 10, 10, 10, )
test_accuracies = []
comb_accuracies = []
num_components = []
start_time = 0
end_time = 0
estimated_time = 0
# Data to analyze the random seed
maximum_accuracies = []
num_random_seeds = []
# Random state based accuracies
test_acc_rnd = []

# Ignore the warnings
filterwarnings('ignore')

# Initialize the required values for the program to run
column_labels = []

# Read the csv file into a python dataframe
df = pd.read_csv('sonar_all_data_2.csv')

# Provide the column labels for the data
num_features = df.shape[1]-2
representation_matrix = np.array((num_features, 2))

for i in range(num_features):
    temp = 'Time Sample' + str(i+1)
    column_labels.append(temp)

column_labels.append('Class integer')
column_labels.append('Class string')

df.columns = column_labels

# Split the input parameters and the output parameters
X = df.iloc[:, 0:num_features - 1]
y = df.iloc[:, num_features]
# print(X)
# print(y)

# Apply the Principal Component Analysis
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=None)

# Apply standard normalization
sca = StandardScaler()
X_train_std = sca.fit_transform(X_train)
X_test_std = sca.transform(X_test)

print("Multi-layered perceptron learning started...\n")

# Init a counter for counting the number of iterations
counter = 0

# Loop over to find the consistency in the accuracy obtained
for i in range(10):

    # Loop to check the maximum accuracy with a change in the random seed parameter in the MLPClassifier
    for random_seed in np.arange(0, 5, 1):

        #Status display
        print("\rCompleting random seed: " + str(random_seed+1))
        sys.stdout.flush()

        # Loop to change the number of PCA components for every iteration and check the accuracy of the PCA
        for i in range(1, num_features):

            start_time = int(time.time())

            # Apply Principal component analysis
            pca = PCA(n_components=i)
            X_train_pca = pca.fit_transform(X_train_std)
            X_test_pca = pca.transform(X_test_std)

            # Create a MultiLayered Perceptron classifier
            mlp = MLPClassifier(activation='relu', max_iter=5000, solver='adam', alpha=0.0001,
                                hidden_layer_sizes=HIDDEN_LAYER_SIZES, tol=0.0001, n_iter_no_change=10,
                                learning_rate_init=0.0015, learning_rate='constant', random_state=random_seed)

            # print("Number of iterations: " + mlp.n_iter_)
            X_train_mlp = mlp.fit(X_train_pca, y_train)

            # Accuracy only for the test data
            y_predict = mlp.predict(X_test_pca)
            acc_test = round(accuracy_score(y_test, y_predict) * 100, 2)
            test_accuracies.append(acc_test)
            num_components.append(i)

            # Status display
            sys.stdout.write("\rCompleting PCA component number: " + str(i))
            # # Get the estimated time of completion of all the iterations
            # if estimated_time == 0:
            #     estimated_time = (int(time.time()) - start_time) * (num_features - 3)
            # else:
            #     estimated_time = estimated_time - (int(time.time()) - start_time)

            # sys.stdout.write("\rEstimated time of completion: " + str(estimated_time) + "seconds")
            sys.stdout.flush()

        # Set the maximum accuracies for every iteration on the number of PCA components
        max_acc_test = max(test_accuracies)
        maximum_accuracies.append(max_acc_test)
        num_random_seeds.append(counter)
        counter += 1
        test_acc_rnd = np.hstack((test_accuracies, test_acc_rnd))

    representation_matrix = np.vstack((num_components, test_accuracies)).T
    print("Part (2) Relation depicting the model accuracies along with the number\
    of PCA components used for the learning")
    df = pd.DataFrame(representation_matrix, columns=['#PCA Componenets',
                                                           'Test Accuracies'])
    print(df)

    # Code written to plot the relation between number of PCA components used and the accuracy of the model
    # plt.plot(num_components, test_accuracies)
    # plt.xlabel("Number of PCA components")
    # plt.ylabel("Accuracies of the the test samples")
    # plt.title("Accuracies for PCA components")
    # plt.show()

    # give the max number of PCA components used and the accuracy of the model
    print("Part (3) maximum accuracy along with the number of components that achieved\
    that accuracy")
    max_acc_test = max(maximum_accuracies)
    max_index = test_accuracies.index(max_acc_test)
    print(max_acc_test, max_index)
    #The -1 is added to the index as the index usually starts from 0. In this case, the number of components start from 0.
    print("Number of PCA components that accumulated maximum accuracy: " +
          str(num_components[max_index]-1))
    print("Accuracy for first %d PCA components: " %(max_index) + str(max_acc_test))

# # Get the convergence matrix
# con_mat = confusion_matrix(y_test, y_predict)
# print(con_mat)

# # Variations visible with the Random seed variations
# plt.plot(num_random_seeds, maximum_accuracies)
# plt.xlabel("Random seed number")
# plt.ylabel("Max test accuracy")
# plt.show()

