# import the required packages for processing the required data
import pandas as pd
from warnings import filterwarnings
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import sys
from sklearn.metrics import confusion_matrix

# Initialize the parameters for learning
HIDDEN_LAYER_SIZES = [(100, 50, ), (50, 100, ), (10, 10, 10), (100, 100, 100)]
test_accuracies = []
comb_accuracies = []
num_components = []
start_time = 0
end_time = 0
estimated_time = 0

# Ignore the warnings
filterwarnings('ignore')

# Initialize the required values for the program to run
column_labels = []

# Read the csv file into a python dataframe
df = pd.read_csv('sonar_all_data_2.csv')

# Provide the column labels for the data
num_features = df.shape[1]-2
representation_matrix = np.array((num_features, 2))

# Setting the column labels for the timestamp details from the CSV
for i in range(num_features):
    temp = 'Time Sample' + str(i+1)
    column_labels.append(temp)

column_labels.append('Class integer')
column_labels.append('Class string')

df.columns = column_labels

# Split the input parameters and the output parameters
X = df.iloc[:, 0:num_features - 1]
y = df.iloc[:, num_features]

# Apply the Principal Component Analysis
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=1)

print("Multi-layered perceptron learning started...\n")

# Init a counter for counting the number of iterations
counter = 0

# Perform the test for various hidden layer sizes
for hidden_layers in HIDDEN_LAYER_SIZES:

    # Loop to change the number of PCA components for every iteration and check the accuracy of the PCA
    for i in range(1, num_features):

        # Apply Principal component analysis
        pca = PCA(n_components=i, random_state=1)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)

        # Create a MultiLayered Perceptron classifier
        mlp = MLPClassifier(activation='logistic',
                            max_iter=5000,
                            learning_rate_init=0.0012,
                            solver='adam',
                            alpha=0.0001,
                            hidden_layer_sizes=hidden_layers,
                            tol=0.0001,
                            learning_rate='constant',
                            random_state=1
                            )

        # print("Number of iterations: " + mlp.n_iter_)
        X_train_mlp = mlp.fit(X_train_pca, y_train)

        # Accuracy only for the test data
        y_predict = mlp.predict(X_test_pca)
        acc_test = round(accuracy_score(y_test, y_predict) * 100, 2)
        test_accuracies.append(acc_test)
        num_components.append(i)

        # Status display
        sys.stdout.write("\rCompleting PCA component number: " + str(i))
        sys.stdout.flush()

    # Set the maximum accuracies for every iteration on the number of PCA components
    max_acc_test = max(test_accuracies)
    counter = counter + 1

    representation_matrix = np.vstack((num_components, test_accuracies)).T
    print("\n\nPart (2) Relation depicting the model accuracies along with the number\
    of PCA components used for the learning")
    df = pd.DataFrame(representation_matrix, columns=['#PCA Componenets',
                                                           'Test Accuracies'])
    print(df)

    # Code written to plot the relation between number of PCA components used and the accuracy of the model
    plt.plot(num_components, test_accuracies)
    plt.xlabel("Number of PCA components")
    plt.ylabel("Accuracies of the the test samples")
    plot_title = ''
    for i in range(len(hidden_layers)):
        plot_title = plot_title + str(hidden_layers[i]) + ', '
    plt.title("Accuracies for PCA components with Hidden layer size: (" + plot_title + ')')
    plt.show()

    # give the max number of PCA components used and the accuracy of the model
    print("\nPart (3) Maximum accuracy achieved: ")
    max_index = test_accuracies.index(max_acc_test)
    print(max_acc_test)
    print("The number of components that achieved max accuracy: ")
    print(max_index)

    # The -1 is added to the index as the index usually starts from 0. In this case, the number of components start from 0.
    print("Number of PCA components that accumulated maximum accuracy: " +
          str(num_components[max_index]-1))
    print("Accuracy for first %d PCA components: " %(max_index) + str(max_acc_test))
    print("\nPart (4) Confusion matrix")

    # Get the convergence matrix
    con_mat = confusion_matrix(y_test, y_predict)
    print(con_mat)

    # Save the plot for comparisons
    plt.savefig("plot.png")

    # Write the result into a test file


# Plot the PCA Scree plot
# variance_per = np.round(pca.explained_variance_ratio_ * 100, decimals=2)
# Create the column labels list for the PCA
# pca_col_lab = []
# for i in range(num_features):
#     pca_col_lab.append("PCA" + str(i))
# print(num_features, len(pca_col_lab), variance_per.shape)
# # plot the required graph
# plt.bar(x=range(1, num_features), height=variance_per, tick_label=column_labels)
# plt.xlabel("#PCA components")
# plt.ylabel("Variance percentage in the dataset")
# plt.title("PCA Scree Plot")
# plt.show()




