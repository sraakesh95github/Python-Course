import pandas as pd  # Required for reading the CSV dataset
from warnings import filterwarnings   # Required to filter convergence warnings
from sklearn.model_selection import train_test_split  # Required to split the training and test data
from sklearn.neural_network import MLPClassifier  # The main classification model based on Multi-Layered Perceptron
from sklearn.decomposition import PCA  # Pre-processing of data using Principal Component Analysis for choosing the features to be used for classification
from sklearn.metrics import accuracy_score  # Get the accuracy of the test set
import numpy as np  #Perform array manipulations for representing the data as a columnar view
import matplotlib.pyplot as plt  # Plot the test accuracy vs PCA components
import sys  # Used to flush the display after display of MLPClassifier status for each PCA component
from sklearn.metrics import confusion_matrix  # Create a confusion matrix

# Initialize the parameters for learning
HIDDEN_LAYER_SIZES = (100,)  # set the number of hidden layers as 1 and the size of the layer as 100
test_accuracies = []
num_components = []

# Ignore the warnings of incomplete convergence
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

# Split the input parameters and the output parameters from the dataframe
X = df.iloc[:, 0:num_features - 1]
y = df.iloc[:, num_features]

# Split the train and test data with a constant random seed and train to test ratio as 7:3
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=1)

print("Multi-layered perceptron learning started...\n")

# Init a counter for counting the number of iterations
counter = 0

# Loop to change the number of PCA components for every iteration and check the accuracy of the PCA
for i in range(1, num_features):

    # Apply Principal component analysis
    pca = PCA(n_components=i, random_state=1)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    # Create a MultiLayered Perceptron classifier
    mlp = MLPClassifier(activation='relu',
                        max_iter=5000,
                        learning_rate_init=0.0012,
                        solver='adam',
                        alpha=0.0001,
                        hidden_layer_sizes=HIDDEN_LAYER_SIZES,
                        tol=0.0001,
                        learning_rate='constant',
                        random_state=1,
                        n_iter_no_change=20
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
max_index = test_accuracies.index(max_acc_test)
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
for i in range(len(HIDDEN_LAYER_SIZES)):
    plot_title = plot_title + str(HIDDEN_LAYER_SIZES[i]) + ', '
plt.title("Part (4) Accuracies for PCA components with Hidden layer size: (" + plot_title + ')')
plt.show()

# give the max number of PCA components used and the accuracy of the model
# The -1 is added to the index as the index usually starts from 0. In this case, the number of components start from 0.
print("\n\nPart(3) \nNumber of PCA components that accumulated maximum accuracy: " +
      str(num_components[max_index]-1))
print("Accuracy for first %d PCA components: " %(max_index) + str(max_acc_test))

# Get the confusion matrix
print("\nPart (5) Confusion matrix")
con_mat = confusion_matrix(y_test, y_predict)
print(con_mat)
