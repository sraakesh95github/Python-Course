from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.neural_network import MLPClassifier
import pandas as pd
from warnings import filterwarnings
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

# Import the required data from the csv file
df = pd.read_csv("sonar_all_data_2.csv")

# Filter the convergence warnings that arise after the max_iter exceeds the max set
filterwarnings('ignore')

# Split the dataset into feature matrix and target vector
no_columns_df = df.shape[1] - 2
column_labels = []
for i in range(no_columns_df + 2):
    column_labels.append("Time Stamp" + str(i))
df.columns = column_labels
X = df.iloc[:, 0: no_columns_df - 1]
y = df.iloc[:, no_columns_df]
print(X, y)

# Split the training and the testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.3)

# Setup the search parameters for the MLPClassifier to perform the gridSearchCV
# solver_options = ['adam']
# learning_rate_init_opt = np.arange(0.0001, 0.01, 0.0001)
# activation_opt = ['relu']
# alpha_opt = np.arange(0.0001, 0.01, 0.0001)
hidden_layer_sizes_opt = [(10, 10, 10, ), (100, )]

# Create a dictionary to pass the arguments to the GridSearchCV
#  solver=solver_options,
#  learning_rate_init=learning_rate_init_opt,
#  activation=activation_opt,
param_grid = dict(
                  hidden_layer_sizes=hidden_layer_sizes_opt
                  )

# Setup the classifier model
mlp = MLPClassifier(solver="adam", activation="relu", learning_rate_init=0.0001, alpha=0.0009, max_iter=5000, tol=0.0001, random_state=1)

# Perform the gridSearchCV based upon the mlp and the parameter grid ; "cv"\
# represents the cross-validation set to be taken from the entire data (Feature matrix and\
# the target vector.
# Also, specify based on what the "scoring" of the parameters need to be done
# # "n_jobs" specifies the parallelism of the processes of the grid search based on the
# # capability of the computer
grid = GridSearchCV(mlp, param_grid, cv=10, scoring='accuracy', n_jobs=-1, verbose=True)

# Save the scores
best_scores_test = []
model_params = []

# Iterate the PCA
for i in np.arange(4, 40, 1):

    # Print the iteration number
    print("\n\nIteration: " + str(i))

    # Set the PCA transform parameters
    pca = PCA(n_components=i, random_state=1)
    X_pca = pca.fit_transform(X_train)

    # Perform a fit on the gridSearchCV model
    grid.fit(X_pca, y_train)

    # Display the various grid parameters
    temp = grid.best_score_
    best_scores_test.append(temp)
    model_params.append(grid.best_estimator_)
    print(temp)
    # print(grid.cv_results_)
    print(grid.best_params_)
    print(grid.best_estimator_)

# Get the max of the best scores and print that estimator parameters
print("The max accuracy occured at the PCA component: " + str(max(best_scores_test)))
print("The parameters corresponding to the max accuracy for cross validation: ", best_scores_test[best_scores_test.index(max(best_scores_test))])