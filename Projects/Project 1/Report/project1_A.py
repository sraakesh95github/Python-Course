# Correlation examples
# author: olhartin@asu.edu updated by sdm

import numpy as np                      # needed for arrays and math
import pandas as pd                     # needed to read the data
import matplotlib.pyplot as plt         # used for plotting
import seaborn as sns                   # data visualization

#Function to find the highest correlation between points - variable - wise
def cor_matrices(dataframe):

    # Find the correlation of all the variables of the data set
    cor_mat = dataframe.corr()
    size = dataframe.values.shape[1]
    array_index = np.zeros((size))
    array_cor = np.zeros((size))
    array_cor_class_temp = np.zeros((size-1))
    array_cor_class_index = np.zeros((size-1))
    cols_temp = cor_mat.iloc[:]['Class']

    #Remove the number 1 from the diagonal elements of the correlation matrix
    array = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            array[i][i] = 1
    fin_mat = cor_mat - array

    #Find the high correlation for all the variables with every other variable and store it in an array
    #and the values for the correlation
    for i in range(size):
        temp2 = fin_mat.iloc[[i]].to_numpy()
        idx = np.argmax(abs(temp2))
        array_index[i] = idx
        array_cor[i] = round(temp2[0][idx],2)
        if(i!=size-1):
            array_cor_class_temp[i] = temp2[0][size-1]

    #Finding the oder / ranking of the variables with the most correlation with the variable of prediction
    #Variable of prediction is class in this case
    #array_cor_class_index represents the correlation ranking of the entities with the class variable
    array_cor_class = np.sort(abs(array_cor_class_temp[0:size-1]))
    array_cor_class = np.flipud(array_cor_class)
    for i in range(size-1):
        for j in range(size-1):
            if(array_cor_class[j] == abs(array_cor_class_temp[i])):
                array_cor_class_index[i] = j
                break

    feature_array_var1 = ['variance', 'skewness', 'curtosis', 'entropy', 'class']
    feature_array_lab = []
    feature_array_cor = []
    for i in range(size-1):
        feature_array_lab.append(feature_array_var1[int(array_cor_class_index[i])])
        feature_array_cor.append(round(cols_temp[int(array_cor_class_index[i])],2))

    #Create a representation for the correlation of the individual variables with the Class variable
    print("\nTable 4: Correlation (with variable to be predicted - \'Class\') \n")
    representation_mat1 = list(zip( feature_array_lab, feature_array_cor))
    df3 = pd.DataFrame(representation_mat1, index=list(range(size-1)), columns=['Variable', 'Correlation with class'])
    print(df3)

    #Create a list of all the variables compared with
    feature_array_var2 = []

    #Create a list of variables with the highest correlation with the compared variable
    for i in range(size):
        feature_array_var2.append(feature_array_var1[int(array_index[i])])

    #Tabular representation of the matric in the form of dataframe
    representation_mat2 = list(zip(feature_array_var1, feature_array_var2, array_cor))

    #Print the representation matrix
    print("\nTable 5: Highest Correlation Matrix (Element-wise distribution) - Table 2\n")
    df2 = pd.DataFrame(representation_mat2, index = list(range(size)), columns = ['Variable 1', 'Variable 2', 'Correlation'])
    print(df2)

def mosthighlycorrelated(mydataframe, numtoreport):
    # find the correlations
    cormatrix = mydataframe.corr()
    print("\nTable 1: Initial Correlation Matrix - Table 3\n")
    print(cormatrix)

    #Find the covariance
    covariance = mydataframe.cov()
    print("\nTable 2: Covariance Matrix\n")
    print(covariance)

    #Upper triangular matrix is considered for multiplication because the matrix
    #is a symmetric matrix and hence the lower triangular elements are the same as the upper triangular matrix
    #The diagonal elements are not required as it accounts for the correlation to be 1 which the maximum for all
    #the variables and it doesn't give the required relationship or dependency on the other variables
    cormatrix *= np.tri(*cormatrix.values.shape, k=-1).T #.T refers to the transpose of the correlation matrix at the end

    # find the top n correlations
    cormatrix = cormatrix.stack()

    # Reorder the entries so they go from largest at top to smallest at bottom
    # based on absolute value
    cormatrix = cormatrix.reindex(cormatrix.abs().sort_values(ascending=False).index).reset_index()


    # assign human-friendly names
    cormatrix.columns = ["FirstVariable", "SecondVariable", "Correlation"]
    print("\nTable 3: Most Highly Correlated\n")
    print(cormatrix.head(numtoreport))     # print the top values

# The plot that defines the plots based upon every 2 variables. It contains the histograms and the colorgraphs by default
def pairplotting(df):
    sns.set(style='ticks', context='notebook')   # set the apearance
    sns.pairplot(df,height=2.5)                      # create the pair plots
    plt.show()                                       # and show them
    sns.pairplot(df,height=2.5,hue = 'Class')
    plt.show()

# this creates a dataframe similar to a dictionary
# a data frame can be constructed from a dictionary
# https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html
cntft_notes = pd.read_csv('data_banknote_authentication.txt', names = ['Variance', 'Skewness', 'Curtosis', 'Entropy', 'Class'])
print("\nDatabase size:" + str(cntft_notes.values.shape[0]))
cols = cntft_notes.columns

mosthighlycorrelated(cntft_notes,5)                # generate most highly correlated list
pairplotting(cntft_notes)                          # generate the pair plot
cor_matrices(cntft_notes)

#  descriptive statistics
print('\nDescriptive Statistics')
print(cntft_notes.describe()) #describe used in dataframes







