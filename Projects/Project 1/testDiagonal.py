import pandas as pd
import numpy as np
import operator

df = pd.read_csv('data_banknote_authentication.txt')
cor_mat = df.corr()
print(cor_mat)

size = df.values.shape[1]
print(size)

array = np.zeros((size, size))

for i in range(size):
    for j in range(size):
        array[i][i] = 1

print(array)

fin_mat = cor_mat - array

print(fin_mat)

array_index = np.zeros((size))
array_cor = np.zeros((size))

# temp = fin_mat.iloc[[1]]
# temp1 = temp.to_numpy()
# print(temp1)

for i in range(size):
    temp2 = fin_mat.iloc[[i]].to_numpy()
    idx = np.argmax(abs(temp2))
    array_index[i] = idx
    array_cor[i] = round(temp2[0][idx],2)

print(array_index)
# print(array_cor)

feature_array_var1 = ['variance', 'skewness', 'curtosis', 'entropy', 'class']
feature_array_var2 = []
for i in range(size):
    feature_array_var2.append(feature_array_var1[int(array_index[i])])

representation_mat = list(zip(feature_array_var1, feature_array_var2, array_cor))

print(representation_mat)
print("\nHighest Correlation Matrix")
df2 = pd.DataFrame(representation_mat, index = list(range(size)), columns = ['Variable 1', 'Variable 2', 'Correlation'])
print(df2)
#print(feature_array_var2)
#df = DataFrame(data)


