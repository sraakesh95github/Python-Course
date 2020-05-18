import pandas as pnd

a = pnd.read_csv('data_banknote_authentication.txt', names = ['variance', 'skewness', 'curtosis', 'entropy', 'class'])
#temp = a[:,[1,2]]
temp = a.to_numpy()
#print(temp[:,[1,2]])
print(temp[:,[4]])