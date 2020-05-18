import matplotlib.pyplot as plt

x_list1 = [1,2,3,4,5]
y_list1 = [12,13,14,15,16]
x_list2 = [i for i in range(10)]
y_list2 = [i**2 for i in x_list2]

plt.plot(x_list1,y_list1, label = "test graph")
plt.plot(x_list2,y_list2, label = "test 2 graph")
plt.xlabel("x values")
plt.ylabel("y values")

plt.show()