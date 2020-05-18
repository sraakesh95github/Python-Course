import numpy as np
print(np.size(np.arange(1,2.1,0.2)))

test = np.arange(1,2.1,0.2)

for a in np.nditer(test):
    print(a)
    
i = 0
for s in range(20):
    print(i)
    i+=1
