import numpy as np

test = np.array([1,2,3])

for i in test:
    print(i)
    print(type(i))



test = np.random.random_integers (0, 100, 100)
print(test)
print(len(test))
for i in test.flat:
    print(i)
