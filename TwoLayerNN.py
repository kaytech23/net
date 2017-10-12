import numpy as np

#sigmoid
def nonlin(x, deriv=False):
	if(deriv==True):
		return x*(1-x)
	return 1/(1 + np.exp(-x))


#input dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

#output dataset
y = np.array([[0], [0], [0], [1]])

print("X: ", X)
print("y: ", y)

np.random.seed(1)

syn0 = 2 * np.random.random((2, 1)) - 1


#np.random.seed(1)
bias0 = np.random.random()
print(syn0)
print(bias0)


for _ in range(1):
	
	#forward prop
	l0 = X
	z = np.dot(l0, syn0) + bias0
	l1 = nonlin(z)
	
	#error
	print(l1 -y)
	print((l1 - y)**2)
	print(np.sum((l1 - y)**2))
	error = np.sum((l1 - y)**2) / (2 * 4)
	print("error: ", error)
	
	#error deriative
	l1_error = y - l1
	
	#delta
	l1_delta = l1_error * nonlin(l1, True)
	
	#update w
	syn0 += np.dot(l0.T, l1_delta)
	bias0 += np.sum(l1_delta)
	
print("Done")
print(l1)

print(2 * np.random.random((2, 1)) - 1)