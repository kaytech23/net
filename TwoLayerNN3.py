import numpy as np

#sigmoid
def nonlin(x, deriv=False):
	if(deriv==True):
		return x*(1-x)
	return 1/(1 + np.exp(-x))


#input dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

#output dataset
y = np.array([[0], [1], [1], [0]])

print("X: ", X)
print("y: ", y)

np.random.seed(1)

syn0 = 2 * np.random.random((2, 4)) - 1
bias0 = np.random.random()

syn1 = 2 * np.random.random((4, 1)) - 1
bias1 = np.random.random()


print(syn1)
print(bias1)


for _ in range(100000):
	
	l0 = X
	
	z = np.dot(l0, syn0) + bias0
	l1 = nonlin(z)
		
	z = np.dot(l1, syn1) + bias1
	l2 = nonlin(z)
	
	error = np.sum((l2 - y)**2) / (2 * 4)
	print("error: ", error)
	
	#error deriative
	l2_error = y - l2
	
	#delta
	l2_delta = l2_error * nonlin(l2, True)
	
	#update w
	syn1 += np.dot(l1.T, l2_delta)
	bias1 += np.sum(l2_delta)
	
	l1_error = l2_delta.dot(syn1.T)
	l1_delta = l1_error * nonlin(l1, deriv=True)
	
	syn0 += np.dot(l0.T, l1_delta)
	bias0 = np.sum(l1_delta)
	
	
print("Done")
print(l2)

