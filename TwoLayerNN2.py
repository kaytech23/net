import numpy as np

#sigmoid
def nonlin(x, deriv=False):
	if(deriv==True):
		return x*(1-x)
	return 1/(1 + np.exp(-x))


#input dataset
X = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])

#output dataset
y = np.array([[0, 0, 0, 1]]).T

print("X: ", X)
print("y: ", y)

#np.random.seed(1)

syn0 = 2 * np.random.random((3, 1)) - 1
#np.random.seed(1)
bias0 = np.random.random()
print(syn0)
print(bias0)


for _ in range(10000):
	
	#forward prop
	l0 = X	
	l1 = nonlin(np.dot(l0, syn0))
	
	#error
	l1_error = y - l1
	#print(l1_error)
	
	#delta
	l1_delta = l1_error * nonlin(l1, True)
	
	#update w
	
	syn0 += np.dot(l0.T, l1_delta)
	print(l0.T)
	
print("Done")
print(l1)