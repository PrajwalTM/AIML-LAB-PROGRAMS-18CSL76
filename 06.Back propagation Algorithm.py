PROGRAM.No.6 (Back propagation Algorithm)
"""Build an Artificial Neural Network by implementing the Back propagation algorithm and test the same using appropriate data sets.""" 


import numpy as np
X = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)
y = np.array(([92], [86], [89]), dtype=float)
X = X/np.amax(X, axis=0)
y = y/100
def sigmoid(x):
 return 1/(1+np.exp(-x))
def derivatives_sigmoid(x):
 return x*(1-x)
epoch = 7000
lr = 0.1
inputLayer_neurons = 2
hiddenLayer_neurons = 3
output_neurons = 1
wh = np.random.uniform(size=(inputLayer_neurons, hiddenLayer_neurons))
bh = np.random.uniform(size=(1, hiddenLayer_neurons))
wout = np.random.uniform(size=(hiddenLayer_neurons, output_neurons))
bout = np.random.uniform(size=(1, output_neurons))
for i in range(epoch):
 hinp1 = np.dot(X, wh)
 hinp = hinp1 + bh
 hlayer_act = sigmoid(hinp)
 outinp1 = np.dot(hlayer_act, wout)
 outinp = outinp1 + bout
 output = sigmoid(outinp)
 EO = y - output
 outgrad = derivatives_sigmoid(output)
 d_output = EO * outgrad
 EH = d_output.dot(wout.T)
 hiddengrad = derivatives_sigmoid(hlayer_act)
 d_hiddenlayer = EH * hiddengrad
 wout += hlayer_act.T.dot(d_output)*lr
 wh += X.T.dot(d_hiddenlayer)*lr
print('Input: \n' + str(X))
print('Actual Output: \n' + str(y))
print('Predicted output: \n', output)

########################################################################################################################
# OUTPUT:
# Ignore single quotes at beginning and end
########################################################################################################################

Input:
[[0.66666667 1. ]
[0.33333333 0.55555556]
[1. 0.66666667]]
Actual Output:
[[0.92]
[0.86]
[0.89]]
Predicted output:
[[0.89871247]
[0.871736 ]
[0.89883643]]
