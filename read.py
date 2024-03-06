import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

#----------Functions----------#
def forwardPropagation(W1, b1, W2, b2, x):
    #z1 is the linear transformation of the data based on the current weights and biases
    Z1 = W1.dot(x) + b1
    #apply the ReLU function
    A1 = relu(Z1)
    #apply the current weights and biases to a1
    Z2 = W2.dot(A1) + b2
    #apply the SoftMax function to z2 to normalize
    A2 = softMax(Z2)

    return Z1, A1, Z2, A2


def prediction(index, W1, b1, W2, b2):
    #extract the current image
    currImage = xTest[:, index, None]
    #make a prediction
    prediction = makePredictions(xTest[:, index, None], W1, b1, W2, b2)
    label = yTest[index]
    #output the prediction to console
    print("Prediction=", prediction)
    print("Label=", label)
    
    #show reference image for user to verify
    currImage = currImage.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(currImage, interpolation='nearest')
    plt.show()

def makePredictions(X, W1, b1, W2, b2):
    #call the forward propagation method
    _, _, _, A2 = forwardPropagation(W1, b1, W2, b2, X)
    #call the get predictions method
    predictions = getPredictions(A2)
    return predictions

def getPredictions(a2):
    #return the maximum from each a2 column 
    return np.argmax(a2, 0)

def relu(x):
    #return the max between 0 and x
    return np.maximum(0 ,x)

def softMax(x):
    #softmax function
    softmax = np.exp(x) / sum(np.exp(x) )
    return softmax

#----------Execution----------#
#import data from testing dataset
data = pd.read_csv('mnist_test.csv')
data = np.array(data)
m, n = data.shape
#shuffle the data in the numpy array
np.random.shuffle(data) # shuffle before splitting into dev and training sets

#select the first 1000 images from the shuffled dataset
dataDev = data[0:1000].T
#extract the labels
yDev = dataDev[0]
#extract the rest of the array
xDev = dataDev[1:n]
#normalize the data
xDev = xDev / 255 

#select the remaning portion dataset 
dataTest = data[1000:m].T
#extract the labels
yTest = dataTest[0]
#extract the rest of the array
xTest = dataTest[1:n]
#normalize the data
xTest = xTest / 255.
mTest = xTest.shape[1]

#load the weights and biases from file
weights = np.load('weights.npz')

W1 = weights['W1']
b1 = weights['b1']
W2 = weights['W2']
b2 = weights['b2']

#select number of items to test
numTests = 5

#run the predictions numTest times
for test in range(1, numTests+1):
    prediction(test, W1, b1, W2, b2)