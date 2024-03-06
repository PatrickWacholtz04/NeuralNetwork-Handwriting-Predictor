import numpy as np
import pandas as pd

#----------Functions----------#
def gradientDescent(x, y, alpha, iterations):
    #neural net function calls
    W1, b1, W2, b2 = generateParamaters()
    for i in range(iterations):
        z1, a1, z2, a2 = forwardPropagation(W1, b1, W2, b2, x)
        dW1, db1, dW2, db2 = backwardsPropagation(z1, a1, z2, a2, W1, W2, x, y)
        W1, b1, W2, b2 = updateParamaters(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        #output the current interation, predictons, and accuray
        if i % 10 == 0:
            print("Iteration=", i)
            predictions = getPredictions(a2)
            print("Accuracy=", getAccuracy(predictions, y) )
    return W1, b1, W2, b2

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

def backwardsPropagation(Z1, A1, Z2, A2, W1, W2, x, y):
    #execute one hot encoding
    oneHot = oneHotEncodeing(y)
    #calculate the gradient of weights/biases
    dZ2 = A2 - oneHot
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * reluDerivative(Z1)
    dW1 = 1 / m * dZ1.dot(x.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2

def oneHotEncodeing(y):
    #create empty(zeros) np array of size y.size by y.max()+1
    encoded = np.zeros( (y.size, y.max()+1) )
    #convert the catagories into 1s instead of 0s 
    encoded[np.arange(y.size), y] = 1
    encoded = encoded.T
    return encoded

def generateParamaters():
    #randomly generate the weights and biases
    #The dataset images are 28x28, which means there are 784 pixels.
    #subtracting 0.5 from all the weights and biases will center them around 1
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5

    return W1, b1, W2, b2

def updateParamaters(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    #update the weights/biases by subtracting the alpha * the gradient 
    #from the weight/bias
    W1 = W1 - alpha * dW1
    W2 = W2 - alpha * dW2
    b1 = b1 - alpha * db1
    b2 = b2 - alpha * db2
    #return the updated weights and biases
    return W1, b1, W2, b2

def relu(x):
    #return the max between 0 and x
    return np.maximum(0 ,x)

def reluDerivative(x):
    #derivative of the relu function
    return x > 0

def softMax(x):
    #softmax function
    softmax = np.exp(x) / sum(np.exp(x) )
    return softmax

def getPredictions(a2):
    #return the maximum from each a2 column 
    return np.argmax(a2, 0)

def getAccuracy(predictions, y):
    print(predictions, y)
    #calculate the number of correct predictions and divide by total predictions
    return np.sum(predictions == y) / y.size


#----------Execution----------#
#import data from training dataset
data = pd.read_csv('mnist_train.csv')
data = np.array(data)
m, n = data.shape
#shuffle the data in the numpy array
np.random.shuffle(data)

#select the first 1000 images from the shuffled dataset
dataDev = data[0:1000].T
#extract the labels
yDev = dataDev[0]
#extract the rest of the array
xDev = dataDev[1:n]
#normalize the data
xDev = xDev / 255 

#select the remaning portion dataset 
dataTrain = data[1000:m].T
#extract the labels
yTrain = dataTrain[0]
#extract the rest of the array
xTrain = dataTrain[1:n]
#normalize the data
xTrain = xTrain / 255.
#assign the number of rows to mTrain
_, mTrain = xTrain.shape

#execute the gradient descent
W1, b1, W2, b2 = gradientDescent(xTrain, yTrain, 0.10, 250)

#save the weights and biases to a file 
print("Saving weights and biases to file...")
np.savez("weights.npz", W1=W1, b1=b1, W2=W2, b2=b2)
print("Saved")