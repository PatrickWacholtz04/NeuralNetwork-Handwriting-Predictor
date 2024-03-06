Neural Network using backwards propagation to identify handwrittin digits.
The data set used for this program is the MNIST dataset which can be found here:
https://www.kaggle.com/datasets/oddrationale/mnist-in-csv?select=mnist_test.csv
The dataset is split into 10,000 test cases and 60,000 training cases.
pandas is used to read these .csv files
Numpy arrays are heavily utilized as well.

train.py will calculate the weights and biases based on the training data set. During its runtime, 
every 10th iteration it will output the current iteration, the prediction, and the accuracy.
After it is finished running it will output the weights and biases to weights.npz
read.py will read the wegihts and biases from weights.npz and use them to make guesses.
It will make n guesses where n is equal to numTests. For each guess, it will output what the
prediction is as well as the actual value. Additionally, it will show a grayscale image of the character
to the user to verify what the actual value is.

Sample run 1:
train.py:
    Iterations = 250
    Final Accuracy = 0.7811016949152543
read.py
    numTests = 5
    Prediction= [1]
    Label= 1
    Prediction= [9]
    Label= 9
    Prediction= [9]
    Label= 9
    Prediction= [5]
    Label= 5
    Prediction= [7]
    Label= 7

Sample run 2:
train.py:
    Iterations = 250
    Final Accuracy = 0.8124915254237288
read.py:
    Prediction= [1]
    Label= 1
    Prediction= [3]
    Label= 5
    Prediction= [5]
    Label= 5
    Prediction= [8]
    Label= 8
    Prediction= [1]
    Label= 1

Sample run 3:
train.py:
    Iterations = 250
    Final Accuracy = 0.7735254237288136
read.py:
    Prediction= [0]
    Label= 0
    Prediction= [9]
    Label= 9
    Prediction= [2]
    Label= 2
    Prediction= [6]
    Label= 6
    Prediction= [0]
    Label= 0

Sample run 4:
train.py:
    Iterations = 250
    Final Accuracy = 0.7378813559322034
read.py:
    Prediction= [9]
    Label= 9
    Prediction= [3]
    Label= 3
    Prediction= [8]
    Label= 8
    Prediction= [1]
    Label= 1
    Prediction= [6]
    Label= 6

Sample run 5:
train.py:
    Iterations = 250
    Final Accuracy = 0.7396101694915255
read.py:
    Prediction= [0]
    Label= 0
    Prediction= [1]
    Label= 5
    Prediction= [8]
    Label= 8
    Prediction= [3]
    Label= 3
    Prediction= [3]
    Label= 3