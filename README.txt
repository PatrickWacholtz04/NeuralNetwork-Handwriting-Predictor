Neural Network for Handwritten Digit Recognition
This Python project implements a Neural Network using backward propagation to identify handwritten digits from the MNIST dataset. 
The program utilizes pandas for reading the .csv files and heavily relies on NumPy arrays for efficient computation.
The dataset is comprised of 60,000 training cases and 10,000 test cases. The MNIST dataset used can be found on Kaggle at
https://www.kaggle.com/datasets/oddrationale/mnist-in-csv?select=mnist_test.csv.

Python Script Description
train.py:
    This is the training script and it will calculate the weights and biases based on the MNIST training dataset. During the script
    execution, every 10th iteration the script will output the current iteration, prediction, and accuracy. After completing the
    specified iterations the script will save the weights and balances to weights.npz. If this file does not already exist it 
    will create the file, otherwise, it will overwrite it.
read.py:
    This is the prediction script and it will read the weights and biases from weights.npz. It will then use them to make a prediction
    about a random image from the testing dataset. It will make predictions about n different images, where n is equal to numTests. For
    each prediction, the script will output the Neural Networks prediction, the actual value of the image, and display the image itself
    for the user to verify the prediction and actual value.

#############################################################################################################################
To run the program, extract mnist_test.csv and mnist_train.csv from dataset.zip into the same folder as train.py and read.py. 
#############################################################################################################################

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
    numTests = 5
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
    numTests = 5
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
    numTests = 5
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
    numTests = 5
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
