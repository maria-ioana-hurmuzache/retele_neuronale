import pickle
import os
import pandas as pd
import numpy as np
train_file = "./input/extended_mnist_train.pkl"
test_file = "./input/extended_mnist_test.pkl"

# Constants
NUM_INPUTS = 784
NUM_OUTPUTS = 10
LEARNING_RATE = 0.1
BATCH_SIZE = 64
DECAY_RATE = 0.001
NUM_EPOCHS = 100
PATIENCE = 10
TOLERANCE = 1e-5

with open(train_file, "rb") as fp:
    train = pickle.load(fp)

with open(test_file, "rb") as fp:
    test = pickle.load(fp)

# Reading data from files and normalizing it

# Train data
train_data = []
train_labels = []

for image, label in train:
    train_data.append(image.flatten() / 255.0)
    train_labels.append(label)

# Test data
test_data = []

for image, label in test:
    test_data.append(image.flatten() / 255.0)

X_train = np.array(train_data)
NUM_INSTANCES = X_train.shape[0]

train_labels = np.array(train_labels)
Y_train = np.eye(NUM_OUTPUTS)[train_labels]

X_test = np.array(test_data)

# Functions used
def softmax(Z):
    """Computes Softmax Activation"""
    # normalizing z so that the exponential doesn't become too big or too small
    # process: subtract the maximum in the line
    stable_Z = Z - np.max(Z, axis=1, keepdims=True)
    exp_Z = np.exp(stable_Z)
    sum_exp_Z = np.sum(exp_Z, axis=1, keepdims=True)
    A = exp_Z / sum_exp_Z
    return A

def forward_propagation(X, W, b):
    """Computes Z and Y"""
    Z = np.dot(X, W) + b
    A = softmax(Z)
    return Z, A

def cross_entropy_loss(Y, A):
    """Computes the error"""
    A_clipped = np.clip(A, 1e-12, 1.0)
    total_loss = np.mean(np.sum(Y * np.log(A_clipped)))
    error = -total_loss
    return error

def backward_and_update(X, Y, A, W, b, learning_rate):
    """Computes gradients and updates W and b."""
    m = X.shape[0]
    dZ = A - Y
    
    dW = np.dot(X.T, dZ) / m
    db = np.sum(dZ, axis=0, keepdims=True) / m
    
    W_new = W - learning_rate * dW
    b_new = b - learning_rate * db
    
    return W_new, b_new

def accuracy(Y, A):
    """Computes the accuracy"""
    return np.mean(np.argmax(Y, axis=1) == np.argmax(A, axis=1))


# Training section

# Initialization of the weights and biases
np.random.seed(145)
Weights = np.random.randn(NUM_INPUTS, NUM_OUTPUTS) * 0.01
bias = np.zeros((1, NUM_OUTPUTS))

# Bests initialization
best_Weights, best_bias = Weights.copy(), bias.copy()
_, A_all = forward_propagation(X_train, best_Weights, best_bias)
best_loss = cross_entropy_loss(Y_train, A_all)

no_improvement_epochs = 0

for epoch in range(NUM_EPOCHS):
    lr = LEARNING_RATE / (1 + DECAY_RATE * epoch)
    permutation = np.random.permutation(NUM_INSTANCES)
    X_shuffled = X_train[permutation]
    Y_shuffled = Y_train[permutation]

    for i in range(0, NUM_INSTANCES, BATCH_SIZE):
        X_batch = X_shuffled[i:i+BATCH_SIZE]
        Y_batch = Y_shuffled[i:i+BATCH_SIZE]

        Z, A = forward_propagation(X_batch, Weights, bias)
        Weights, bias = backward_and_update(X_batch, Y_batch, A, Weights, bias, lr)
    
    # Validation set check
    _, A_all = forward_propagation(X_train, Weights, bias)
    loss = cross_entropy_loss(Y_train, A_all)
    current_accuracy = accuracy(Y_train, A_all)

    if best_loss - loss > TOLERANCE:
        no_improvement_epochs = 0
        best_loss = loss
        best_Weights, best_bias = Weights.copy(), bias.copy()
        print(f"Improved loss: {best_loss=}. {current_accuracy=}")
    else:
        no_improvement_epochs += 1
        if(no_improvement_epochs >= PATIENCE):
            break
        print("No improvement")


# Submission preparation
predictions_csv = {
    "ID": [],
    "target": [],
}

_, A_predictions = forward_propagation(X_test, best_Weights, best_bias)
predictions = np.argmax(A_predictions, axis=1)

for i, label in enumerate(predictions):
    predictions_csv["ID"].append(i)
    predictions_csv["target"].append(label)

df = pd.DataFrame(predictions_csv)
df.to_csv("submission.csv", index=False)