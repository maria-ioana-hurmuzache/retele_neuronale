import pickle
import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F

# Constants
NUM_INPUTS = 784
NUM_HIDDEN = 100
NUM_OUTPUTS = 10
LEARNING_RATE = 0.2
BATCH_SIZE = 64
DECAY_RATE = 0.002
NUM_EPOCHS = 50
PATIENCE = 10
TOLERANCE = 1e-5

train_file = "./input/extended_mnist_train.pkl"
test_file = "./input/extended_mnist_test.pkl"

with open(train_file, "rb") as fp:
    train = pickle.load(fp)

with open(test_file, "rb") as fp:
    test = pickle.load(fp)

train_data = []
train_labels = []
for image, label in train:
    train_data.append(image.flatten() / 255.0)
    train_labels.append(label)

test_data = []
for image, label in test:
    test_data.append(image.flatten() / 255.0)

# Changing to numpy
X_train = np.array(train_data)
NUM_INSTANCES = X_train.shape[0]

train_labels = np.array(train_labels)
Y_train = np.eye(NUM_OUTPUTS)[train_labels]
# reminder: np.eye creates the identity matrix and then I associate
#           to each instancețs class the onehot encoding which is 
#           the line from the identity matrix of equal index to the
#           class (e.g.: 3 -> 0001000000) 

X_test = np.array(test_data)

# Functions:

def relu(Z):
    return np.maximum(0, Z)

def relu_derivative(Z):
    return (Z > 0).astype(float)

def softmax(Z):
    """Computes Softmax Activation"""
    # normalizing z so that the exponential doesn't become too big or too small
    # process: subtract the maximum in the line
    stable_Z = Z - np.max(Z, axis=1, keepdims=True)
    exp_Z = np.exp(stable_Z)
    sum_exp_Z = np.sum(exp_Z, axis=1, keepdims=True)
    A = exp_Z / sum_exp_Z
    return A

def forward_propagation(X, W1, b1, W2, b2):
    """Computes Z and Y"""
    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def backward_and_update(X, Y, Z1, A1, A2, W1, b1, W2, b2, learning_rate, l2_lambda):
    """Computes gradients and updates W and b."""
    m = X.shape[0]
    dZ2 = (A2 - Y)
    dW2 = np.dot(A1.T, dZ2) / m + l2_lambda * W2
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m

    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * relu_derivative(Z1)
    dW1 = np.dot(X.T, dZ1) / m + l2_lambda * W1
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    return W1, b1, W2, b2    

def cross_entropy_loss(Y, Z):
    logits = torch.tensor(Z, dtype=torch.float32)
    targets = torch.tensor(np.argmax(Y, axis=1), dtype=torch.long)
    # the loss is computed per instance and then the mean is computed to give the overall loss 
    loss = F.cross_entropy(logits, targets, reduction='mean')
    return loss.item()

def accuracy(Y, A):
    """Computes the accuracy"""
    return np.mean(np.argmax(Y, axis=1) == np.argmax(A, axis=1))

# Training

# Initialization
np.random.seed(14)
W1 = np.random.randn(NUM_INPUTS, NUM_HIDDEN) * np.sqrt(2.0 / NUM_INPUTS)
# He initialization: For ReLU activation, only 50% of neurons are activated on average (others are 0).
#                  To maintain constant variation between layers, He scales the normal distribution
#                  with sqrt(2/nr_intrari) which makes the variation of each output neuron fiecărui to be
#                  proportional with size of entry, avoiding saturation in activation and gradient.
b1 = np.zeros((1, NUM_HIDDEN))
W2 = np.random.randn(NUM_HIDDEN, NUM_OUTPUTS) * np.sqrt(2.0 / NUM_HIDDEN)
b2 = np.zeros((1, NUM_OUTPUTS))

best_loss =  float('inf')
best_params = (W1.copy(), b1.copy(), W2.copy(), b2.copy())
no_improvement_epochs = 0

# Training loop
for epoch in range(NUM_EPOCHS):
    lr = LEARNING_RATE / (1 + DECAY_RATE * epoch)
    
    permutation = np.random.permutation(NUM_INSTANCES)
    X_shuffled = X_train[permutation]
    Y_shuffled = Y_train[permutation]

    epoch_loss = 0
    num_batches = 0

    for i in range(0, NUM_INSTANCES, BATCH_SIZE):
        X_batch = X_shuffled[i:i+BATCH_SIZE]
        Y_batch = Y_shuffled[i:i+BATCH_SIZE]
        
        Z1, A1, Z2, A2 = forward_propagation(X_batch, W1, b1, W2, b2)
        batch_loss = cross_entropy_loss(Y_batch, Z2)
        W1, b1, W2, b2 = backward_and_update(X_batch, Y_batch, Z1, A1, A2, W1, b1, W2, b2, lr, l2_lambda=1e-4)

        epoch_loss += batch_loss
        num_batches += 1

    # Validation
    loss = epoch_loss / num_batches
    _, A1_all, _, A_all = forward_propagation(X_train, W1, b1, W2, b2)
    current_accuracy = accuracy(Y_train, A_all)

    if best_loss - loss > TOLERANCE:
        best_loss = loss
        best_params = (W1.copy(), b1.copy(), W2.copy(), b2.copy())
        no_improvement_epochs = 0
        print(f"Epoch {epoch+1}: Improved -> Loss={loss:.4f}, Acc={current_accuracy*100:.2f}%")
    else:
        no_improvement_epochs += 1
        print(f"Epoch {epoch+1}: No improvement ({no_improvement_epochs}) - Loss={loss:.4f}, Acc={current_accuracy*100:.2f}%")
        if no_improvement_epochs >= PATIENCE:
            break

# Outputing predicitions

W1, b1, W2, b2 = best_params
_, _, _, A_pred = forward_propagation(X_test, W1, b1, W2, b2)
predictions = np.argmax(A_pred, axis=1)

predictions_csv = {
    "ID": [],
    "target": [],
}

for i, label in enumerate(predictions):
    predictions_csv["ID"].append(i)
    predictions_csv["target"].append(label)

df = pd.DataFrame(predictions_csv)
df.to_csv("submission.csv", index=False)