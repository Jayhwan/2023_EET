import autograd.numpy as np
from autograd import grad, jacobian


def c1(input, coef):
    return np.dot(input, coef)


def f(input):
    y = np.array([2., 3., 4.])
    return c1(input, y)


def vec(input):
    y = np.random.random(len(input))
    z = np.random.random(len(input))
    print(y, z)
    return np.dot(input, y), np.dot(input, z)


grad_f = grad(f)
grad_c1 = grad(lambda x: c1(x, np.array([2., 3., 4.])))
x = np.array([1.,2.,3.], dtype=float)
print(grad_f(x))
print(grad_c1(x))
print(jacobian(vec)(3.))

#print(c1(x, y))
"""
def sigmoid(x):
    return 0.5 * (np.tanh(x / 2.) + 1)

def logistic_predictions(weights, inputs):
    # Outputs probability of a label being true according to logistic model.
    return sigmoid(np.dot(inputs, weights))

def training_loss(weights):
    # Training loss is the negative log-likelihood of the training labels.
    preds = logistic_predictions(weights, inputs)
    label_probabilities = preds * targets + (1 - preds) * (1 - targets)
    return -np.sum(np.log(label_probabilities))

# Build a toy dataset.
inputs = np.array([[0.52, 1.12,  0.77],
                   [0.88, -1.08, 0.15],
                   [0.52, 0.06, -1.30],
                   [0.74, -2.49, 1.39]])
targets = np.array([True, True, False, True])

# Define a function that returns gradients of training loss using Autograd.
training_gradient_fun = grad(training_loss)

# Optimize weights using gradient descent.
weights = np.array([0.0, 0.0, 0.0])
print("Initial loss:", training_loss(weights))
print(training_gradient_fun(weights))
for i in range(100):
    weights -= training_gradient_fun(weights) * 0.01

print("Trained loss:", training_loss(weights))
"""