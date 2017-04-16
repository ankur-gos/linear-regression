'''
    linear_regression.py
    Ankur Goswami, agoswam3@ucsc.edu
    Emily Hockel, echockel@ucsc.edu
    P3 on HW1
'''

import numpy as np
import matplotlib as mp

def generateMSE(num_examples, variance):
    true_theta = [2, 1]
    result_features = []
    result_labels = []
    for i in range(0, num_examples):
        f1, f2 = np.random.rand(-1, 1), np.random.rand(-1, 1)
        label = 2 * f1 - f2 + generateNoise(variance)
        result_features.append((f1, f2))
        result_labels.append(label)
    return result_features, result_labels

def generateNoise(variance):
    return np.random.randn(0, variance)

def minimize(X, y):
    X_t = np.transpose(X)
    mul = np.dot(X, X_t)
    inv = np.linalg.inv(mul)
    mul2 = np.dot(inv, X_t)
    return np.dot(mul2, y)

def generateSets(m=10000, k):
    training_features, training_labels = generateMSE(m, 0.01)
    test_features, test_labels = generateMSE(m, 0.01)
    X = np.matrix(training_features[:k])
    label_y = np.matrix(training_labels)
    result_vector = minimize(X, label_y)
    print result_vector
    # training_errors = label_y - result_vector
