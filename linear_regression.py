'''
    linear_regression.py
    Ankur Goswami, agoswam3@ucsc.edu
    Emily Hockel, echockel@ucsc.edu
    P3 on HW1
'''

import numpy as np
import random
import matplotlib.pyplot as plt


def generateMSE(num_examples, variance):
    true_theta = [2, 1]
    result_features = []
    result_labels = []
    for i in range(0, num_examples):
        f1, f2 = random.uniform(-1, 1), random.uniform(-1, 1)
        label = 2 * f1 - f2 + generate_noise(variance)
        result_features.append((f1, f2))
        result_labels.append(label)
    return result_features, result_labels

def generate_noise(variance):
    return variance * np.random.randn()

def minimize(X, y):
    X_t = np.transpose(X)
    mul = np.dot(X_t, X)
    inv = np.linalg.inv(mul)
    mul2 = np.dot(inv, X_t)
    return np.dot(mul2, np.transpose(y))

def calculate_result(weights, examples):
    return np.dot(np.transpose(weights), np.transpose(examples))

def calculate_error(results, labels):
    return labels - results

def graph_error(x, training, test):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, training)
    ax.plot(x, test)
    plt.show()

def generateSets(k, m=10000):
    training_features, training_labels = generateMSE(m, 0.01)
    test_features, test_labels = generateMSE(m, 0.01)
    X = np.matrix(training_features[:k])
    label_y = np.matrix(training_labels[:k])
    result_weight = minimize(X, label_y)
    training_result = calculate_result(result_weight, training_features)
    training_mean = np.mean(np.squeeze(np.asarray(calculate_error(training_result, training_labels))))

    test_result = calculate_result(result_weight, test_features)
    test_mean = np.mean(np.squeeze(np.asarray(calculate_error(test_result, test_labels))))
    return (training_mean, test_mean)
    # training_errors = label_y - result_vector

def calculate_j(theta, data, labels):
    f, rows = data.shape
    summ = 0
    for i in range(0, rows):
        res = np.squeeze(np.asarray(np.dot(np.transpose(theta), np.transpose(data[i,:]))))
        # print res - labels[i]
        summ += pow((res - labels[i]), 2)
    return summ / rows

def calculate_epsilon(theta, data, labels, closed_result):
    j = calculate_j(theta, data, labels)
    return j - closed_result

def squeeze(val):
    return np.squeeze(np.asarray(val))

def gradient_descent(alpha, data, labels, closed_result):
    theta = [0, 0]
    num_iterations = 0
    gd_result = calculate_epsilon(theta, data, labels, closed_result)
    # print gd_result
    gd_result_p = float('inf')
    while(gd_result < gd_result_p):
        gd_result_p = gd_result
        for j in range(0, len(theta)):
            # print 'Theta: ' + str(theta)
            theta_m = np.transpose(np.matrix(theta))
            summ = 0
            for i in range(0, data.shape[1]):
                res = np.squeeze(np.asarray(np.dot(np.transpose(theta_m), np.transpose(data[i,:]))))
                summ += (labels[i] - res)*np.transpose(data[i,:])[j]
            delta = (2*alpha / data.shape[1]) * summ
            theta[j] += squeeze(delta)
        gd_result = calculate_epsilon(theta_m, data, labels, closed_result)
        # print gd_result
        num_iterations += 1
    return gd_result, num_iterations

def gradient_descent_dalpha(alpha, data, labels, closed_result):
    theta = [0, 0]
    num_iterations = 0
    gd_result = calculate_epsilon(theta, data, labels, closed_result)
    # print gd_result
    gd_result_p = float('inf')
    while(alpha >= 0.001):
        if(gd_result >= gd_result_p):
            alpha /= 2
        gd_result_p = gd_result
        for j in range(0, len(theta)):
            # print 'Theta: ' + str(theta)
            theta_m = np.transpose(np.matrix(theta))
            summ = 0
            for i in range(0, data.shape[1]):
                res = np.squeeze(np.asarray(np.dot(np.transpose(theta_m), np.transpose(data[i,:]))))
                summ += (labels[i] - res)*np.transpose(data[i,:])[j]
            delta = (2*alpha / data.shape[1]) * summ
            theta[j] += squeeze(delta)
        gd_result = calculate_epsilon(theta_m, data, labels, closed_result)
        # print gd_result
        num_iterations += 1
    return gd_result, num_iterations

def run_gd_dalpha(alpha, data, labels):
    X = np.matrix(data)
    label_y = np.matrix(labels)
    result_weight = minimize(X, label_y)
    closed_result = calculate_j(result_weight, X, labels)
    gd_result, num_iterations = gradient_descent_dalpha(alpha, X, labels, closed_result)
    print 'gd_result: %.8f\n Number of iterations %d\n' % (gd_result, num_iterations)

def run_gd(alpha, data, labels):
    X = np.matrix(data)
    label_y = np.matrix(labels)
    result_weight = minimize(X, label_y)
    closed_result = calculate_j(result_weight, X, labels)
    gd_result, num_iterations = gradient_descent(alpha, X, labels, closed_result)
    print 'gd_result: %.8f\n Number of iterations %d\n' % (gd_result, num_iterations)

def gd_results():
    data, labels = generateMSE(1000, 0.01)
    run_gd_dalpha(1, data, labels)
    run_gd(1, data, labels)
    run_gd(0.1, data, labels)
    run_gd(0.01, data, labels)
    run_gd(0.001, data, labels)

gd_results()

def run():
    # ks = [1000 * n for n in range(1, 10)]
    ks = [10, 100, 1000, 10000]
    errors = [generateSets(k) for k in ks]
    training_errors = [e[0] for e in errors]
    test_errors = [e[1] for e in errors]
    print 'Training errors: ' + str(training_errors) + '\n'
    print 'Test errors: ' + str(test_errors) + '\n'
    graph_error(ks, training_errors, test_errors)
