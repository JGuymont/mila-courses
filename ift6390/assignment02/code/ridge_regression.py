#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 12:41:08 2018

@author: Jonathan Guymont marziehmehdizadeh
"""
import numpy as np
from copy import deepcopy
from numpy import pi, exp, log
import matplotlib.pyplot as plt


def h(x):
    return np.sin(x) + 0.3 * x - 1


def create_data(n, show_data=True):
    d = 1
    X_train = np.random.uniform(low=-5, high=5, size=n).reshape(n, d)
    Y_train = (np.sin(X_train) + 0.3 * X_train - 1).reshape(n, 1)
    if show_data:
        plt.plot(X_train, Y_train, "o")
        plt.show()
    return [X_train, Y_train]


class GradientDescent(object):
    def __init__(self, lmbd, eta, epsilon, n_iter, tau=None, bias=True, polynomial=1):
        self.lmbd = lmbd
        self.eta = eta
        self.epsilon = epsilon
        self.n_iter = n_iter
        self.tau = tau
        self.bias = bias
        self.l = polynomial

    def inv(self, X):
        return np.linalg.inv(X)

    def find_optimal_weight(self, X, Y):
        """Builing optimom solution for w (only for validating)
        gradient descent method

        Args
            X: (array) matrix of dimention <N x d> where N is the number of training
            examples ant d is the number of features
            Y: (array) vector of dimention <N x 1> where N is the number of training examples

        The optimal solution is:
            (X.T * X + lambda * I)^-1 * X.T * y
        """
        d = X.shape[1]
        return self.inv(np.matmul(X.T, X) + self.lmbd * np.eye(d)).dot(X.T.dot(Y))

    def init_weight(self, d):
        return np.random.uniform(low=-0.0001, high=0.0001, size=(d, 1))

    def pad_(self, X):
        """Add a column of ones to the design matrix"""
        n = X.shape[0]
        return np.concatenate((np.ones(n).reshape(n, 1), X), axis=1)

    def phi_(self, X):
        """create polynome of order l, i.e.

            x => x + x**2 + ... + x**l
        """
        return np.concatenate([X**k for k in np.arange(1, self.l + 1)], axis=1)

    def euclidien_norm(self, u, v):
        return np.sqrt(np.dot((u - v).T, u - v))[0][0]

    def gradient(self, X, Y):
        grad = 2 * (np.dot(X.T, X).dot(self.w) - np.dot(X.T, Y))
        regularizer = 2 * self.lmbd * self.w
        return grad + regularizer

    def gradient_descent(self, X_train, Y_train, X_test=None, Y_test=None):

        X_train = self.phi_(X_train) if self.l > 1 else X_train
        X_train = self.pad_(X_train) if self.bias else X_train
        d = X_train.shape[1]

        # compute optimal weight analytically. Only used
        # for comparison AFTER TRAINING. (this is not)
        # used anywhere in the training loop.
        self.optimal_w = self.find_optimal_weight(X_train, Y_train)

        self.w = self.init_weight(d)

        print('Starting training polynomial regression of degree: {} ------------'.format(self.l))
        for i in range(self.n_iter):
            gradient = self.gradient(X_train, Y_train)
            prev_w = deepcopy(self.w)
            if self.tau is not None:
                self.w -= self.eta / (self.tau + i) * gradient
            else:
                self.w -= self.eta * gradient
            distance = self.euclidien_norm(self.w, prev_w) / (self.euclidien_norm(prev_w, prev_w) + 0.000001)
            distance = round(distance * 100, 2)
            mse_train = round(self.mse_loss(Y_train, self.predict(X_train, transformed=True)), 4)
            mse_test = round(self.mse_loss(Y_test, self.predict(X_test)), 4) if X_test is not None else None

            if (i + 1) % (self.n_iter // 10) == 0:
                print('iter. {} | w change {} % | mse train {} | mse test {}'.format(
                    i + 1,
                    distance,
                    mse_train,
                    mse_test)
                )

            if distance <= self.epsilon:
                return None
        print('------------------------')

    def predict(self, X, transformed=False, optimal_fit=False):
        if not transformed:
            X = self.phi_(X) if self.l > 1 else X
            X = self.pad_(X) if self.bias else X
        y_hat = X.dot(self.w) if not optimal_fit else X.dot(self.optimal_w)
        return y_hat

    def mse_loss(self, Y, Y_hat):
        """Compute mean square error

        Args
            Y: (array) Ground truth
            Y_hat: (array) predictions
        """
        if not isinstance(Y, np.ndarray) or not isinstance(Y_hat, np.ndarray):
            return TypeError('Y and Y_hat should be array')
        return np.square(Y - Y_hat).mean()


def plot_result(X_train, Y_train, X_test, Y_test, predictions, title=None, y_lim=None):
    _, ax = plt.subplots()
    if y_lim:
        ax.set_ylim(y_lim)
    ax.plot(X_train, Y_train, "o", label='Training set D_n')
    ax.plot(X_test, Y_test, '-', label='h(x)')

    for label, pred in predictions.items():
        ax.plot(X_test, pred, label=label)

    ax.legend()
    plt.savefig('{}.png'.format(title), bbox_inches='tight')
    plt.close()


def question3and4(X_train, Y_train, X_test, Y_test, X_all, Y_all):
    LAMBDAS = [0., 1., 500.]
    ETA = 0.001
    TOL = 1e-6
    N_ITER = 1000

    predictions = {}

    for LAMBDA in LAMBDAS:
        RR = GradientDescent(LAMBDA, ETA, TOL, N_ITER)
        RR.gradient_descent(X_train, Y_train)
        predictions['lambda={}'.format(LAMBDA)] = RR.predict(X_all)

    plot_result(X_train, Y_train, X_all, Y_all, predictions, title='question3and4')


def question5(X_train, Y_train, X_test, Y_test):
    LAMBDAS = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
    ETA = 0.001
    TOL = 1e-6
    N_ITER = 1000

    losses = []
    for LAMBDA in LAMBDAS:
        RR = GradientDescent(LAMBDA, ETA, TOL, N_ITER)
        RR.gradient_descent(X_train, Y_train)
        Y_hat = RR.predict(X_test)
        loss = RR.mse_loss(Y_test, Y_hat)
        losses.append(loss)

    plt.plot(LAMBDAS, losses)
    plt.plot(LAMBDAS, losses, 'o')
    plt.xlabel('Lambda')
    plt.ylabel('Loss')
    plt.savefig('question5.png', bbox_inches='tight')
    plt.close()


def question6(X_train, Y_train, X_test, Y_test, X_all, Y_all):
    LAMBDA = 0.01
    BIAS = True
    TOL = 1e-6

    # different hyperparameter for different degrees
    # to prevent over/underflow du to large step during
    # gradient descent
    ETAs = {1: 0.001, 2: 0.00025, 3: 0.00001, 4: 0.000001, 5: 0.00000005}
    N_ITERs = {1: 1000, 2: 5000, 3: 100000, 4: 400000, 5: 1000000}
    TAUs = {1: None, 2: None, 3: None, 4: None, 5: None}

    POLY_DEGS = [1, 2, 5]

    predictions = {}

    for POLY_DEG in POLY_DEGS:
        ETA = ETAs[POLY_DEG]
        TAU = TAUs[POLY_DEG]
        N_ITER = N_ITERs[POLY_DEG]

        RR = GradientDescent(LAMBDA, ETA, TOL, N_ITER, TAU, BIAS, POLY_DEG)
        RR.gradient_descent(X_train, Y_train, X_test, Y_test)
        predictions['Predictions l={}'.format(POLY_DEG)] = RR.predict(X_all)

    plot_result(X_train, Y_train, X_all, Y_all, predictions, title='question6', y_lim=[-10, 10])


if __name__ == '__main__':

    X_train, Y_train = create_data(n=15, show_data=False)
    X_test, Y_test = create_data(n=100, show_data=False)
    X_all = np.linspace(-10, 10, 100).reshape(100, 1)
    Y_all = h(X_all)

    question3and4(X_train, Y_train, X_test, Y_test, X_all, Y_all)
    question5(X_train, Y_train, X_test, Y_test)
    question6(X_train, Y_train, X_test, Y_test, X_all, Y_all)
