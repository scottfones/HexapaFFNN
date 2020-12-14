"""graph.py satisfies Part 3 of the writeup.

Sourced from textbook chapter 21

Notes:
    From page 752
        a_j = g_j(w.Tx)
            a_j = output of unit j
            g_j = nonlinear activation function
            w.T = vector of weights leading into unit j, including bias unit 0 -> w_0,j
            x = vector of inputs to unit j, including bias unit 0 -> 1

    We define:
        a_i: outputs of layer i
        w_i: input weights to layer i
        b_i: input bias vector to layer i

        alpha: learning rate modification
"""

import numpy as np


def activation_sigmoid(x):
    return 1 / (1 + np.exp(-x))

def activation_relu(x):
    return np.maximum(0,x)


class FeedForwardNet:
    def __init__(self, act_func, alpha, h_layers, h_units, in_units, out_units):
        self.activ_func = act_func
        self.alpha = alpha
        self.hidden_layers = h_layers
        self.hidden_units = h_units
        self.in_units = in_units
        self.out_units = out_units

        self.create_network()

    def create_network(self):
        self.network_layers = []

        # Hidden Layers
        for i in range(self.hidden_layers):
            if i > 0: 
                if type(self.network_layers[i-1]) is tuple:
                    m = self.network_layers[i-1][0].shape[1]
                else:
                    m = self.network_layers[i-1].shape[1]
            else:
                m = self.in_units
            n = self.hidden_units
            
            tmp_w = np.random.randn(m, n) * self.alpha
            tmp_b = np.zeros((n, 1))
            self.network_layers.append((tmp_w, tmp_b))

        # Output Layer
        tmp_w = np.random.randn(self.hidden_units, self.out_units) * self.alpha
        tmp_b = np.zeros((self.out_units, 1))
        self.network_layers.append((tmp_w, tmp_b))

    def classify(self, train_data):
        self.forward = []

        # Input to Hidden Transition
        b_0 = self.network_layers[0][1]
        w_0 = self.network_layers[0][0]
        x = train_data

        a_i = self.activ_func(w_0.T @ x + b_0)
        self.forward.append(a_i)

        for i in range(1, len(self.network_layers)):
            b_i = self.network_layers[i][1]
            w_i = self.network_layers[i][0]
            a_im1 = self.forward[i - 1]

            a_i = self.activ_func(w_i.T @ a_im1 + b_i)
            self.forward.append(a_i)

    def update_weights(self, train_ans):
        