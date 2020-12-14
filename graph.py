"""graph.py satisfies Part 3 of the writeup.

Sourced from textbook chapter 21 and https://towardsdatascience.com/neural-net-from-scratch-using-numpy-71a31f6e3675

Notes:
    From page 752
        a_j = g_j(w'x)
            a_j = output of unit j
            g_j = nonlinear activation function
            w' = vector of weights leading into unit j, including bias unit 0 -> w_0,j
            x = vector of inputs to unit j, including bias unit 0 -> 1
    
    From site:
        Dimensions:
            n_x = number of inputs
            n_h = nodes in hidden layer
            n_y = nodes of output layer

            a_1 = n_h, 1
            w_1 = n_h, n_x
            b_1 = n_h, 1

            a_end = n_y, 1
            w_end = n_y, n_h
            b_end = n_y, 1

    We define:
        a_i: outputs of layer i
        w_i: input weights to layer i
        b_i: input bias vector to layer i

        alpha: learning rate modification
"""

import numpy as np


class FeedForwardNet:
    def __init__(self, alpha, act_func, h_layers, in_matrix, num_h, num_in, num_out):
        self.alpha = alpha
        self.activ_func = act_func
        self.hidden_layers = h_layers
        self.input_matrix = in_matrix
        self.num_inputs = num_in
        self.num_hidden = num_h
        self.num_output = num_out

        self.network_layers = []
        self.create_network()

    def create_network(self):
        # Input Layer
        self.network_layers.append((self.input_matrix, None))

        # Hidden Layers
        n_x = self.input_matrix.shape[0]
        for _ in range(self.hidden_layers):
            tmp_w = np.random.randn(self.num_hidden, n_x) * self.alpha
            tmp_b = np.zeros((self.num_hidden, 1))
            self.network_layers.append((tmp_w, tmp_b))

        # Output Layer
        tmp_w = np.random.randn(self.num_hidden, n_x) * self.alpha
        tmp_b = np.zeros((self.num_hidden, 1))
        self.network_layers.append((tmp_w, tmp_b))

    def calc_forward(self):
        self.forward = []

        for i in range(1, len(self.network_layers)):
            vec_b = self.network_layers[i][1]
            vec_w = self.network_layers[i][0]
            vec_x = self.network_layers[i - 1][0]

            a_i = self.activ_func(np.dot(vec_w, vec_x) + vec_b)
            self.network_layers.append(a_i)
