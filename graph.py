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

    From page 755
        delta_5 = L'*g'(in_5) => gradient wrt w_3,5 = delta_5 * a_3
            "Perceived Error" for output layer?

        delta_3 = delta_5*w_3,5*g'(in_3) => gradient wrt w_1,3 = delta_3*x1
            "Perceived Error" for hidden layer following input vector

    We define:
"""
import numpy as np
from typing import Callable


def activation_relu(x, ddx=False):
    if ddx:
        return np.where(x > 0, 1, 0)
    else:
        return np.maximum(0, x)


def activation_sigmoid(x, ddx=False):
    if ddx:
        return activation_sigmoid(x) * (1 - activation_sigmoid(x))

    else:
        return 1 / (1 + np.exp(-x))


class FeedForwardNet:
    def __init__(
        self,
        act_func: Callable,
        alpha: float,
        h_layers: int,
        h_units: int,
        in_units: int,
        out_units: int,
    ):
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
                if type(self.network_layers[i - 1]) is list:
                    m = self.network_layers[i - 1][0].shape[1]
                else:
                    m = self.network_layers[i - 1].shape[1]
            else:
                m = self.in_units
            n = self.hidden_units

            tmp_w = np.random.normal(loc=0.0, scale=0.5, size=(m, n))
            tmp_b = np.zeros((n, 1))
            self.network_layers.append([tmp_w, tmp_b])

        # Output Layer
        tmp_w = np.random.normal(
            loc=0.0, scale=0.5, size=(self.hidden_units, self.out_units)
        )
        tmp_b = np.zeros((self.out_units, 1))
        self.network_layers.append([tmp_w, tmp_b])

    def classify(self, train_data: np.ndarray):
        self.forward_zi = []
        self.forward_ai = []
        self.train_data = train_data

        # Input to Hidden Transition
        b_0 = self.network_layers[0][1]
        w_0 = self.network_layers[0][0]
        x = train_data

        z_i = w_0.T @ x + b_0
        a_i = self.activ_func(z_i)
        self.forward_zi.append(z_i)
        self.forward_ai.append(a_i)

        # Work Remaining Layers
        for i in range(1, len(self.network_layers)):
            b_i = self.network_layers[i][1]
            w_i = self.network_layers[i][0]
            a_im1 = self.forward_ai[i - 1]

            z_i = w_i.T @ a_im1 + b_i
            a_i = self.activ_func(z_i)
            self.forward_zi.append(z_i)
            self.forward_ai.append(a_i)

    def update_weights(self, train_ans: np.ndarray):
        self.back_delta = []

        L = np.power(train_ans - self.forward_ai[-1], 2)
        dL = -2 * (train_ans - self.forward_ai[-1])
        for i in reversed(range(len(self.network_layers))):
            # print(f'Working {i}')
            z_i = self.forward_zi[i]
            gp_i = self.activ_func(z_i, True)  # g prime

            if i == len(self.network_layers) - 1:
                # print(f'dL: {dL.shape}')
                # print(f'gp_i: {gp_i.shape}')
                d_i = dL * gp_i
            else:
                d_ip1 = self.back_delta[-1]
                w_ip1 = self.network_layers[i + 1][0]
                # print(f'd_ip1: {d_ip1.shape}')
                # print(f'w_ip1: {w_ip1.shape}')
                # print(f'gp_i: {gp_i.shape}')
                d_i = (w_ip1 @ d_ip1) * gp_i

            self.back_delta.append(d_i)

        self.back_delta.reverse()
        for i, layer in enumerate(self.network_layers):
            w_i = layer[0]
            d_i = self.back_delta[i]
            # print(f'working: {i}')
            if i == 0:
                x_i = self.train_data
                # print(f'd_i: {d_i.shape}')
                # print(f'x_i: {x_i.shape}')
                w_i += (x_i @ d_i.T) * self.alpha
            else:
                a_i = self.forward_ai[i - 1]
                # print(f'd_i: {d_i.shape}')
                # print(f'a_i: {a_i.shape}')
                # print(f'w_i: {w_i.shape}')
                w_i += (a_i @ d_i.T) * self.alpha
