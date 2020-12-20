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
import random
from typing import Callable, NoReturn


def activation_relu(x: np.ndarray, ddx: bool = False) -> np.ndarray:
    """Define the ReLU function and its derivative.

    Optionally access the derivative value by passing ddx=true.

    Args:
        x (np.ndarray): Array to evaluate
        ddx (bool, optional): Evaluate with derivative. Defaults to False.

    Returns:
        np.ndarray: Function evaluation
    """
    if ddx:
        return np.where(x > 0, 1, 0)
    else:
        return np.maximum(0, x)


def activation_sigmoid(x: np.ndarray, ddx: bool = False) -> np.ndarray:
    """Define the Sigmoid function and its derivative.

    Optionally access the derivative value by passing ddx=true.

    Args:
        x (np.ndarray): Array to evaluate
        ddx (bool, optional): Evaluate with derivative. Defaults to False.

    Returns:
        np.ndarray: Function evaluation
    """
    if ddx:
        return activation_sigmoid(x) * (1 - activation_sigmoid(x))

    else:
        return 1 / (1 + np.exp(-x))


class FeedForwardNet:
    """Feed Forward Network Class."""

    def __init__(
        self,
        act_func: Callable,
        alpha: float,
        h_layers: int,
        h_units: int,
        in_units: int,
        out_units: int,
    ):
        """FeedForwardNet Constructor.

        Args:
            act_func (Callable): Activation function. activation_relu or activation_sigmoid
            alpha (float): Learning rate
            h_layers (int): Number of hidden layers
            h_units (int): Number of units per hidden layer
            in_units (int): Number of inputs
            out_units (int): Number of outputs
        """
        self.activ_func = act_func
        self.alpha = alpha
        self.hidden_layers = h_layers
        self.hidden_units = h_units
        self.in_units = in_units
        self.out_units = out_units

        self.create_network()

    def create_network(self):
        """Create the network given constructor options.

        Follows notes at top of the file to crete a
        network as defined in the textbook.
        """
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

            # tmp_w = np.random.normal(loc=0.0, scale=0.5, size=(m, n))
            tmp_w = np.random.randn(m, n) * 0.01
            tmp_b = np.zeros((n, 1))
            self.network_layers.append([tmp_w, tmp_b])

        # Output Layer
        # tmp_w = np.random.normal(
        #    loc=0.0, scale=0.5, size=(self.hidden_units, self.out_units)
        # )
        tmp_w = np.random.randn(self.hidden_units, self.out_units) * 0.01
        tmp_b = np.zeros((self.out_units, 1))
        self.network_layers.append([tmp_w, tmp_b])

    def classify(self, train_data: np.ndarray):
        """Classify input data.

        Follows description on p 752 as per notes at top of file.

        Args:
            train_data (np.ndarray): Input data
        """
        self.forward_zi = []
        self.forward_ai = []
        self.train_data = train_data

        if len(self.train_data.shape) == 1:
            self.train_data = self.train_data.reshape((self.train_data.shape[0], 1))

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
        """Update weights using the L2 loss function and a known good value.

        Follows description on p 755 as per notes at top of file.

        Args:
            train_ans (np.ndarray): Known good output given input data
        """
        self.back_delta = []

        if len(train_ans.shape) == 1:
            train_ans = train_ans.reshape((train_ans.shape[0], 1))

        L = np.power(train_ans - self.forward_ai[-1], 2)  # L2 loss function
        dL = -2 * (train_ans - self.forward_ai[-1])  # derivative of L2
        for i in reversed(range(len(self.network_layers))):
            z_i = self.forward_zi[i]
            gp_i = self.activ_func(z_i, True)  # derivative of activation function

            if i == len(self.network_layers) - 1:
                d_i = dL * gp_i
            else:
                d_ip1 = self.back_delta[-1]  # most recent delta value
                w_ip1 = self.network_layers[i + 1][0]  # weights
                d_i = (w_ip1 @ d_ip1) * gp_i

            self.back_delta.append(d_i)

        self.back_delta.reverse()  # reverse so all arrays are syncd
        for i, layer in enumerate(self.network_layers):
            w_i = layer[0]
            d_i = self.back_delta[i]

            # input to hidden transition
            if i == 0:
                x_i = self.train_data
                w_i += (x_i @ d_i.T) * self.alpha
            else:
                a_i = self.forward_ai[i - 1]
                w_i += (a_i @ d_i.T) * self.alpha


def test_adder(net: FeedForwardNet, epochs: int = 20) -> NoReturn:
    data = [
        (np.array([[0], [1]]), np.array([[0], [1]])),
        (np.array([[0], [0]]), np.array([[0], [0]])),
        (np.array([[1], [0]]), np.array([[0], [1]])),
        (np.array([[1], [1]]), np.array([[1], [0]])),
    ]

    for i in range(epochs):
        d = random.choice(data)
        net.classify(d[0])
        net.update_weights(d[1])

    for pair in data:
        net.classify(pair[0])
        pred = net.forward_ai[-1]
        print(
            f"Input: {pair[0].flatten()}\n"
            f"Output: {pair[1].flatten()}\n"
            f"Predicted: {pred.flatten()}\n"
        )
