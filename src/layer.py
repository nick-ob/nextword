"""File containing the logic of a single layer, represented as a class.

Usage example:

    layer = Layer(3, 1)
    input = np.array([1, 0, 1])
    output = layer.forward(input)
"""
import numpy as np

class Layer:
    """A class representing a layer in the neural network.

    Attributes:
        __w: Weights of the layer.
        __x: Caches the input of the layer (needed for backpropagation).
    """
    def __init__(self, in_nodes: int, out_nodes: int) -> None:
        """Initialises instances based on the amount of inputs it recieves and outputs it passes on.

        Args:
            in_nodes: The amount of nodes this layer recieves from the previous layer.
            out_nodes: the amount of nodes of this layer / that this layers passes to the next.
        """
        # initialise weights
        self.__w: np.ndarray = np.random.randn(in_nodes, out_nodes) * np.sqrt(2 / in_nodes)

        self.__x: np.ndarray = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute the output values and forward them to the next layer.

        Args:
            x: The input recieved from the previous layer.

        Returns:
            np.ndarray: The input the upcoming layer should recieve.
        """
        # store the input (needed for gradient computation)
        self.__x = x

        return x @ self.__w

    def backward(self, delta: np.ndarray, learning_rate: float) -> np.ndarray:
        """Backward pass to the previous layer.
        Compute the gradient for this layer, nudge own weights
        and give new delta to the previous layer.

        Args:
            delta: The derivative of the loss w.r.t. the output of this layer.
            learning_rate: The learning rate for gradient descent.

        Returns:
            np.ndarray: Delta, the derivative of the loss w.r.t. the ouput of the previous layer.
        """
        # derivative of the loss w.r.t. the weights
        grad = self.__x.T @ delta

        # derivative of the loss w.r.t. the ouput of the previous layer
        delta = delta @ self.__w.T

        # adjust the weights using the gradient
        self.__w = self.__w - learning_rate * grad

        return delta

    def get_params(self) -> np.ndarray:
        """Get the layers parameters.

        Returns:
            np.ndarray: The weights matrix.
        """
        return self.__w

    def set_params(self, weights: np.ndarray) -> None:
        """Set the layers parameters.

        Args:
            weights: The weights matrix to set.
        """
        self.__w = weights
