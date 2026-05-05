"""File containing the logic of the softmax function, represented as a class.

Usage example:

    softmax = Softmax()
    input = np.array([1, 0, 1])
    output = softmax.forward(input)
"""
import numpy as np

class Softmax:
    """A class representing the softmax activation in a neural network.
    """
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax and forward it to the next layer.

        Args:
            x: The input from the previous layer.

        Returns:
            np.ndarray: The input for the upcoming Layer.
        """
        # normalise to avoid numerical overflow
        # this works since the softmax function is shift-invariant
        x = x - np.max(x, axis=1, keepdims=True)

        exp = np.exp(x)
        return exp / np.sum(exp, axis=1, keepdims=True)

    def backward(self, delta: np.ndarray, _) -> np.ndarray:
        """Pass on delta to the previous Layer.

        Since our softmax layer is always the last layer in our network, its delta is the
        derivative of the loss w.r.t. the output of the network, which is calculated in the Loss
        class. This is done so that this class can be called in the same way as the Layer class.

        Args:
            delta: The derivative of the loss w.r.t. the output of this layer.
            _: A placeholder to be able to use this class the same way as the Layer class.

        Returns:
            np.ndarray: Delta, the derivative of the loss w.r.t. the output of the previous layer.
        """
        # simply pass on delta recieved from the Loss class
        return delta
