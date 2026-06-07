"""File containing the logic of the network, represented as a class.

Usage example:

    network = Network(3, 2, 2)
    x_train = np.array([1, 0, 1])
    y_train = np.array([[1]])
    network.train((x_train, y_train), 0.01, 100)
"""

import os

import numpy as np

from src.activations import Softmax
from src.layer import Layer
from src.loss import CCE, accuracy


class Network:
    """A class representing the actual network.

    Pieces together the individual parts (layers, activations, loss) to get a running network.

    Attributes:
        __arch: The architecture of the network, meaning the layers and their node amounts.
        __layers: The layers of the network, including activation layers.
        __history: A variable to cache the training history. Needed to save in the save function.
    """

    def __init__(self, *nodes: int) -> None:
        """Initialises instances using node amounts.

        Args:
            nodes: All node counts. Each node count represents the amount of nodes of one layer.
        """
        # make sure inputs are valid
        if len(nodes) < 2:
            raise ValueError("""Network must have at least 2 layers (input + output).
                             Make sure at least 2 arguments are being passed.""")
        if nodes[-1] < 2:
            raise ValueError("""Last layer must have at least 2 nodes,
                             representing the probability of all possilbe outputs.""")
        for node in nodes:
            if type(node) is not int or node < 1:
                raise ValueError(
                    """All layer node amounts must be positive integers."""
                )

        self.__arch: tuple[int] = nodes
        self.__layers = self.__init_layers(nodes)
        self.__history: dict[str, list] = {}

    def __init_layers(self, nodes: tuple[int]) -> list:
        """Takes a tuple containing layer nodes and creates a list of actual layers.

        Args:
            nodes: Tuple of integers representing the layers and their node counts.

        Returns:
            list: List of layers representing the network.
        """
        layers: list = []

        # iterate every node count except the last
        for i, in_nodes in enumerate(nodes[:-1]):
            # get node count of next layer and add a Layer
            out_nodes = nodes[i + 1]
            layers.append(Layer(in_nodes, out_nodes))

        layers.append(Softmax())
        return layers

    def __forward_feed(self, x: np.ndarray) -> np.ndarray:
        """Forward propagate through all of the layers.

        Args:
            x: The input data to feed through the network.

        Returns:
            np.ndarray: The output of the last layer, i.e. the predicted labels.
        """
        # feed values through all of the layers,
        for layer in self.__layers:
            x = layer.forward(x)

        return x

    def __backpropagate(self, delta: np.ndarray, learning_rate: float) -> None:
        """Backpropagate through all network layers.

        Args:
            delta: The derivative of the loss w.r.t. the final output.
            learning_rate: The learning rate for gradient descent.
        """
        # backpropagate through  all layers, passing on delta to previous layers
        for layer in reversed(self.__layers):
            delta = layer.backward(delta, learning_rate)
            # since there is no use for delta, it is simply discarded

    def __shuffle_data(
        self, x: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Shuffle x and y data row-wise so that they still align.

        Args:
            x: The input of the training data.
            y: The labels of the training data.

        Returns:
            tuple: The input and output of the shuffled data.
        """
        # randomly shuffle indices
        indices = np.random.permutation(x.shape[0])

        # apply shuffled indices to data
        return (x[indices], y[indices])

    def __display_progress(
        self, idx: int, e: int, batches: int, epochs: int, acc: float
    ) -> None:
        """Display the progress of training.

        Args:
            idx: The current batch index.
            e: The current epoch.
            batches: The amount of batches.
            epochs: The amount of epochs.
            acc: The accuracy of the batch.
        """
        bar_len = 50
        filled = int(bar_len * (idx + 1) / batches)
        progress_bar = "#" * filled + " " * (bar_len - filled)

        print(
            f"\rEpoch: {e}/{epochs} | "
            f"[{progress_bar}] Batch: {idx}/{batches} - Batch accuracy: {acc}% ",
            end="",
            flush=True,
        )

    def train(
        self,
        data: tuple[np.ndarray, np.ndarray],
        learning_rate: float,
        epochs: int,
        batch_size: int | None = None,
    ) -> dict[str, list]:
        """Train the network on provided training data.

        Args:
            data: The input and labels for the training data.
            learning_rate: The learning rate used for gradient descent.
            epochs: The amount of training epochs.
            batch_size: The size of the batches used to train. Defaults to None.
            If None, then the full dataset is used.

        Returns:
            dict[str, list]: A history of the cost and accuracy of the model.
        """
        x, y = data

        # validate inputs
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if type(epochs) is not int or epochs <= 0:
            raise ValueError("epochs must be a positive integer")
        if batch_size is not None and (type(batch_size) is not int or batch_size <= 0):
            raise ValueError("batch_size must be a positive integer")
        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y must have the same number of batches")

        # use full data if no batch size is given
        if batch_size is None:
            batch_size = x.shape[0]

        total_batches: int = x.shape[0] // batch_size

        cce = CCE()
        history: dict[str, list] = {"cost": [], "accuracy": []}

        print("Learning...")
        # training loop
        for e in range(1, epochs + 1):
            # shuffle input and output
            x_shuffled, y_shuffled = self.__shuffle_data(x, y)

            # go through all batches for the data
            i: int = 0
            for idx, j in enumerate(
                range(batch_size, x.shape[0] + batch_size, batch_size)
            ):
                # get batch & prepare variables for the next
                x_b, y_b = x_shuffled[i:j], y_shuffled[i:j]
                i = j

                # forward feed and backpropagate through the network
                y_pred = self.__forward_feed(x_b)
                self.__backpropagate(cce.delta(y_pred, y_b), learning_rate)

                # calculate stats
                loss: float = cce.cost(y_pred, y_b)
                acc: float = accuracy(y_pred, y_b)
                history["cost"].append(loss)
                history["accuracy"].append(acc)

                self.__display_progress(idx, e, total_batches, epochs, acc)

        # compute and display accuracy with the full data
        y_pred = self.__forward_feed(x)
        print(f"\nLearning completed! Ending accuracy: {accuracy(y_pred, y)}%")

        # cache and return history
        self.__history = history
        return history

    def test(
        self, data: tuple[np.ndarray, np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Test the network with the provided testing data.

        Args:
            data: The input and labels of the testing data.

        Returns:
            tuple[np.ndarray, np.ndarray, float]: Predicted labels, actual labels & accuracy.
        """
        x, y = data

        pred = self.__forward_feed(x)
        return (pred, y, accuracy(pred, y))

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict the output for the given input.

        Args:
            x: The input data.

        Returns:
            np.ndarray: The predicted output.
        """
        return self.__forward_feed(x)

    def save(self, name: str) -> None:
        """Save all of the network weights.

        Args:
            name: The name under which it should be saved.
        """
        # folder of this file
        src_dir = os.path.dirname(os.path.abspath(__file__))
        # go back one step (into the project root), then into the data folder
        root_dir = os.path.dirname(src_dir)
        save_dir = os.path.join(root_dir, "data", "networks")
        os.makedirs(save_dir, exist_ok=True)
        save_file = os.path.join(save_dir, name)

        params: dict[str, np.ndarray] = {}
        # iterate all Layer classes
        for i, layer in enumerate(self.__layers[:-1]):
            weights = layer.get_params()
            params[f"w_{i}"] = weights

        # save architecture and all parameters
        np.savez(
            save_file,
            allow_pickle=False,
            arch=self.__arch,
            cost_history=self.__history["cost"],
            acc_history=self.__history["accuracy"],
            **params,
        )

    @classmethod
    def load(cls, name: str) -> tuple["Network", dict[str, list]]:
        """Load network from a saved file stat.

        Args:
            name: The name under which the network was saved.
        """
        # folder of this file
        src_dir = os.path.dirname(os.path.abspath(__file__))
        # go back one step (into the project root), then into the data folder
        root_dir = os.path.dirname(src_dir)
        nw_dir = os.path.join(root_dir, "data", "networks")
        saved_dir = os.path.join(nw_dir, f"{name}.npz")

        # make sure file exists
        existing_files = os.listdir(nw_dir)
        if f"{name}.npz" not in existing_files:
            raise FileNotFoundError(
                f"Network data does not exist in {nw_dir}. Missing: {f'{name}.npz'}\n"
                f"Try checking which files exist, perhaps the wrong name was entered."
            )

        # restore architecture and parameters
        with np.load(saved_dir) as data:
            arch = data["arch"]
            # convert to integers to pass validation checks
            arch = tuple(int(n) for n in data["arch"])
            costs = data["cost_history"]
            acc = data["acc_history"]

            params: dict[str, np.ndarray] = {}

            for key in data.files:
                if key != "arch":
                    params[key] = data[key]

        # create network and load weights into it
        network = cls(*arch)
        for i, layer in enumerate(network.__layers[:-1]):
            w = params[f"w_{i}"]
            layer.set_params(w)

        return (network, {"cost": costs, "accuracy": acc})
