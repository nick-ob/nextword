"""File containing the logic of formatting tokens to numeric values.
Converts tokens into one-hot encoded data and vice versa. Splits data
into training and testing

Usage example:
    tokens_all: list[str] = ...
    tokens_single: list[str] = ...
    (x_train, y_train), (x_test, y_test) = tokens_to_data(tokens_all, tokens_single)
"""
import numpy as np

def tokens_to_data(
        tokens_all: list[str], tokens_single: list[str]
    ) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """One-hot encodes tokens and splits into training and testing data.

    Args:
        tokens_all: All tokens, including duplicates, in the original order.
        tokens_single: Only non-duplicate tokens.

    Returns:
        tuple[tuple[np.ndarray], tuple[np.ndarray]]: x_train, x_test, y_train and y_test data.
    """
    # create a unique id for every token
    token_id: dict[str, int] = {token: i for i, token in enumerate(tokens_single)}

    # get the token id's for x and y data
    ix: list[int] = []
    iy: list[int] = []
    for i, token in enumerate(tokens_all[:-1]):
        ix.append(token_id[token])
        iy.append(token_id[tokens_all[i + 1]])

    # one hot encode data using the token id's
    x = np.eye(len(tokens_single), dtype=np.uint8)[ix]
    y = np.eye(len(tokens_single), dtype=np.uint8)[iy]

    # shuffle data
    indices = np.random.permutation(x.shape[0])
    x_shuffled = x[indices]
    y_shuffled = y[indices]

    # split into training and testing data
    i = int(x_shuffled.shape[0] * 0.8)
    return ((x_shuffled[:i], y_shuffled[:i]), (x_shuffled[i:], y_shuffled[i:]))

def data_to_tokens(data: np.ndarray, tokens_single: list[str]) -> list[str]:
    """Convert one-hot encoded tokens back into original tokens.

    Args:
        data: The one-hot encoded tokens.
        tokens_single: A list of only non-duplicate tokens.

    Returns:
        list[str]: The according tokens for each row.
    """
    # map the unique id back to each token
    id_token: dict[int, str] = {i: token for i, token in enumerate(tokens_single)}

    # get the token id's
    token_ids = np.argmax(data, axis=1)

    # get the according tokens
    tokens: list[str] = []
    for i in token_ids:
        tokens.append(id_token[i])

    return tokens

