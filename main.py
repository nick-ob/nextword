"""Main entrypoint file.
See also notebooks/main.ipynb for a jupyter notebook version.
"""
# %% [markdown]
# # NextWord

# %%
import numpy as np

from src.formatter import data_to_top_tokens, token_to_data, tokens_to_data
from src.loader import load_input
from src.model import Network

# %% [markdown]
# ## Loading input data

# %%
data: str = "demo_data.txt"
tokens_all, tokens_single = load_input(data)
(x_train, y_train), (x_test, y_test) = tokens_to_data(tokens_all, tokens_single)
print(f"Loaded data from {data}:")
print(f"Training shapes: {x_train.shape} & {y_train.shape}")
print(f"Testing shapes: {x_test.shape} & {y_test.shape}")

# %% [markdown]
# ## Initialise a network

# %%
# either create
network = Network(x_train.shape[1], 100, x_train.shape[1])

# or load
# network, histoy = Network.load("your_network")

# %% [markdown]
# ## Train the network

# %%
# skip/comment out if loading an existing network
history = network.train((x_train, y_train), 0.01, 1000, batch_size=16)

# %% [markdown]
# ## Test the network

# %%
pred, y, acc = network.test((x_test, y_test))
print(f"Testing accuracy: {acc}%")

# %% [markdown]
# ## Top next word predictions

# %%
while True:
    user_input = input("Enter a word: ")
    user_input = user_input.encode().decode("unicode_escape")
    token = token_to_data(user_input, tokens_single)
    if token is None:
        print("Word not found in tokens")
        break
    else:
        pred = network.predict(np.array([token]))
        top = data_to_top_tokens(pred, tokens_single)
        for token, prob in top.items():
            print(f"{token!r}: {prob}%")
