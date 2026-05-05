"""Main entrypoint file.
See also notebooks/main.ipynb for a jupyter notebook version.
"""
# %% [markdown]
# # NextWord

# %% [markdown]
# ## Loading input data

# %%
from src.loader import load_input
from src.formatter import tokens_to_data

# %%
data: str = "mini_test.txt"
tokens_all, tokens_single = load_input(data)
(x_train, y_train), (x_test, y_test) = tokens_to_data(tokens_all, tokens_single)
print(f"Loaded data from {data}:")
print(f"Training shapes: {x_train.shape} & {y_train.shape}")
print(f"Testing shapes: {x_test.shape} & {y_test.shape}")

# %% [markdown]
# ## Initialise a network

# %%
from src.model import Network

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
