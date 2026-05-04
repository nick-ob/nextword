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
tokens_all, tokens_single = load_input("mini_test")
(x_train, y_train), (x_test, y_test) = tokens_to_data(tokens_all, tokens_single)
print(f"Training shapes: {x_train.shape} & {y_train.shape}")
print(f"Testing shapes: {x_test.shape} & {y_test.shape}")
