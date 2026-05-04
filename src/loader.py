"""File containing the logic of loading the input text data.
Loads the data from a .txt file and cleans it for usage.

Usage example:
    data = load_input("tiny_shakespeare")
"""
import os
import re

def load_input(file: str) -> tuple[list[str], list[str]]:
    """Load the input data using a provided filename.
    Gets the input file from the data directory and tokenises it using a
    regex key.

    file: The input file to load (file must be in the data directory).

    Returns:
        tuple[list[str], list[str]]: The list of the 'tokens' in their order
        and the list of 'tokens' without duplicates.
    """
    # get the data directory path
    src_dir: str = os.path.dirname(os.path.abspath(__file__))
    root_dir: str = os.path.dirname(src_dir)
    data_dir: str = os.path.join(root_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    # make sure the input exists
    if not file.endswith(".txt"):
        file = file + ".txt"
    existing_data: set[str] = set(os.listdir(data_dir))
    if file not in existing_data:
        raise FileNotFoundError(f"Data ({file}) does not exist in {data_dir}.")

    # try loading the input
    data_file = os.path.join(data_dir, file)
    try:
        with open(data_file, "r", encoding="utf-8-sig") as input:
                raw = input.read()
    except Exception as e:
        raise RuntimeError(f"Failed to load {file}") from e

    if not raw:
        raise ValueError(f"Empty dataset: {file}")

    # 'tokenise' data using regex
    # \w+(?:'\w+) matches all regular words
    # [^\w\s] matches single characters that arent a whitespace or a word
    # \n to include newlines as tokens
    rx = r"\w+(?:'\w+)?|[^\w\s]|\n"
    tokens_all = re.findall(rx, raw)

    # remove duplicates
    tokens_single = list(set(tokens_all))

    return (tokens_all, tokens_single)

if __name__ == "__main__":
     tokens_all, tokens = load_input("tiny_shakespeare.txt")
