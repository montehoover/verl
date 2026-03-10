import re

def tokenize_input_text(txt):
    """
    Tokenize a user-provided string into individual words.

    Args:
        txt (str): The raw text input supplied by the user.

    Returns:
        list: A list comprising the individual words extracted from the input string.

    Raises:
        ValueError: If the input is not a string or if any complications arise during processing.
    """
    if not isinstance(txt, str):
        raise ValueError("Input must be a string.")

    try:
        # Unicode-aware word extraction:
        # - Match sequences of word characters excluding underscores
        # - Allow internal apostrophes (e.g., don't, l'État)
        tokens = re.findall(r"[^\W_]+(?:'[^\W_]+)*", txt, flags=re.UNICODE)
        return tokens
    except Exception as exc:
        raise ValueError(f"Error tokenizing input: {exc}") from exc
