import re


def tokenize_input_text(txt):
    """
    Tokenize a user-provided string into individual words.

    Args:
        txt (str): The raw text input supplied by the user.

    Returns:
        list: A list of the individual words extracted from the input string.

    Raises:
        ValueError: If input processing fails or input type is invalid.
    """
    if not isinstance(txt, str):
        raise ValueError("txt must be a string")

    try:
        # Match sequences of word characters excluding underscores, allowing internal apostrophes.
        # [^\W_] is equivalent to Unicode letters and digits (i.e., \w without underscore).
        tokens = re.findall(r"[^\W_]+(?:'[^\W_]+)*", txt, flags=re.UNICODE)
        return tokens
    except Exception as exc:
        raise ValueError(f"Failed to tokenize input text: {exc}") from exc
