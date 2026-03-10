import re


def tokenize_input_text(txt):
    """
    Tokenizes a user-provided string into individual words.

    Args:
        txt: The raw text input supplied by the user (string).

    Returns:
        list: A list comprising the individual words extracted from the input string.

    Raises:
        ValueError: If the input 'txt' is not a string or if any other
                    complications arise during input processing.
    """
    if not isinstance(txt, str):
        raise ValueError("Input 'txt' must be a string.")
    
    try:
        # Use \w+ to match sequences of word characters (letters, numbers, and underscore)
        words = re.findall(r'\w+', txt)
        return words
    except Exception as e:
        # Catch any unexpected errors during regex processing, though unlikely with \w+
        # and string input already validated.
        raise ValueError(f"An error occurred during tokenization: {e}")
