import re

def tokenize_input_text(txt):
    """
    Tokenizes a user-provided string into individual words.

    Args:
        txt: The raw text input supplied by the user.

    Returns:
        list: A list comprising the individual words extracted from the input string.

    Raises:
        ValueError: If any complications arise during input processing,
                    or if the input is not a string.
    """
    if not isinstance(txt, str):
        raise ValueError("Input 'txt' must be a string.")

    try:
        # \w+ matches one or more word characters (letters, numbers, and underscore)
        words = re.findall(r'\w+', txt)
        return words
    except Exception as e:
        # Catch any unexpected errors during regex processing
        raise ValueError(f"Error tokenizing input text: {e}")
