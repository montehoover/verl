import re

# Precompile the regular expression pattern once at import time for efficiency.
# \w matches Unicode word characters (letters, digits, and underscore).
_WORD_PATTERN = re.compile(r"\w+", flags=re.UNICODE)


def parse_text_input(provided_input):
    """
    Parse a user-provided string and return a list of extracted words.

    This function uses a Unicode-aware regular expression to identify "word"
    tokens in the input text. Word characters include letters, digits, and
    underscore, as defined by the \\w character class.

    Args:
        provided_input (str): The raw text input from the user.

    Returns:
        list: A list of extracted word tokens, in the order they appear.

    Raises:
        ValueError: If provided_input is not a string or if processing fails.
    """
    # Validate the type of the input early to fail fast and clearly.
    if not isinstance(provided_input, str):
        raise ValueError("provided_input must be a string")

    try:
        # Find all sequences of word characters in the input.
        # Example: "Hello, world!" -> ["Hello", "world"]
        extracted_words = _WORD_PATTERN.findall(provided_input)
    except re.error as err:
        # Raise a ValueError to conform to the function's error contract.
        raise ValueError(f"Regex processing failed: {err}") from err
    except Exception as err:
        # Catch-all to ensure only ValueError escapes this function.
        raise ValueError(f"Failed to process input: {err}") from err

    # Return the list of extracted words (may be empty if no matches found).
    return extracted_words
