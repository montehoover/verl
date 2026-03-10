import logging
import re

# Set up a module-level logger. We attach a NullHandler to avoid "No handler"
# warnings if the application using this module has not configured logging.
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Precompile the regular expression pattern once at import time for efficiency.
# \w matches Unicode word characters (letters, digits, and underscore).
_WORD_PATTERN = re.compile(r"\w+", flags=re.UNICODE)


def parse_text_input(provided_input):
    """
    Parse a user-provided string and return a list of extracted words.

    This function uses a Unicode-aware regular expression to identify "word"
    tokens in the input text. Word characters include letters, digits, and
    underscore, as defined by the \\w character class.

    Logging:
        - Debug: Records the original input and the extracted words.
        - Error: Records exceptions encountered during parsing.

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

    # Log the incoming raw input for traceability.
    logger.debug("parse_text_input called with input: %r", provided_input)

    try:
        # Find all sequences of word characters in the input.
        # Example: "Hello, world!" -> ["Hello", "world"]
        extracted_words = _WORD_PATTERN.findall(provided_input)

        # Log the parsing result for observability.
        logger.debug("parse_text_input extracted words: %r", extracted_words)

        return extracted_words
    except Exception as err:
        # Log the exception with traceback and normalize the error type.
        logger.exception("Failed to parse input")
        raise ValueError(f"Failed to process input: {err}") from err
