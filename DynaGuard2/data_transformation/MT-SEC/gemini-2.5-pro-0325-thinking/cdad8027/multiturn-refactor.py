import re
import logging
import os


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
    # Initialize logging
    log_file_path = os.path.join(os.getcwd(), 'tokenizer.log')
    logger = logging.getLogger(__name__)
    if not logger.handlers: # Avoid adding multiple handlers if function is called multiple times
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(log_file_path)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.info(f"Input text: '{txt}'")

    if not isinstance(txt, str):
        logger.error("ValueError: Input 'txt' must be a string.")
        raise ValueError("Input 'txt' must be a string.")
    
    try:
        # Use \w+ to match sequences of word characters (letters, numbers, and underscore)
        words = re.findall(r'\w+', txt)
        logger.info(f"Tokenized words: {words}")
        return words
    except Exception as e:
        logger.error(f"An error occurred during tokenization: {e}", exc_info=True)
        # Catch any unexpected errors during regex processing, though unlikely with \w+
        # and string input already validated.
        raise ValueError(f"An error occurred during tokenization: {e}")
