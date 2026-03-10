import re
import logging
import os


def tokenize_input_text(txt):
    """
    Tokenize a user-provided string into individual words.
    
    This function serves as a crucial component in text analysis pipelines
    where word-level processing is required. It uses regular expressions
    to extract words from the input text.
    
    Args:
        txt (str): The raw text input supplied by the user.
        
    Returns:
        list: A list comprising the individual words extracted from 
              the input string.
        
    Raises:
        ValueError: If any complications arise during the input processing,
                    such as when the input is not a string type.
    """
    # Initialize logging configuration
    log_file = os.path.join(os.getcwd(), 'tokenizer.log')
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger(__name__)
    
    try:
        # Log the input text
        logger.info(f"Input text: {repr(txt)}")
        
        # Validate that the input is a string type
        if not isinstance(txt, str):
            logger.error(f"Invalid input type: {type(txt)}")
            raise ValueError("Input must be a string")
        
        # Use regex to extract words
        # Pattern \b\w+\b matches word boundaries and word characters
        words = re.findall(r'\b\w+\b', txt)
        
        # Log the resulting words
        logger.info(f"Extracted words: {words}")
        logger.info(f"Total words found: {len(words)}")
        
        return words
        
    except Exception as e:
        # Log the error before re-raising
        logger.error(f"Error during tokenization: {str(e)}")
        # Re-raise any exception as ValueError with additional context
        raise ValueError(f"Error processing input: {str(e)}")
