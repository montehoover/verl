import re
import logging
from typing import Tuple, Optional, Match

# Configure basic logging
# In a real application, this might be configured at a higher level
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def _execute_regex_match(text: str, pattern: str) -> Optional[Match[str]]:
    """
    Executes a regular expression full match.

    Args:
        text: The input string to match against.
        pattern: The regular expression pattern.

    Returns:
        A match object if the entire string matches the pattern, otherwise None.
    """
    return re.fullmatch(pattern, text)

def match_strings(text: str) -> Optional[Tuple[str, str]]:
    """
    Matches a string of the format 'some_text(other_text)' and captures
    the text outside and inside the parentheses.

    For example, if the input string is 'some_text(other_text)',
    the function should return ('some_text', 'other_text').

    The function also handles cases where the part outside or inside
    the parentheses (or both) might be empty:
    - 'text()' returns ('text', '')
    - '(text)' returns ('', 'text')
    - '()' returns ('', '')

    Args:
        text: The input string to be matched. The string must conform
              entirely to the pattern `outside_text(inside_text)`.

    Returns:
        A tuple containing two strings:
        1. The text outside (before) the first opening parenthesis.
        2. The text inside the first pair of parentheses.
        Returns None if the input string does not match this pattern.
    """
    logger.debug(f"match_strings called with text: '{text}'")
    # This regex pattern aims to capture two groups:
    # 1. `([^(]*)`: Captures any characters (zero or more) that are not an opening parenthesis.
    #    This is the text "outside" or before the parentheses.
    # 2. `\(`: Matches the literal opening parenthesis.
    # 3. `([^)]*)`: Captures any characters (zero or more) that are not a closing parenthesis.
    #    This is the text "inside" the parentheses.
    # 4. `\)`: Matches the literal closing parenthesis.
    # The `^` and `$` anchors ensure that the entire string must match this pattern.
    regex_pattern = r'^([^(]*)\(([^)]*)\)$'
    
    match_obj = _execute_regex_match(text, regex_pattern)
    
    if not match_obj:
        logger.debug(f"No match found for text: '{text}'")
        return None
    
    # groups() will return a tuple of all captured groups.
    # In this case, ('text_outside', 'text_inside')
    result = match_obj.groups() # type: ignore
    logger.debug(f"Match found for '{text}': {result}")
    return result
