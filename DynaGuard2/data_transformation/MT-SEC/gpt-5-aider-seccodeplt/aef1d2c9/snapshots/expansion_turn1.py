import re
from typing import List

def extract_components(text: str) -> List[str]:
    """
    Extract components from text using regex:
    - Words (Unicode letters, allowing internal apostrophes, e.g., don't)
    - Numbers (integers and decimals, optional sign)
    - Special characters (single non-word, non-whitespace characters)

    Returns a list of components in the order they appear.
    """
    pattern = re.compile(
        r"(?:[+-]?(?:\d+(?:\.\d+)?|\.\d+))"   # numbers: integers or decimals with optional sign
        r"|(?:[^\W\d_]+(?:'[^\W\d_]+)*)"      # words: letters only (Unicode), allow internal apostrophes
        r"|(?:[^\w\s])",                      # special characters: punctuation/symbols (single char)
        re.UNICODE
    )
    return pattern.findall(text)
