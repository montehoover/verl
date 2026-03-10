import re
from typing import List

_TOKEN_PATTERN = re.compile(r"""
    (?:
        (?: (?:\d+(?:\.\d*)?|\.\d+) (?:[eE][+-]?\d+)? )   # numbers: int/float/scientific
      | (?: [^\W\d_]+ )                                   # words: letters (Unicode), no digits/underscore
      | (?: _ | [^\w\s] )                                 # special: underscore and other non-word, non-space chars
    )
""", re.VERBOSE)

def extract_components(text: str) -> List[str]:
    """
    Extract words, numbers, and special characters from the given text.

    - Words: sequences of letters (Unicode), excluding digits and underscores.
    - Numbers: integers, decimals, and scientific notation (e.g., 3, 3.14, .5, 1e10, 2.5E-3).
    - Special characters: individual punctuation/symbol characters, including underscore.
    Whitespace is ignored.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a str")
    if not text:
        return []
    return _TOKEN_PATTERN.findall(text)
