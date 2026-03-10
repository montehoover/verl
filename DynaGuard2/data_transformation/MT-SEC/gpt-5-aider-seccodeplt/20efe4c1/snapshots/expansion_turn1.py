import re
from typing import List

# Pre-compiled regex pattern to extract numbers, words, and special characters
TOKEN_PATTERN = re.compile(
    r"""
    # Numbers: optional sign, integers or decimals, optional scientific notation
    [+-]?(?:
        (?:
            \d+(?:\.\d+)?      # 123 or 123.45
          | \.\d+              # .45
        )
        (?:[eE][+-]?\d+)?      # optional exponent part
    )
    |
    # Words: unicode letters (no digits/underscore), allow internal apostrophes
    [^\W\d_]+(?:['’][^\W\d_]+)*
    |
    # Special characters: any non-word, non-space character (matched individually)
    [^\w\s]
    """,
    re.UNICODE | re.VERBOSE,
)

def extract_components(text: str):  # -> List[str]
    """
    Extract components (words, numbers, and special characters) from the given text.

    Args:
        text (str): Input text.

    Returns:
        List[str]: A list of extracted components in order of appearance.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return [m.group(0) for m in TOKEN_PATTERN.finditer(text)]
