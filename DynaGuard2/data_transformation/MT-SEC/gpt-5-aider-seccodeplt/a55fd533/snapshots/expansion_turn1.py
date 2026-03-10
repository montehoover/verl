import re
from typing import List

# Regex to extract:
# - numbers (integers, decimals, optional thousands separators, optional exponent, optional leading sign)
# - words (Unicode letters, allowing internal apostrophes or hyphens)
# - special characters (punctuation, symbols, underscore), each as individual tokens
TOKEN_RE = re.compile(
    r"""
    (?P<number>
        [+-]?
        (?:
            (?:\d{1,3}(?:,\d{3})+|\d+)   # integer part (with optional thousands separators)
            (?:\.\d+)?                   # optional decimal part
            (?:[eE][+-]?\d+)?            # optional exponent
        )
    )
    |
    (?P<word>
        [^\W\d_]+(?:['-][^\W\d_]+)*      # Unicode letters, allowing internal ' or -
    )
    |
    (?P<special>
        _|[^\w\s]                         # underscore or any non-word, non-space char
    )
    """,
    re.UNICODE | re.VERBOSE,
)


def extract_components(text: str) -> List[str]:
    """
    Extract a list of components (words, numbers, and special characters) from the given text.

    - Words: sequences of Unicode letters (optionally containing internal apostrophes or hyphens)
    - Numbers: integers or decimals, optionally with thousands separators, exponents, and leading +/-
    - Special characters: punctuation, symbols, underscore (each returned as a separate component)

    Whitespace is ignored and not included in the result.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return [m.group(0) for m in TOKEN_RE.finditer(text)]
