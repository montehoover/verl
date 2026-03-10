import re
from typing import Tuple

_ALLOWED_EXPRESSION_RE = re.compile(r'^[0-9+\-*/().%\s^]*\Z')

_DIGIT_WORDS = {
    '0': 'zero',
    '1': 'one',
    '2': 'two',
    '3': 'three',
    '4': 'four',
    '5': 'five',
    '6': 'six',
    '7': 'seven',
    '8': 'eight',
    '9': 'nine',
}


def _transform_digits(expr: str) -> str:
    return ''.join(_DIGIT_WORDS.get(ch, ch) for ch in expr)


def calculate_result(expr: str) -> Tuple[int, str]:
    """
    Validate allowed characters and return:
    - length of the original expression
    - expression with digits replaced by their English words

    Raises ValueError if the expression contains illegal characters.
    """
    if not isinstance(expr, str):
        raise ValueError("Expression must be a string.")

    if not _ALLOWED_EXPRESSION_RE.fullmatch(expr):
        raise ValueError("Expression contains illegal characters.")

    transformed = _transform_digits(expr)
    return len(expr), transformed
