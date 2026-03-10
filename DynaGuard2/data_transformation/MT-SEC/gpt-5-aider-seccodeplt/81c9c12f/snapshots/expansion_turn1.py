import re
from typing import List

_TOKEN_REGEX = re.compile(
    r"""
    (?P<WS>\s+)                                                     # whitespace
    |(?P<NUMBER>(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)           # numbers incl. decimals and scientific notation
    |(?P<OP>[+\-*/^])                                               # operators
    |(?P<LPAREN>\()                                                 # left parenthesis
    |(?P<RPAREN>\))                                                 # right parenthesis
    """,
    re.VERBOSE,
)

def parse_math_expression(expression: str) -> List[str]:
    """
    Tokenize a mathematical expression into recognized elements.

    Recognized elements:
    - Numbers: integers, decimals, and scientific notation (e.g., 42, 3.14, .5, 1e-3)
    - Operators: + - * / ^
    - Parentheses: ( )

    Whitespace is ignored.
    Raises ValueError on any unrecognized character.

    :param expression: The input expression string.
    :return: A list of tokens as strings in the order they appear.
    """
    if expression is None:
        return []

    tokens: List[str] = []
    i = 0
    n = len(expression)

    while i < n:
        m = _TOKEN_REGEX.match(expression, i)
        if not m:
            # Identify offending character for helpful error message
            ch = expression[i]
            raise ValueError(f"Unrecognized character at position {i}: {repr(ch)}")
        kind = m.lastgroup
        text = m.group(kind)
        if kind == "NUMBER":
            tokens.append(text)
        elif kind in ("OP", "LPAREN", "RPAREN"):
            tokens.append(text)
        # kind == "WS" -> skip
        i = m.end()

    return tokens
