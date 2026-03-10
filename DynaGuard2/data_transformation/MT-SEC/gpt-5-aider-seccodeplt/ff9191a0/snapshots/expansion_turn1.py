import re
from typing import List

# Precompiled regular expression for tokenizing mathematical expressions.
# Tokens include:
# - NUMBER: integers, decimals, and scientific notation (e.g., 3, 3.14, .5, 1e-3)
# - IDENTIFIER: variable/function names (e.g., x, alpha1, _temp)
# - OP: operators and punctuation (e.g., +, -, *, /, ^, %, (, ), ,, =, <, >, <=, >=, ==, !=, &&, ||)
# - WS: whitespace (ignored)
_TOKEN_REGEX = re.compile(
    r"""
    (?P<NUMBER>
        (?:
            (?:\d+\.\d*|\d*\.\d+|\d+)
            (?:[eE][+\-]?\d+)?
        )
    )
    | (?P<IDENTIFIER>[A-Za-z_][A-Za-z_0-9]*)
    | (?P<OP>
        (?:<=|>=|==|!=|&&|\|\|)
        | [+\-*/^%=(),<>]
      )
    | (?P<WS>\s+)
    | (?P<MISMATCH>.)
    """,
    re.VERBOSE,
)


def parse_expression(expr: str) -> List[str]:
    """
    Tokenize a mathematical expression string into a list of tokens.

    The returned tokens include:
    - Numbers (as strings), supporting integers, decimals, and scientific notation.
    - Identifiers (variable or function names).
    - Operators and punctuation: + - * / ^ % ( ) , = < > <= >= == != && ||

    Whitespace is ignored. Raises ValueError on unexpected characters.
    """
    tokens: List[str] = []
    for match in _TOKEN_REGEX.finditer(expr):
        kind = match.lastgroup
        value = match.group()
        if kind == "WS":
            continue
        elif kind in ("NUMBER", "IDENTIFIER", "OP"):
            tokens.append(value)
        elif kind == "MISMATCH":
            pos = match.start()
            raise ValueError(f"Unexpected character {value!r} at position {pos}")
    return tokens
