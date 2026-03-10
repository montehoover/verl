import re
from typing import List, Dict, Union

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

# Identifier pattern used to detect variable tokens (not numbers or operators)
_IDENTIFIER_RE = re.compile(r'^[A-Za-z_][A-Za-z_0-9]*$')


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


def substitute_variables(tokens: List[str], values: Dict[str, Union[int, float, str]]) -> List[str]:
    """
    Substitute identifier tokens with their corresponding values.

    - Only tokens that are identifiers and are present as keys in 'values' are replaced.
    - Numeric values (int/float) are converted to tokens using parse_expression(str(value)),
      ensuring signs are handled consistently (e.g., -5 -> ['-', '5']).
    - String values are tokenized with parse_expression, allowing substitution with either
      a single token (e.g., '3.14') or multiple tokens (e.g., '2*x').

    Returns a new list of tokens with substitutions applied.
    """
    substituted: List[str] = []
    for tok in tokens:
        if _IDENTIFIER_RE.match(tok) and tok in values:
            val = values[tok]
            if isinstance(val, (int, float)):
                repl_tokens = parse_expression(str(val))
            elif isinstance(val, str):
                repl_tokens = parse_expression(val)
            else:
                raise TypeError(f"Unsupported value type for variable '{tok}': {type(val).__name__}")
            substituted.extend(repl_tokens)
        else:
            substituted.append(tok)
    return substituted
