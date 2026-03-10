import ast
from typing import Any


def parse_python_code(source: str) -> ast.AST:
    """
    Parse a string of Python code and return its AST if syntactically correct.

    Args:
        source: A string containing Python source code.

    Returns:
        An ast.AST object representing the parsed code.

    Raises:
        ValueError: If the code has syntax issues.
        TypeError: If 'source' is not a string.
    """
    if not isinstance(source, str):
        raise TypeError("source must be a string of Python code")

    try:
        return ast.parse(source, mode="exec")
    except SyntaxError as e:
        msg = e.msg or "invalid syntax"
        parts = []
        if e.lineno is not None:
            parts.append(f"line {e.lineno}")
        if e.offset is not None:
            parts.append(f"column {e.offset}")
        location = f" ({', '.join(parts)})" if parts else ""
        snippet = e.text.strip() if e.text else None
        if snippet:
            raise ValueError(f"{msg}{location}: {snippet}") from e
        raise ValueError(f"{msg}{location}") from e
