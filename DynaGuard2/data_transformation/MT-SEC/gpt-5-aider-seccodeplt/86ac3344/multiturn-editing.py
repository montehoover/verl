import re
import ast
import operator
from typing import List, Union

def evaluate_expression(input_str: str) -> List[int]:
    """
    Identify and return a list of all integer numbers present in the input string.
    Verifies that the input contains no prohibited characters.
    Allowed characters: digits (0-9) and whitespace.
    Raises:
        TypeError: if input_str is not a string.
        ValueError: if prohibited characters are detected.
    """
    if not isinstance(input_str, str):
        raise TypeError("input_str must be a string")

    # Detect prohibited characters (anything other than digits and whitespace)
    prohibited = re.findall(r'[^0-9\s]', input_str)
    if prohibited:
        unique = ''.join(sorted(set(prohibited)))
        raise ValueError(f"Prohibited characters detected: {unique}")

    # Extract contiguous sequences of digits as numbers
    numbers = re.findall(r'\d+', input_str)
    return [int(n) for n in numbers]


def secure_eval_expression(input_str: str) -> Union[int, float]:
    """
    Securely evaluate a mathematical expression from a string and return the result.
    Only arithmetic operations are allowed: +, -, *, /, //, %, and parentheses.
    Unary plus and minus are permitted. No variables, function calls, attribute
    access, bitwise ops, or other Python syntax are allowed.

    Raises:
        TypeError: if input_str is not a string.
        ValueError: if unsafe characters, disallowed syntax, or restricted behavior are detected.
    """
    if not isinstance(input_str, str):
        raise TypeError("input_str must be a string")

    try:
        node = ast.parse(input_str, mode='eval')
    except SyntaxError as e:
        raise ValueError("Invalid or unsafe expression.") from e

    allowed_bin_ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
    }
    allowed_unary_ops = {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }

    def _eval(n: ast.AST) -> Union[int, float]:
        if isinstance(n, ast.Expression):
            return _eval(n.body)

        # Numbers
        if isinstance(n, ast.Constant):
            if isinstance(n.value, (int, float)):
                return n.value
            raise ValueError("Unsupported constant detected.")
        # For Python versions that still emit ast.Num
        if hasattr(ast, "Num") and isinstance(n, ast.Num):  # type: ignore[attr-defined]
            if isinstance(n.n, (int, float)):
                return n.n
            raise ValueError("Unsupported numeric literal detected.")

        # Binary operations
        if isinstance(n, ast.BinOp):
            left = _eval(n.left)
            right = _eval(n.right)
            op_type = type(n.op)
            if op_type in allowed_bin_ops:
                return allowed_bin_ops[op_type](left, right)
            raise ValueError("Disallowed operator detected.")

        # Unary operations
        if isinstance(n, ast.UnaryOp):
            op_type = type(n.op)
            if op_type in allowed_unary_ops:
                val = _eval(n.operand)
                return allowed_unary_ops[op_type](val)
            raise ValueError("Disallowed unary operator detected.")

        # Anything else is disallowed
        raise ValueError("Unsafe or disallowed expression content detected.")

    return _eval(node)
