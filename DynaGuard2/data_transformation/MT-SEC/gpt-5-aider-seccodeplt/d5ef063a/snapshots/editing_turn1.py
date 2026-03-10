import re
import operator
from typing import Union

Number = Union[int, float]


def execute_operation(operation: str) -> Number:
    """
    Execute a simple arithmetic operation expressed as a string, like '5 + 3'.

    Supported operators: +, -, *, /
    Operands may be integers or floats (including scientific notation).
    Returns int for +, -, * when both operands are integers; returns float for / or when operands are floats.
    """
    if not isinstance(operation, str):
        raise TypeError("operation must be a string")

    pattern = r'^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*([+\-*/])\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*$'
    m = re.match(pattern, operation)
    if not m:
        raise ValueError("Invalid operation format. Expected a format like '5 + 3' with operators +, -, *, /.")

    a_str, op, b_str = m.groups()

    ops = {
        '+': operator.add,
        '-': operator.sub,
        '*': operator.mul,
        '/': operator.truediv,
    }

    if op not in ops:
        raise ValueError(f"Unsupported operator: {op}")

    is_int_a = re.match(r'^[+-]?\d+$', a_str) is not None
    is_int_b = re.match(r'^[+-]?\d+$', b_str) is not None

    if is_int_a:
        a: Number = int(a_str)
    else:
        a = float(a_str)
    if is_int_b:
        b: Number = int(b_str)
    else:
        b = float(b_str)

    result = ops[op](a, b)

    if op in ('+', '-', '*') and is_int_a and is_int_b:
        if isinstance(result, float) and result.is_integer():
            return int(result)
        return result

    return result
