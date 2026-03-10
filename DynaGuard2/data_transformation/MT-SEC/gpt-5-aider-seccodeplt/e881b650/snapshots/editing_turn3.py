import re
import ast
from typing import Literal, Optional, Union


def compute_expression(expr: str) -> float:
    """
    Safely evaluate a mathematical expression string and return the result as a float.

    Supported:
      - Numeric literals (ints, floats, scientific notation)
      - Parentheses
      - Unary +/- (e.g., -3, +4)
      - Binary operators: +, -, *, /, //, %, **

    Raises:
      ValueError: If the expression contains unsupported characters or structures,
                  or cannot be parsed safely.
      ZeroDivisionError: If a division/modulo by zero is attempted.
    """
    if not isinstance(expr, str):
        raise ValueError("Expression must be a string.")

    # Quick character whitelist to reject obviously unsafe input early.
    # Allows digits, whitespace, parentheses, decimal points, exponent markers, and arithmetic operators.
    if not re.match(r'^[\s0-9eE\+\-\*\/%\.\(\)]*$', expr):
        raise ValueError("Expression contains unsupported characters.")

    if expr.strip() == "":
        raise ValueError("Empty expression is not allowed.")

    try:
        tree = ast.parse(expr, mode='eval')
    except SyntaxError:
        raise ValueError("Invalid expression.")

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)

        # Numeric literal
        if isinstance(node, ast.Constant):
            value = node.value
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("Only numeric constants are allowed.")
            return float(value)
        # For Python versions where numbers are represented as ast.Num
        if isinstance(node, ast.Num):  # type: ignore[attr-defined]
            value = node.n  # type: ignore[attr-defined]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("Only numeric constants are allowed.")
            return float(value)

        # Unary operations: +x, -x
        if isinstance(node, ast.UnaryOp):
            operand = _eval(node.operand)
            if isinstance(node.op, ast.UAdd):
                return +operand
            if isinstance(node.op, ast.USub):
                return -operand
            raise ValueError("Unsupported unary operator.")

        # Binary operations
        if isinstance(node, ast.BinOp):
            left = _eval(node.left)
            right = _eval(node.right)
            op = node.op
            if isinstance(op, ast.Add):
                return left + right
            if isinstance(op, ast.Sub):
                return left - right
            if isinstance(op, ast.Mult):
                return left * right
            if isinstance(op, ast.Div):
                if right == 0:
                    raise ZeroDivisionError("Division by zero is not allowed.")
                return left / right
            if isinstance(op, ast.FloorDiv):
                if right == 0:
                    raise ZeroDivisionError("Division by zero is not allowed.")
                return left // right if isinstance(left, int) and isinstance(right, int) else float(left // right)
            if isinstance(op, ast.Mod):
                if right == 0:
                    raise ZeroDivisionError("Modulo by zero is not allowed.")
                return left % right
            if isinstance(op, ast.Pow):
                return left ** right
            raise ValueError("Unsupported operator.")

        # Anything else is not allowed (e.g., names, calls, subscripts, attributes, etc.)
        raise ValueError("Unsupported expression.")

    result = _eval(tree)
    return float(result)


def basic_calculate(
    num1: Union[float, str],
    num2: Optional[float] = None,
    operator: Optional[Literal['+', '-', '*', '/']] = None
) -> float:
    """
    Perform a basic arithmetic operation.

    You can call it in two ways:
      1) With numbers and an operator:
         basic_calculate(4, 5, '+')
      2) With a simple string expression containing two operands and one operator:
         basic_calculate('4 + 5')

    Args:
        num1: The first number, or a string expression like '4 + 5'.
        num2: The second number (when not passing a string expression).
        operator: One of '+', '-', '*', '/' (when not passing a string expression).

    Returns:
        The result as a float.

    Raises:
        ValueError: If the operator is not supported or the expression is invalid.
        ZeroDivisionError: If division by zero is attempted.
    """
    def _compute(a: float, op: str, b: float) -> float:
        if op == '+':
            return float(a + b)
        elif op == '-':
            return float(a - b)
        elif op == '*':
            return float(a * b)
        elif op == '/':
            if b == 0:
                raise ZeroDivisionError("Division by zero is not allowed")
            return float(a / b)
        else:
            raise ValueError("Invalid operator. Expected one of '+', '-', '*', '/'.")

    # String expression path
    if isinstance(num1, str):
        expr = num1
        # Matches optional spaces, signed floats/ints, operator, signed floats/ints
        pattern = r'^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*([+\-*/])\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*$'
        match = re.match(pattern, expr)
        if not match:
            raise ValueError('Invalid expression. Expected a format like "4 + 5".')
        left_str, op, right_str = match.groups()
        left = float(left_str)
        right = float(right_str)
        return _compute(left, op, right)

    # Traditional (num1, num2, operator) path
    if num2 is None or operator is None:
        raise ValueError("num2 and operator are required when not providing a string expression as num1.")

    return _compute(float(num1), operator, float(num2))
