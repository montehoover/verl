import re
import ast
from typing import Optional, Union

def basic_calculator(num1: Union[float, str], num2: Optional[float] = None, operator: Optional[str] = None) -> float:
    """
    Perform a basic arithmetic operation.

    Usage:
        - basic_calculator(5.0, 7.0, '+')
        - basic_calculator("5 + 7")

    Args:
        num1: First operand as float, or an expression string like '5 + 7'.
        num2: Second operand (optional if using expression string).
        operator: One of '+', '-', '*', '/' (optional if using expression string).

    Returns:
        The result as a float.

    Raises:
        ValueError: If operator or expression is invalid.
        ZeroDivisionError: If division by zero is attempted.
        TypeError: If arguments do not match expected formats.
    """
    # Expression mode
    if isinstance(num1, str) and num2 is None and operator is None:
        expr = num1.strip()
        pattern = r'^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*([+\-*/])\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*$'
        match = re.match(pattern, expr)
        if not match:
            raise ValueError("Invalid expression format. Expected a simple binary expression like '5 + 7'.")
        a_str, op, b_str = match.groups()
        a = float(a_str)
        b = float(b_str)
        return basic_calculator(a, b, op)

    # Direct args mode
    if num2 is None or operator is None:
        raise TypeError("Expected (num1: float, num2: float, operator: str) or a single expression string like '5 + 7'.")

    if operator not in {"+", "-", "*", "/"}:
        raise ValueError(f"Invalid operator: {operator}. Expected one of '+', '-', '*', '/'.")

    a = float(num1)
    b = float(num2)

    if operator == "/":
        if b == 0:
            raise ZeroDivisionError("Division by zero.")
        result = a / b
    elif operator == "+":
        result = a + b
    elif operator == "-":
        result = a - b
    else:  # operator == "*"
        result = a * b

    return float(result)


def secure_math_eval(exp_str: str) -> float:
    """
    Safely evaluate a simple mathematical expression string.

    Only the following are allowed:
      - Numbers (ints/floats, including scientific notation)
      - Binary operators: +, -, *, /
      - Unary operators: +, -
      - Parentheses

    Any other syntax (names, function calls, attribute access, bitwise ops, etc.)
    will raise a ValueError. Division by zero raises ZeroDivisionError.

    Args:
        exp_str: The expression string to evaluate.

    Returns:
        The evaluated numerical result as a float.

    Raises:
        ValueError: If the expression contains invalid or unsafe constructs.
        ZeroDivisionError: If division by zero is attempted.
    """
    if not isinstance(exp_str, str):
        raise ValueError("Expression must be a string.")

    try:
        tree = ast.parse(exp_str, mode="eval")
    except SyntaxError:
        raise ValueError("Invalid expression.")

    def eval_node(node) -> float:
        if isinstance(node, ast.Expression):
            return eval_node(node.body)

        if isinstance(node, ast.BinOp):
            left = eval_node(node.left)
            right = eval_node(node.right)

            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                if right == 0:
                    raise ZeroDivisionError("Division by zero.")
                return left / right

            # Disallow all other binary operators (e.g., **, //, %, bitwise ops)
            raise ValueError("Unsafe or disallowed operator in expression.")

        if isinstance(node, ast.UnaryOp):
            operand = eval_node(node.operand)
            if isinstance(node.op, ast.UAdd):
                return +operand
            if isinstance(node.op, ast.USub):
                return -operand
            raise ValueError("Unsafe or disallowed unary operator in expression.")

        if isinstance(node, ast.Constant):  # Python 3.8+
            val = node.value
            # Disallow booleans and non-numeric constants
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise ValueError("Invalid literal in expression.")
            return float(val)

        if hasattr(ast, "Num") and isinstance(node, ast.Num):  # Back-compat
            return float(node.n)

        # Explicitly reject all other node types
        raise ValueError("Unsafe expression construct detected.")

    result = eval_node(tree)
    return float(result)
