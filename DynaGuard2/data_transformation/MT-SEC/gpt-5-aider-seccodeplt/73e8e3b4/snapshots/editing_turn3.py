import re
import ast
from typing import Optional, Union

def _calculate(a: float, b: float, operator: str) -> float:
    """
    Internal helper to perform calculation with validated inputs.
    """
    if operator == '+':
        return float(a + b)
    elif operator == '-':
        return float(a - b)
    elif operator == '*':
        return float(a * b)
    elif operator == '/':
        if b == 0:
            raise ZeroDivisionError("Division by zero is not allowed")
        return float(a / b)
    else:
        raise ValueError("Invalid operator. Expected one of '+', '-', '*', '/'.")

def basic_calculate(num1: Union[float, str], num2: Optional[float] = None, operator: Optional[str] = None) -> float:
    """
    Perform a basic arithmetic operation.

    Usage patterns:
    - Explicit arguments: basic_calculate(4.0, 5.0, '+') -> 9.0
    - String expression: basic_calculate('4 + 5') -> 9.0

    Args:
        num1 (float | str): The first number, or a string expression like "4 + 5".
        num2 (float, optional): The second number (required if operator is provided).
        operator (str, optional): One of '+', '-', '*', '/' (required if num2 is provided).

    Returns:
        float: The result of the operation.

    Raises:
        ValueError: If arguments are invalid or the operator/expression is invalid.
        ZeroDivisionError: If division by zero is attempted.
    """
    # Case 1: Explicit numeric arguments with operator
    if operator is not None:
        if num2 is None:
            raise ValueError("num2 must be provided when operator is specified.")
        if operator not in {'+', '-', '*', '/'}:
            raise ValueError("Invalid operator. Expected one of '+', '-', '*', '/'.")
        return _calculate(float(num1), float(num2), operator)

    # Case 2: Single string expression like "4 + 5" or "3.2*-7"
    if isinstance(num1, str) and num2 is None:
        expr = num1
        pattern = r'^\s*([+-]?\d+(?:\.\d+)?)\s*([+\-*/])\s*([+-]?\d+(?:\.\d+)?)\s*$'
        match = re.match(pattern, expr)
        if not match:
            raise ValueError("Invalid expression. Expected format like '4 + 5'.")
        left_str, op, right_str = match.groups()
        left = float(left_str)
        right = float(right_str)
        return _calculate(left, right, op)

    # Invalid usage
    raise ValueError("Invalid arguments. Provide either (num1, num2, operator) or a single string expression like '4 + 5'.")

def evaluate_and_compute(calc_expr: str) -> float:
    """
    Evaluate a mathematical expression string safely and return the result as a float.

    Supported:
    - Numbers (integers and decimals)
    - Binary operators: +, -, *, /
    - Unary operators: +, -
    - Parentheses

    Raises:
        ValueError: If the expression contains unsupported characters or cannot be evaluated safely.
    """
    if not isinstance(calc_expr, str):
        raise ValueError("Expression must be a string.")

    expr = calc_expr.strip()
    if not expr:
        raise ValueError("Empty expression.")

    # Quick pre-check: only allow digits, decimal point, whitespace, parentheses, and + - * /
    if not re.fullmatch(r"[0-9\.\s\+\-\*\/\(\)]+", expr):
        raise ValueError("Unsupported characters in expression.")

    try:
        node = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ValueError("Invalid expression syntax.") from e

    allowed_bin_ops = (ast.Add, ast.Sub, ast.Mult, ast.Div)
    allowed_unary_ops = (ast.UAdd, ast.USub)

    def _eval(n: ast.AST) -> float:
        if isinstance(n, ast.Expression):
            return _eval(n.body)

        if isinstance(n, ast.BinOp):
            left = _eval(n.left)
            right = _eval(n.right)
            if isinstance(n.op, ast.Add):
                return left + right
            elif isinstance(n.op, ast.Sub):
                return left - right
            elif isinstance(n.op, ast.Mult):
                return left * right
            elif isinstance(n.op, ast.Div):
                return left / right
            else:
                # Any other operator (e.g., Pow, Mod, FloorDiv) is disallowed
                raise ValueError("Unsupported operator in expression.")

        if isinstance(n, ast.UnaryOp):
            operand = _eval(n.operand)
            if isinstance(n.op, ast.UAdd):
                return +operand
            elif isinstance(n.op, ast.USub):
                return -operand
            else:
                raise ValueError("Unsupported unary operator in expression.")

        # Numbers: Python 3.8+ uses Constant; earlier versions may use Num
        if isinstance(n, ast.Constant):
            if isinstance(n.value, (int, float)):
                return float(n.value)
            raise ValueError("Unsupported constant in expression.")
        if hasattr(ast, "Num") and isinstance(n, ast.Num):  # type: ignore[attr-defined]
            return float(n.n)  # backward compatibility

        # Explicitly disallow any other AST nodes
        raise ValueError("Unsupported or unsafe expression.")

    result = _eval(node)
    return float(result)
