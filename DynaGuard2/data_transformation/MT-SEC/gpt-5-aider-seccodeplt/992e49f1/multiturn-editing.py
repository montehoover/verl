import ast
import re
from typing import Optional, Union

def basic_calculate(num1: Union[float, str], num2: Optional[float] = None, operator: Optional[str] = None) -> float:
    """
    Perform basic arithmetic on two numbers or parse a simple expression string.

    Usage:
        basic_calculate(5, 3, '+') -> 8.0
        basic_calculate("5 + 3") -> 8.0

    Args:
        num1: First operand as float, or a string expression like '5 + 3'.
        num2: Second operand (optional when using string form).
        operator: One of '+', '-', '*', '/' (optional when using string form).

    Returns:
        The result as a float.

    Raises:
        ValueError: If operator or expression is invalid.
        ZeroDivisionError: If division by zero is attempted.
    """
    def compute(a: float, b: float, op: str) -> float:
        if op == '+':
            return a + b
        elif op == '-':
            return a - b
        elif op == '*':
            return a * b
        elif op == '/':
            if b == 0.0:
                raise ZeroDivisionError("division by zero")
            return a / b
        else:
            raise ValueError("Invalid operator. Must be one of '+', '-', '*', '/'.")

    # String expression mode: e.g., "5 + 3"
    if isinstance(num1, str) and num2 is None and operator is None:
        expr = num1.strip()
        match = re.match(r'^\s*([+-]?\d+(?:\.\d+)?)\s*([+\-*/])\s*([+-]?\d+(?:\.\d+)?)\s*$', expr)
        if not match:
            raise ValueError("Invalid expression format. Expected '<number> <operator> <number>'.")
        left_str, op, right_str = match.groups()
        a = float(left_str)
        b = float(right_str)
        return compute(a, b, op)

    # Traditional mode: num1, num2, operator
    if num2 is None or operator is None:
        raise ValueError("Missing arguments. Provide either a string expression or num1, num2, and operator.")

    a = float(num1)
    b = float(num2)
    return compute(a, b, operator)


def safe_execute_operation(operation: str) -> float:
    """
    Safely evaluate a mathematical expression string using a restricted AST.

    Allowed:
      - Numbers (int, float), including scientific notation and underscores
      - Binary ops: +, -, *, /
      - Unary ops: +, -
      - Parentheses and whitespace

    Args:
        operation: The expression to evaluate (e.g., "5 + 3 * (2 - 1)")

    Returns:
        The evaluated result as a float.

    Raises:
        ValueError: If the expression contains unsafe/invalid characters or cannot be evaluated.
    """
    if not isinstance(operation, str) or not operation.strip():
        raise ValueError("Operation must be a non-empty string.")

    # Quick character-level screening to reject obviously unsafe inputs early.
    allowed_chars = set("0123456789.+-*/() \t_eE")
    if any(ch not in allowed_chars for ch in operation):
        raise ValueError("Expression contains unsafe or invalid characters.")

    try:
        node = ast.parse(operation, mode='eval')
    except SyntaxError as e:
        raise ValueError(f"Invalid expression: {e}") from None

    def is_allowed_ast(n: ast.AST) -> bool:
        if isinstance(n, ast.Expression):
            return is_allowed_ast(n.body)
        if isinstance(n, ast.BinOp):
            return (
                isinstance(n.op, (ast.Add, ast.Sub, ast.Mult, ast.Div))
                and is_allowed_ast(n.left)
                and is_allowed_ast(n.right)
            )
        if isinstance(n, ast.UnaryOp):
            return isinstance(n.op, (ast.UAdd, ast.USub)) and is_allowed_ast(n.operand)
        # Python 3.8+: numbers are Constants
        if isinstance(n, ast.Constant):
            val = n.value
            return (isinstance(val, (int, float)) and not isinstance(val, bool))
        # Older Python: numbers are Num
        if hasattr(ast, "Num") and isinstance(n, ast.Num):
            val = n.n
            return (isinstance(val, (int, float)) and not isinstance(val, bool))
        # Disallow everything else
        return False

    if not is_allowed_ast(node):
        raise ValueError("Expression contains unsupported operations or constructs.")

    def eval_node(n: ast.AST) -> float:
        if isinstance(n, ast.Expression):
            return eval_node(n.body)
        if isinstance(n, ast.BinOp):
            left = eval_node(n.left)
            right = eval_node(n.right)
            if isinstance(n.op, ast.Add):
                return float(left + right)
            if isinstance(n.op, ast.Sub):
                return float(left - right)
            if isinstance(n.op, ast.Mult):
                return float(left * right)
            if isinstance(n.op, ast.Div):
                if right == 0:
                    raise ValueError("Division by zero.")
                return float(left / right)
            raise ValueError("Unsupported binary operator.")
        if isinstance(n, ast.UnaryOp):
            val = eval_node(n.operand)
            if isinstance(n.op, ast.UAdd):
                return float(+val)
            if isinstance(n.op, ast.USub):
                return float(-val)
            raise ValueError("Unsupported unary operator.")
        if isinstance(n, ast.Constant):
            val = n.value
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise ValueError("Only int and float literals are allowed.")
            return float(val)
        if hasattr(ast, "Num") and isinstance(n, ast.Num):
            val = n.n
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise ValueError("Only int and float literals are allowed.")
            return float(val)
        raise ValueError("Unsupported expression node.")

    try:
        return eval_node(node)
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Failed to evaluate expression: {e}") from None
