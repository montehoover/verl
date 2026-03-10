from typing import Literal, Optional, Union, overload
import re
import ast

Operator = Literal['+', '-', '*', '/']


@overload
def basic_arithmetic(expression: str) -> float: ...
@overload
def basic_arithmetic(num1: float, num2: float, operator: Operator) -> float: ...


def basic_arithmetic(
    num1: Union[float, str],
    num2: Optional[float] = None,
    operator: Optional[Operator] = None
) -> float:
    """
    Perform a basic arithmetic operation.

    Usage 1 (explicit args):
        basic_arithmetic(5.0, 7.0, '+') -> 12.0

    Usage 2 (string expression):
        basic_arithmetic('5 + 7') -> 12.0

    Args:
        num1: First operand as float, or a string expression like '5 + 7'.
        num2: Second operand (required if num1 is float).
        operator: One of '+', '-', '*', '/' (required if num1 is float).

    Returns:
        The result as a float.

    Raises:
        ValueError: If inputs are invalid or operator is unsupported.
        ZeroDivisionError: If division by zero is attempted.
    """
    def _compute(a: float, b: float, op: Operator) -> float:
        if op == '+':
            return float(a + b)
        elif op == '-':
            return float(a - b)
        elif op == '*':
            return float(a * b)
        elif op == '/':
            if b == 0:
                raise ZeroDivisionError("Division by zero is not allowed.")
            return float(a / b)
        else:
            raise ValueError(f"Unsupported operator: {op!r}. Expected one of '+', '-', '*', '/'.")

    # String expression mode
    if isinstance(num1, str) and num2 is None and operator is None:
        expr = num1
        match = re.match(
            r'^\s*([+-]?(?:\d+(?:\.\d+)?|\.\d+))\s*([+\-*/])\s*([+-]?(?:\d+(?:\.\d+)?|\.\d+))\s*$',
            expr
        )
        if not match:
            raise ValueError("Invalid expression format. Expected format like '5 + 7'.")
        a_str, op, b_str = match.groups()
        a = float(a_str)
        b = float(b_str)
        return _compute(a, b, op)  # type: ignore[arg-type]

    # Explicit args mode
    if isinstance(num1, (int, float)) and isinstance(num2, (int, float)) and operator in ('+', '-', '*', '/'):
        a = float(num1)
        b = float(num2)
        op = operator  # type: ignore[assignment]
        return _compute(a, b, op)  # type: ignore[arg-type]

    raise ValueError(
        "Invalid arguments. Provide either (num1: float, num2: float, operator: '+', '-', '*', '/') "
        "or a single expression string like '5 + 7'."
    )


def perform_safe_math(expression: str) -> float:
    """
    Safely evaluate a mathematical expression containing only numbers,
    parentheses, and the operators +, -, *, / with optional unary + or -.

    Args:
        expression: A string like "5 + (7 / 2) - -3".

    Returns:
        The evaluated result as a float.

    Raises:
        ValueError: If the expression is unsafe, malformed, too complex,
                    or contains unsupported elements.
        ZeroDivisionError: If division by zero is attempted.
    """
    if not isinstance(expression, str):
        raise ValueError("Expression must be a string.")
    expr = expression.strip()
    if not expr:
        raise ValueError("Expression cannot be empty.")
    if len(expr) > 1000:
        raise ValueError("Expression too long.")

    try:
        tree = ast.parse(expr, mode='eval')
    except SyntaxError as e:
        raise ValueError("Invalid expression format.") from e

    # Limit expression complexity to mitigate resource exhaustion
    if sum(1 for _ in ast.walk(tree)) > 200:
        raise ValueError("Expression too complex.")

    allowed_binops = (ast.Add, ast.Sub, ast.Mult, ast.Div)
    allowed_unops = (ast.UAdd, ast.USub)

    def eval_node(node) -> float:
        if isinstance(node, ast.Expression):
            return eval_node(node.body)
        elif isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return float(node.value)
            raise ValueError("Only numeric literals are allowed.")
        elif hasattr(ast, "Num") and isinstance(node, getattr(ast, "Num")):  # backward compatibility
            return float(node.n)
        elif isinstance(node, ast.BinOp):
            left = eval_node(node.left)
            right = eval_node(node.right)
            op = node.op
            if isinstance(op, ast.Add):
                return float(left + right)
            elif isinstance(op, ast.Sub):
                return float(left - right)
            elif isinstance(op, ast.Mult):
                return float(left * right)
            elif isinstance(op, ast.Div):
                if right == 0:
                    raise ZeroDivisionError("Division by zero is not allowed.")
                return float(left / right)
            else:
                raise ValueError("Unsupported operator.")
        elif isinstance(node, ast.UnaryOp):
            operand = eval_node(node.operand)
            if isinstance(node.op, ast.UAdd):
                return float(+operand)
            elif isinstance(node.op, ast.USub):
                return float(-operand)
            else:
                raise ValueError("Unsupported unary operator.")
        # Explicitly reject all other node types
        elif isinstance(node, (ast.Call, ast.Name, ast.Attribute, ast.Subscript, ast.Compare,
                               ast.BoolOp, ast.Lambda, ast.List, ast.Tuple, ast.Dict, ast.Set,
                               ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp,
                               ast.IfExp, ast.AugAssign, ast.Assign, ast.Module)):
            raise ValueError("Unsupported or unsafe expression element.")
        else:
            # For any AST nodes not accounted for above
            raise ValueError("Unsupported or unsafe expression element.")

    return float(eval_node(tree))
