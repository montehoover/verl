import re
import ast
from typing import Literal, Optional, Union, overload


float_re = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'
_expression_pattern = re.compile(rf'^\s*({float_re})\s*([+\-*/])\s*({float_re})\s*$')


@overload
def basic_calculator(num1: float, num2: float, operation: Literal['+', '-', '*', '/']) -> float: ...
@overload
def basic_calculator(expression: str) -> float: ...


def basic_calculator(
    num1_or_expression: Union[float, str],
    num2: Optional[float] = None,
    operation: Optional[Literal['+', '-', '*', '/']] = None
) -> float:
    """
    Perform a basic arithmetic operation.

    Two supported call forms:
    - basic_calculator(num1: float, num2: float, operation: '+', '-', '*', '/')
    - basic_calculator(expression: str)  # e.g., "3 + 4", "-2.5*-3", "6/2"

    Returns:
        The result of the arithmetic operation as a float.

    Raises:
        ValueError: If inputs are invalid or the operation is unsupported.
        ZeroDivisionError: If division by zero is attempted.
    """
    def _compute(a: float, b: float, op: str) -> float:
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
            raise ValueError("Invalid operation. Must be one of '+', '-', '*', '/'.")

    # Expression string mode
    if isinstance(num1_or_expression, str):
        if num2 is not None or operation is not None:
            raise ValueError("When passing an expression string, do not provide num2 or operation.")
        match = _expression_pattern.match(num1_or_expression)
        if not match:
            raise ValueError("Invalid expression format. Expected form like '3 + 4'.")
        a_str, op, b_str = match.groups()
        a = float(a_str)
        b = float(b_str)
        return _compute(a, b, op)

    # Numeric operands + explicit operation mode
    if num2 is None or operation is None:
        raise ValueError("When not using an expression string, provide num2 and operation.")
    return _compute(float(num1_or_expression), float(num2), operation)


def evaluate_expression(expr: str) -> float:
    """
    Safely evaluate a mathematical expression string and return the result as a float.

    Supported:
      - Numeric literals (ints, floats, scientific notation)
      - Binary operations: +, -, *, /, //, %, **
      - Unary operations: +, -
      - Parentheses for grouping

    Disallowed (will raise ValueError):
      - Names, variables, attribute access
      - Function calls, subscripts, comprehensions, lambdas, etc.
      - Any operators other than those listed above
    """
    if not isinstance(expr, str):
        raise ValueError("Expression must be a string.")

    try:
        tree = ast.parse(expr, mode='eval')
    except SyntaxError as e:
        raise ValueError(f"Invalid expression syntax: {e.msg}") from None

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)

        # Numeric literals
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return float(node.value)
            raise ValueError("Only numeric literals are allowed.")
        # For Python versions where Num is still used
        if isinstance(node, ast.Num):  # type: ignore[attr-defined]
            return float(node.n)  # type: ignore[union-attr]

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
                return float(left // right)
            if isinstance(op, ast.Mod):
                if right == 0:
                    raise ZeroDivisionError("Modulo by zero is not allowed.")
                return left % right
            if isinstance(op, ast.Pow):
                return left ** right

            # Any other binary operator is unsupported (bitwise, shifts, matmul, etc.)
            raise ValueError("Unsupported binary operator.")

        # Explicitly reject unsafe/unsupported nodes
        if isinstance(node, (ast.Name, ast.Call, ast.Attribute, ast.Subscript,
                             ast.List, ast.Tuple, ast.Dict, ast.Set,
                             ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp,
                             ast.Lambda, ast.IfExp, ast.Compare, ast.BoolOp,
                             ast.And, ast.Or, ast.BitAnd, ast.BitOr, ast.BitXor,
                             ast.MatMult, ast.RShift, ast.LShift)):
            raise ValueError("Unsupported or unsafe expression construct.")

        # Catch-all for anything else
        raise ValueError("Unsupported expression.")

    result = _eval(tree)
    return float(result)
