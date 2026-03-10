import ast
import operator
import re
from typing import Any, Dict, Union

Number = Union[int, float]


def evaluate_math_expression(formula: str, vars: Dict[str, Number]) -> str:
    """
    Evaluate a mathematical expression string with optional variables and return the result as a string.

    Args:
        formula (str): A string representing a mathematical formula potentially containing variables.
        vars (dict): A mapping of variable names to their numeric values for evaluation.

    Returns:
        str: The result after computing the expression, returned in string format.

    Raises:
        ValueError: If an error occurs due to an invalid expression or unsuccessful processing.
    """
    if not isinstance(formula, str):
        raise ValueError("formula must be a string")
    if not isinstance(vars, dict):
        raise ValueError("vars must be a dictionary")

    # Optional: support '^' as exponent operator by translating to Python '**'
    # This helps align with common math notation expectations.
    # We avoid touching '**' that already exist by doing a simple replace (idempotent for '**').
    normalized_formula = formula.replace("^", "**")

    try:
        parsed = ast.parse(normalized_formula, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"Invalid expression syntax: {e}") from None

    bin_ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
    }

    unary_ops = {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }

    def _ensure_number(value: Any, context: str) -> Number:
        if isinstance(value, bool):
            # Avoid treating booleans as integers
            raise ValueError(f"Boolean value not allowed in {context}")
        if not isinstance(value, (int, float)):
            raise ValueError(f"Non-numeric value in {context}: {value!r}")
        return value

    def _eval(node: ast.AST) -> Number:
        if isinstance(node, ast.Expression):
            return _eval(node.body)

        if isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in bin_ops:
                raise ValueError(f"Operator not allowed: {op_type.__name__}")
            left = _eval(node.left)
            right = _eval(node.right)
            left = _ensure_number(left, "binary operation (left operand)")
            right = _ensure_number(right, "binary operation (right operand)")
            try:
                return bin_ops[op_type](left, right)
            except ZeroDivisionError:
                raise ValueError("Division by zero") from None

        if isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in unary_ops:
                raise ValueError(f"Unary operator not allowed: {op_type.__name__}")
            operand = _eval(node.operand)
            operand = _ensure_number(operand, "unary operation")
            return unary_ops[op_type](operand)

        # Support numeric literals
        if isinstance(node, ast.Constant):
            value = node.value
            value = _ensure_number(value, "constant")
            return value

        # For Python versions where numbers may appear as ast.Num
        if isinstance(node, ast.Num):  # type: ignore[attr-defined]
            value = node.n  # type: ignore[attr-defined]
            value = _ensure_number(value, "numeric literal")
            return value

        # Variable names
        if isinstance(node, ast.Name):
            name = node.id
            if name not in vars:
                raise ValueError(f"Undefined variable: {name}")
            value = vars[name]
            value = _ensure_number(value, f"variable '{name}'")
            return value

        # Allow parenthesis via Tuple with load context is not correct; parentheses are handled implicitly.
        # Disallow any other nodes (calls, attributes, subscripts, comprehensions, etc.)
        raise ValueError(f"Unsupported expression element: {type(node).__name__}")

    try:
        result = _eval(parsed)
    except ValueError:
        # Propagate our own semantic errors directly
        raise
    except Exception as e:
        # Any other unexpected error is considered invalid processing
        raise ValueError(f"Failed to evaluate expression: {e}") from None

    # Convert result to string; simplify floats that are integral
    if isinstance(result, float):
        if result.is_integer():
            return str(int(result))
        return str(result)
    return str(result)
