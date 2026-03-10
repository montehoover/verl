import re
import operator
import ast
from typing import Any, Dict

# Allowed binary operations mapping
_ALLOWED_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

# Allowed unary operations mapping
_ALLOWED_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_]\w*$")
_NUMERIC_TYPES = (int, float)


def execute_calculation_string(calc_expr: str, var_dict: Dict[str, Any]) -> str:
    """
    Evaluate a mathematical expression string with variable substitution in a safe manner.

    Args:
        calc_expr (str): A string representing a mathematical expression.
        var_dict (dict): A dictionary where keys are variable names and values are numbers.

    Returns:
        str: The result of the simplified expression as a string.

    Raises:
        ValueError: If the input expression is invalid, contains disallowed constructs,
                    references unknown variables, or the computation fails.
    """
    if not isinstance(calc_expr, str):
        raise ValueError("calc_expr must be a string.")
    if not isinstance(var_dict, dict):
        raise ValueError("var_dict must be a dict of variable names to numbers.")

    # Validate provided variables (names must be valid identifiers; values numeric)
    for k, v in var_dict.items():
        if not isinstance(k, str) or not _IDENTIFIER_RE.match(k):
            raise ValueError(f"Invalid variable name: {k!r}")
        if isinstance(v, bool) or not isinstance(v, _NUMERIC_TYPES):
            raise ValueError(f"Variable {k!r} must be a number (int or float).")

    try:
        parsed = ast.parse(calc_expr, mode="eval")
    except SyntaxError as exc:
        raise ValueError("Invalid expression syntax.") from exc

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)

        # Numeric literals
        if isinstance(node, ast.Constant):
            val = node.value
            if isinstance(val, bool) or not isinstance(val, _NUMERIC_TYPES):
                raise ValueError("Only numeric literals are allowed.")
            return float(val) if isinstance(val, float) else val  # keep ints as ints
        # For Python <3.8 compatibility, though Constant should cover it
        if hasattr(ast, "Num") and isinstance(node, getattr(ast, "Num")):
            val = node.n
            if isinstance(val, bool) or not isinstance(val, _NUMERIC_TYPES):
                raise ValueError("Only numeric literals are allowed.")
            return float(val) if isinstance(val, float) else val

        # Variable names
        if isinstance(node, ast.Name):
            name = node.id
            if not _IDENTIFIER_RE.match(name):
                raise ValueError(f"Invalid identifier: {name!r}")
            if name not in var_dict:
                raise ValueError(f"Unknown variable: {name!r}")
            val = var_dict[name]
            # After earlier validation, val must be int or float (not bool)
            return float(val) if isinstance(val, float) else val

        # Unary operations (+x, -x)
        if isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in _ALLOWED_UNARY_OPS:
                raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
            operand = _eval(node.operand)
            if isinstance(operand, bool) or not isinstance(operand, _NUMERIC_TYPES):
                raise ValueError("Unary operations require numeric operands.")
            return _ALLOWED_UNARY_OPS[op_type](operand)

        # Binary operations (x + y, x * y, etc.)
        if isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in _ALLOWED_BIN_OPS:
                raise ValueError(f"Unsupported binary operator: {op_type.__name__}")
            left = _eval(node.left)
            right = _eval(node.right)
            if isinstance(left, bool) or isinstance(right, bool):
                raise ValueError("Operands must be numeric.")
            if not isinstance(left, _NUMERIC_TYPES) or not isinstance(right, _NUMERIC_TYPES):
                raise ValueError("Operands must be numeric.")
            try:
                result = _ALLOWED_BIN_OPS[op_type](left, right)
            except ZeroDivisionError as exc:
                raise ValueError("Division by zero.") from exc
            return result

        # Disallow all other node types (calls, attributes, subscripts, lambdas, etc.)
        raise ValueError("Expression contains unsupported constructs.")

    try:
        result = _eval(parsed)
    except ValueError:
        # Re-raise our own semantic errors unchanged
        raise
    except Exception as exc:
        # Wrap all other issues as ValueError
        raise ValueError("Failed to compute the expression.") from exc

    # Convert result to a neatly formatted string.
    # Normalize -0.0 to 0
    if isinstance(result, float) and result == 0.0:
        result = 0.0

    if isinstance(result, float) and result.is_integer():
        return str(int(result))
    return str(result)
