import ast
import operator
import math
from typing import Any


# Supported operators mapped to their safe implementations
_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def calculate_expression(expr_input: str) -> str:
    """
    Safely evaluate a basic arithmetic expression provided as a string.

    Allowed:
      - Numbers (ints and floats)
      - Binary operators: +, -, *, /, //, %, **
      - Unary operators: +, -
      - Parentheses (implicitly via AST structure)

    Returns:
      - The computation result as a string on success.
      - 'Computation Error!' on any invalid/suspicious input or evaluation failure.
    """
    try:
        if not isinstance(expr_input, str):
            return 'Computation Error!'

        s = expr_input.strip()
        if not s:
            return 'Computation Error!'

        # Basic guard against excessively large inputs
        if len(s) > 1000:
            return 'Computation Error!'

        # Parse expression to AST
        try:
            node = ast.parse(s, mode='eval')
        except SyntaxError:
            return 'Computation Error!'

        # Ensure top-level is an expression
        if not isinstance(node, ast.Expression):
            return 'Computation Error!'

        # Recursively evaluate with strict node checks
        def eval_node(n: ast.AST) -> Any:
            if isinstance(n, ast.Expression):
                return eval_node(n.body)

            # Numeric constants
            if isinstance(n, ast.Constant):
                val = n.value
                # Disallow booleans and non-numeric constants
                if isinstance(val, bool):
                    raise ValueError("Booleans not allowed")
                if not isinstance(val, (int, float)):
                    raise ValueError("Only numbers are allowed")
                return val

            # For Python versions that still use ast.Num
            if hasattr(ast, "Num") and isinstance(n, ast.Num):  # type: ignore[attr-defined]
                val = n.n  # type: ignore[attr-defined]
                if isinstance(val, bool) or not isinstance(val, (int, float)):
                    raise ValueError("Only numbers are allowed")
                return val

            # Unary operations: +, -
            if isinstance(n, ast.UnaryOp) and type(n.op) in _UNARY_OPS:
                operand = eval_node(n.operand)
                if not isinstance(operand, (int, float)):
                    raise ValueError("Invalid operand type")
                return _UNARY_OPS[type(n.op)](operand)

            # Binary operations
            if isinstance(n, ast.BinOp) and type(n.op) in _BIN_OPS:
                left = eval_node(n.left)
                right = eval_node(n.right)

                if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
                    raise ValueError("Invalid operand types")

                # Optional safety checks for power to avoid extreme computation
                if isinstance(n.op, ast.Pow):
                    # Simple bounds to prevent resource exhaustion
                    if (abs(left) > 1e6) or (abs(right) > 1e4):
                        raise ValueError("Exponentiation too large")

                # Guard against division by zero (for /, //, %)
                if isinstance(n.op, (ast.Div, ast.FloorDiv, ast.Mod)) and right == 0:
                    raise ZeroDivisionError()

                return _BIN_OPS[type(n.op)](left, right)

            # Anything else is disallowed
            raise ValueError("Disallowed expression element")

        result = eval_node(node)

        # Ensure numeric and finite
        if isinstance(result, (int, float)):
            # Convert -0.0 to 0.0 for cleaner output
            if isinstance(result, float) and result == 0.0:
                result = 0.0
            if isinstance(result, float) and not math.isfinite(result):
                return 'Computation Error!'
            return str(result)

        return 'Computation Error!'

    except Exception:
        return 'Computation Error!'
