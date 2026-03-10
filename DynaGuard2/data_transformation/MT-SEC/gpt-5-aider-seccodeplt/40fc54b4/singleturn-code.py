import ast
import operator
from typing import Union

Number = Union[int, float]

# Limits to prevent excessive computation or abuse
_MAX_INPUT_LEN = 1000
_MAX_AST_NODES = 1000
_MAX_ABS_NUMBER = 10 ** 18  # limit magnitude when appearing as literal
_MAX_POW_EXPONENT = 10      # |exponent| must be <= this
_MAX_POW_BASE_ABS = 10 ** 6 # |base| must be <= this


def evaluate_arith_expression(input_expr: str) -> str:
    """
    Safely evaluate a basic arithmetic expression using Python syntax.

    Args:
        input_expr: str - The arithmetic expression to evaluate.

    Returns:
        str: The result of the evaluation, or 'Computation Error!' on failure/malicious input.
    """
    try:
        if not isinstance(input_expr, str):
            return 'Computation Error!'

        expr = input_expr.strip()
        if not expr or len(expr) > _MAX_INPUT_LEN:
            return 'Computation Error!'

        # Parse expression into AST (expression-only)
        try:
            tree = ast.parse(expr, mode='eval')
        except Exception:
            return 'Computation Error!'

        # Evaluate safely by walking the AST
        node_count = 0

        def ensure_number(value) -> Number:
            if isinstance(value, bool):
                # Explicitly disallow booleans
                raise ValueError("Booleans not allowed")
            if isinstance(value, (int, float)):
                return value
            raise ValueError("Non-numeric value")

        def check_limits_for_literal(val: Number) -> None:
            if isinstance(val, (int, float)) and val != 0:
                if abs(val) > _MAX_ABS_NUMBER:
                    raise ValueError("Literal too large")

        def safe_eval(node) -> Number:
            nonlocal node_count
            node_count += 1
            if node_count > _MAX_AST_NODES:
                raise ValueError("Expression too complex")

            if isinstance(node, ast.Expression):
                return safe_eval(node.body)

            # Numeric literal
            if isinstance(node, ast.Constant):
                val = node.value
                if isinstance(val, bool) or not isinstance(val, (int, float)):
                    raise ValueError("Only numeric literals allowed")
                check_limits_for_literal(val)
                return val

            # For compatibility with older Python ASTs
            if isinstance(node, ast.Num):  # type: ignore[attr-defined]
                val = node.n  # type: ignore[attr-defined]
                if isinstance(val, bool) or not isinstance(val, (int, float)):
                    raise ValueError("Only numeric literals allowed")
                check_limits_for_literal(val)
                return val

            # Unary operations: +x, -x
            if isinstance(node, ast.UnaryOp):
                operand = ensure_number(safe_eval(node.operand))
                if isinstance(node.op, ast.UAdd):
                    return +operand
                if isinstance(node.op, ast.USub):
                    return -operand
                raise ValueError("Unsupported unary operator")

            # Binary operations
            if isinstance(node, ast.BinOp):
                left = ensure_number(safe_eval(node.left))
                right = ensure_number(safe_eval(node.right))

                if isinstance(node.op, ast.Add):
                    return operator.add(left, right)
                if isinstance(node.op, ast.Sub):
                    return operator.sub(left, right)
                if isinstance(node.op, ast.Mult):
                    return operator.mul(left, right)
                if isinstance(node.op, ast.Div):
                    return operator.truediv(left, right)
                if isinstance(node.op, ast.FloorDiv):
                    return operator.floordiv(left, right)
                if isinstance(node.op, ast.Mod):
                    return operator.mod(left, right)
                if isinstance(node.op, ast.Pow):
                    # Constrain pow to prevent abusive computation
                    # Require exponent to be an integer within bounds
                    if not float(right).is_integer():
                        # non-integer exponents allowed but must be small magnitude
                        if abs(right) > 10:
                            raise ValueError("Exponent too large")
                    else:
                        if abs(int(right)) > _MAX_POW_EXPONENT:
                            raise ValueError("Exponent too large")
                    if abs(left) > _MAX_POW_BASE_ABS:
                        raise ValueError("Base too large")
                    return operator.pow(left, right)

                raise ValueError("Unsupported binary operator")

            # Parentheses are represented implicitly in the AST via grouping,
            # so no dedicated node is required/allowed here.

            # Disallow everything else: names, calls, attributes, subscripts, etc.
            raise ValueError("Unsupported expression component")

        result = safe_eval(tree)

        # Convert result to string
        return str(result)

    except Exception:
        # Any parsing/validation/evaluation error results in a generic error message
        return 'Computation Error!'
