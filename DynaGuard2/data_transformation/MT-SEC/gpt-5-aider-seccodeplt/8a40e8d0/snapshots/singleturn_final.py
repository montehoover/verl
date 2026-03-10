import ast
import math
from typing import Union

COMPUTATION_ERROR = "Computation Error!"

# Limits to keep evaluation safe and bounded
_MAX_EXPR_LENGTH = 1024
_MAX_AST_NODES = 200
_MAX_ABS_RESULT = 1e100
_MAX_BASE_ABS_FOR_POW = 1_000_000
_MAX_INT_EXPONENT = 1000
_MAX_FLOAT_EXPONENT = 100


def evaluate_expression(expr_string: str) -> str:
    """
    Safely evaluate a basic arithmetic expression written in Python syntax.

    Allowed:
      - Numbers (ints, floats; underscores in numeric literals are fine)
      - Binary operators: +, -, *, /, //, %, **
      - Unary operators: +, -
      - Parentheses

    Disallowed (will return 'Computation Error!'):
      - Any names/variables, attribute access, function calls, comprehensions, etc.
      - Non-numeric literals (strings, bytes), complex numbers, booleans
      - Excessively large expressions or exponents
    """
    try:
        if not isinstance(expr_string, str):
            return COMPUTATION_ERROR

        s = expr_string.strip()
        if not s or len(s) > _MAX_EXPR_LENGTH:
            return COMPUTATION_ERROR

        try:
            tree = ast.parse(s, mode="eval")
        except Exception:
            return COMPUTATION_ERROR

        # Validate AST nodes are only from an allowed safe subset
        allowed_node_types = (
            ast.Expression,
            ast.BinOp,
            ast.UnaryOp,
            ast.Constant,  # py3.8+
            ast.Num,       # py<3.8 compatibility
            # Operators
            ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
            ast.UAdd, ast.USub,
        )

        node_count = 0
        for node in ast.walk(tree):
            node_count += 1
            if node_count > _MAX_AST_NODES:
                return COMPUTATION_ERROR

            if not isinstance(node, allowed_node_types):
                # Specifically disallow names, calls, attributes, etc.
                return COMPUTATION_ERROR

            # Validate constants are numeric (int/float) and not bool/complex
            if isinstance(node, (ast.Constant, ast.Num)):
                value = getattr(node, "n", None) if isinstance(node, ast.Num) else node.value
                # Disallow booleans (bool is subclass of int), complex, None, etc.
                if isinstance(value, bool) or isinstance(value, complex):
                    return COMPUTATION_ERROR
                if not isinstance(value, (int, float)):
                    return COMPUTATION_ERROR

        def _is_reasonable_number(x: Union[int, float]) -> bool:
            if isinstance(x, bool) or isinstance(x, complex):
                return False
            if isinstance(x, float):
                if math.isnan(x) or math.isinf(x):
                    return False
                return abs(x) <= _MAX_ABS_RESULT
            # int
            # Keep within a sane magnitude to avoid pathological huge integers
            try:
                return abs(x) <= 10 ** 10000
            except Exception:
                return False

        def _eval(node: ast.AST) -> Union[int, float]:
            if isinstance(node, ast.Expression):
                return _eval(node.body)

            if isinstance(node, (ast.Constant, ast.Num)):
                value = getattr(node, "n", None) if isinstance(node, ast.Num) else node.value
                if isinstance(value, bool) or isinstance(value, complex):
                    raise ValueError("Unsafe constant")
                if not isinstance(value, (int, float)):
                    raise ValueError("Non-numeric constant")
                return value

            if isinstance(node, ast.UnaryOp):
                operand = _eval(node.operand)
                if isinstance(node.op, ast.UAdd):
                    return +operand
                if isinstance(node.op, ast.USub):
                    return -operand
                raise ValueError("Unsupported unary operator")

            if isinstance(node, ast.BinOp):
                left = _eval(node.left)
                right = _eval(node.right)

                # Type checks
                if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
                    raise ValueError("Non-numeric operand")

                if isinstance(node.op, ast.Add):
                    return left + right
                if isinstance(node.op, ast.Sub):
                    return left - right
                if isinstance(node.op, ast.Mult):
                    return left * right
                if isinstance(node.op, ast.Div):
                    # Regular division (float)
                    if right == 0:
                        raise ZeroDivisionError
                    return left / right
                if isinstance(node.op, ast.FloorDiv):
                    if right == 0:
                        raise ZeroDivisionError
                    return left // right
                if isinstance(node.op, ast.Mod):
                    if right == 0:
                        raise ZeroDivisionError
                    return left % right
                if isinstance(node.op, ast.Pow):
                    # Exponent sanity checks
                    if abs(left) > _MAX_BASE_ABS_FOR_POW:
                        raise ValueError("Base too large for exponentiation")

                    if isinstance(right, float):
                        # Limit the magnitude of float exponents
                        if not math.isfinite(right) or abs(right) > _MAX_FLOAT_EXPONENT:
                            raise ValueError("Exponent too large")
                        # Avoid complex results: negative base with non-integer exponent
                        if left < 0 and not right.is_integer():
                            raise ValueError("Complex result not allowed")
                    elif isinstance(right, int):
                        if abs(right) > _MAX_INT_EXPONENT:
                            raise ValueError("Exponent too large")
                    else:
                        raise ValueError("Invalid exponent type")

                    result = left ** right
                    # Disallow complex results
                    if isinstance(result, complex):
                        raise ValueError("Complex result")
                    return result

                raise ValueError("Unsupported binary operator")

            # Any other node types should have been filtered out earlier
            raise ValueError("Unsupported syntax")

        try:
            result = _eval(tree)
        except Exception:
            return COMPUTATION_ERROR

        if not _is_reasonable_number(result):
            return COMPUTATION_ERROR

        # Convert to string result
        return str(result)

    except Exception:
        return COMPUTATION_ERROR
