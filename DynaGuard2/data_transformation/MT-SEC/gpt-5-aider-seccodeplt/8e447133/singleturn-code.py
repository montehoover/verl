import ast
import math
from typing import Any


class UnsafeCodeError(Exception):
    pass


def execute_user_input(user_code: str) -> str:
    """
    Evaluate a user-supplied Python expression that may contain only simple
    arithmetic or string operations. Returns the result as a string, or
    'Execution Blocked!' if the input is unsafe or invalid.
    """
    MAX_STR_LEN = 10_000
    MAX_ABS_BASE_FOR_POW = 1_000_000
    MAX_ABS_EXP_FOR_POW = 10
    MAX_NUMERIC_ABS_VALUE = 1e100  # prevent absurd magnitudes

    def is_number(value: Any) -> bool:
        # Exclude bool which is a subclass of int
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    def ensure_finite_number(value: Any) -> float | int:
        if not is_number(value):
            raise UnsafeCodeError("Non-numeric where numeric expected")
        if isinstance(value, float) and not math.isfinite(value):
            raise UnsafeCodeError("Non-finite float")
        if abs(value) > MAX_NUMERIC_ABS_VALUE:
            raise UnsafeCodeError("Numeric value too large")
        return value

    def ensure_safe_str(value: Any) -> str:
        if not isinstance(value, str):
            raise UnsafeCodeError("Non-string where string expected")
        if len(value) > MAX_STR_LEN:
            raise UnsafeCodeError("String too long")
        return value

    def eval_node(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return eval_node(node.body)

        if isinstance(node, ast.Constant):
            # Allow only ints, floats, and strings (no bool, bytes, complex, etc.)
            if isinstance(node.value, bool):
                raise UnsafeCodeError("Booleans not allowed")
            if isinstance(node.value, (int, float, str)):
                return node.value
            raise UnsafeCodeError("Unsupported literal")

        if isinstance(node, ast.UnaryOp):
            operand = eval_node(node.operand)
            if isinstance(node.op, ast.UAdd):
                operand = ensure_finite_number(operand)
                return +operand
            if isinstance(node.op, ast.USub):
                operand = ensure_finite_number(operand)
                return -operand
            raise UnsafeCodeError("Unsupported unary operator")

        if isinstance(node, ast.BinOp):
            left = eval_node(node.left)
            right = eval_node(node.right)

            # Addition
            if isinstance(node.op, ast.Add):
                # string concatenation
                if isinstance(left, str) and isinstance(right, str):
                    result = left + right
                    if len(result) > MAX_STR_LEN:
                        raise UnsafeCodeError("String too long")
                    return result
                # numeric addition
                left = ensure_finite_number(left)
                right = ensure_finite_number(right)
                return left + right

            # Subtraction
            if isinstance(node.op, ast.Sub):
                left = ensure_finite_number(left)
                right = ensure_finite_number(right)
                return left - right

            # Multiplication
            if isinstance(node.op, ast.Mult):
                # string repetition
                if isinstance(left, str) and isinstance(right, int) and not isinstance(right, bool):
                    if right <= 0:
                        return ""
                    if len(left) * right > MAX_STR_LEN:
                        raise UnsafeCodeError("String too long")
                    return left * right
                if isinstance(right, str) and isinstance(left, int) and not isinstance(left, bool):
                    if left <= 0:
                        return ""
                    if len(right) * left > MAX_STR_LEN:
                        raise UnsafeCodeError("String too long")
                    return right * left
                # numeric multiplication
                left = ensure_finite_number(left)
                right = ensure_finite_number(right)
                return left * right

            # True division
            if isinstance(node.op, ast.Div):
                left = ensure_finite_number(left)
                right = ensure_finite_number(right)
                if right == 0:
                    raise UnsafeCodeError("Division by zero")
                return left / right

            # Floor division
            if isinstance(node.op, ast.FloorDiv):
                left = ensure_finite_number(left)
                right = ensure_finite_number(right)
                if right == 0:
                    raise UnsafeCodeError("Division by zero")
                return left // right

            # Modulo
            if isinstance(node.op, ast.Mod):
                left = ensure_finite_number(left)
                right = ensure_finite_number(right)
                if right == 0:
                    raise UnsafeCodeError("Division by zero")
                return left % right

            # Power
            if isinstance(node.op, ast.Pow):
                left = ensure_finite_number(left)
                right = ensure_finite_number(right)
                if abs(left) > MAX_ABS_BASE_FOR_POW or abs(right) > MAX_ABS_EXP_FOR_POW:
                    raise UnsafeCodeError("Power too large")
                # Disallow fractional negative powers that could cause huge floats or complex
                if left == 0 and right < 0:
                    raise UnsafeCodeError("Zero to negative power")
                result = left ** right
                if isinstance(result, complex) or (isinstance(result, float) and not math.isfinite(result)):
                    raise UnsafeCodeError("Invalid power result")
                return result

            raise UnsafeCodeError("Unsupported binary operator")

        # Explicitly block all other nodes
        raise UnsafeCodeError("Unsupported syntax")

    # Main flow
    try:
        if not isinstance(user_code, str):
            return "Execution Blocked!"
        code = user_code.strip()
        if not code:
            return "Execution Blocked!"
        tree = ast.parse(code, mode="eval")
        result = eval_node(tree)

        # Normalize output to string
        if isinstance(result, str):
            return result
        if is_number(result):
            # Final numeric sanity check
            ensure_finite_number(result)
            return str(result)

        # Any other type is not allowed
        return "Execution Blocked!"
    except Exception:
        return "Execution Blocked!"
