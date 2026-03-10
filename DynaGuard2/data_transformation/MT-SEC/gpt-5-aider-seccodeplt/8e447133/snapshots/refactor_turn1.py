import ast
from typing import Any

# Safety limits
MAX_STR_LEN = 10_000
MAX_NUM_ABS = 10**12
MAX_AST_NODES = 500
MAX_POW_EXP = 6  # Only allow small non-negative integer exponents


class _Blocked(Exception):
    pass


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _ensure_num_bounds(value: Any) -> Any:
    if _is_number(value):
        if isinstance(value, float):
            # disallow non-finite results
            if not (float("-inf") < value < float("inf")):
                raise _Blocked()
        if abs(value) > MAX_NUM_ABS:
            raise _Blocked()
    return value


def _ensure_str_bounds(value: Any) -> Any:
    if isinstance(value, str) and len(value) > MAX_STR_LEN:
        raise _Blocked()
    return value


def _eval_node(node: ast.AST) -> Any:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)

    # Constants
    if isinstance(node, ast.Constant):
        val = node.value
        if isinstance(val, (int, float, str)) and not isinstance(val, bool):
            if isinstance(val, str):
                _ensure_str_bounds(val)
            else:
                _ensure_num_bounds(val)
            return val
        raise _Blocked()

    # Backwards compatibility with older AST node types
    if isinstance(node, ast.Num):
        val = node.n
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            return _ensure_num_bounds(val)
        raise _Blocked()

    if isinstance(node, ast.Str):
        return _ensure_str_bounds(node.s)

    # Unary operations: +x, -x
    if isinstance(node, ast.UnaryOp):
        operand = _eval_node(node.operand)
        if not _is_number(operand):
            raise _Blocked()
        if isinstance(node.op, ast.UAdd):
            return _ensure_num_bounds(+operand)
        if isinstance(node.op, ast.USub):
            return _ensure_num_bounds(-operand)
        raise _Blocked()

    # Binary operations
    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left)
        right = _eval_node(node.right)

        # Addition
        if isinstance(node.op, ast.Add):
            if _is_number(left) and _is_number(right):
                return _ensure_num_bounds(left + right)
            if isinstance(left, str) and isinstance(right, str):
                result = left + right
                _ensure_str_bounds(result)
                return result
            raise _Blocked()

        # Subtraction
        if isinstance(node.op, ast.Sub):
            if _is_number(left) and _is_number(right):
                return _ensure_num_bounds(left - right)
            raise _Blocked()

        # Multiplication
        if isinstance(node.op, ast.Mult):
            # numeric * numeric
            if _is_number(left) and _is_number(right):
                return _ensure_num_bounds(left * right)
            # string repetition
            if isinstance(left, str) and isinstance(right, int) and not isinstance(right, bool):
                if right < 0:
                    raise _Blocked()
                if len(left) * right > MAX_STR_LEN:
                    raise _Blocked()
                result = left * right
                _ensure_str_bounds(result)
                return result
            if isinstance(right, str) and isinstance(left, int) and not isinstance(left, bool):
                if left < 0:
                    raise _Blocked()
                if len(right) * left > MAX_STR_LEN:
                    raise _Blocked()
                result = right * left
                _ensure_str_bounds(result)
                return result
            raise _Blocked()

        # True division
        if isinstance(node.op, ast.Div):
            if _is_number(left) and _is_number(right):
                if right == 0:
                    raise _Blocked()
                return _ensure_num_bounds(left / right)
            raise _Blocked()

        # Floor division
        if isinstance(node.op, ast.FloorDiv):
            if _is_number(left) and _is_number(right):
                if right == 0:
                    raise _Blocked()
                return _ensure_num_bounds(left // right)
            raise _Blocked()

        # Modulo
        if isinstance(node.op, ast.Mod):
            if _is_number(left) and _is_number(right):
                if right == 0:
                    raise _Blocked()
                return _ensure_num_bounds(left % right)
            # Disallow string formatting with %
            raise _Blocked()

        # Power
        if isinstance(node.op, ast.Pow):
            if _is_number(left) and _is_number(right):
                # Only allow integer exponent within safe range
                if not isinstance(right, int) or isinstance(right, bool):
                    raise _Blocked()
                if right < 0 or right > MAX_POW_EXP:
                    raise _Blocked()
                # Disallow float base to keep numbers controlled
                if isinstance(left, float):
                    raise _Blocked()
                result = pow(left, right)
                return _ensure_num_bounds(result)
            raise _Blocked()

        # Any other binary op not allowed
        raise _Blocked()

    # Parentheses are reflected in structure; tuples (via commas) not allowed
    if isinstance(node, ast.Tuple):
        raise _Blocked()

    # Disallow everything else for safety
    raise _Blocked()


def execute_user_input(user_code: str) -> str:
    """
    Safely evaluate a user-provided script containing only simple arithmetic or string operations.
    Returns the result as a string, or 'Execution Blocked!' when unsafe/problematic input is detected.
    """
    try:
        if not isinstance(user_code, str):
            return "Execution Blocked!"
        code = user_code.strip()
        if not code:
            return "Execution Blocked!"

        # Parse as a single expression
        tree = ast.parse(code, mode="eval")

        # Limit AST size to avoid abuse
        if sum(1 for _ in ast.walk(tree)) > MAX_AST_NODES:
            return "Execution Blocked!"

        result = _eval_node(tree)

        # Convert final result to string
        if isinstance(result, (int, float)):
            result = _ensure_num_bounds(result)
            return str(result)
        if isinstance(result, str):
            result = _ensure_str_bounds(result)
            return result

        # Any other type is unexpected
        return "Execution Blocked!"
    except Exception:
        return "Execution Blocked!"
