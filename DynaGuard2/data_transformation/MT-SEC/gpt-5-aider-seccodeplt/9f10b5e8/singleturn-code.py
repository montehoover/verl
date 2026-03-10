import ast
import math
import operator
from typing import Any


def run_user_query(query: str) -> str:
    """
    Evaluate a limited, safe subset of Python expressions from a user-provided query string.

    Args:
        query: str - the user's query to evaluate.

    Returns:
        str - the result of evaluation as a string, or 'Error!' on any error or unsafe input.
    """
    ERROR = "Error!"

    # Basic validation
    try:
        if not isinstance(query, str):
            return ERROR
        query = query.strip()
        if not query:
            return ERROR
        # Limit query size to reduce risk of resource abuse
        if len(query) > 1000:
            return ERROR
    except Exception:
        return ERROR

    # Limits to keep computations bounded
    MAX_CONST_INT_ABS = 10 ** 6
    MAX_RESULT_NUM_ABS = 10 ** 12
    MAX_STR_CONST_LEN = 1000
    MAX_STR_RESULT_LEN = 10000
    MAX_REPEAT = 1000  # string repetition cap
    MAX_POW_EXP_ABS = 10
    MAX_POW_BASE_ABS = 10 ** 6

    allowed_bin_ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
    }
    allowed_unary_ops = {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
        ast.Not: operator.not_,
    }
    allowed_cmp_ops = {
        ast.Eq: operator.eq,
        ast.NotEq: operator.ne,
        ast.Lt: operator.lt,
        ast.LtE: operator.le,
        ast.Gt: operator.gt,
        ast.GtE: operator.ge,
    }

    def is_number(x: Any) -> bool:
        return isinstance(x, (int, float)) and not isinstance(x, bool)

    def check_number_bounds(x: Any) -> None:
        if is_number(x):
            if isinstance(x, float):
                if not math.isfinite(x):
                    raise ValueError("non-finite number")
            if abs(x) > MAX_RESULT_NUM_ABS:
                raise ValueError("number too large")

    def check_str_bounds(s: Any) -> None:
        if isinstance(s, str) and len(s) > MAX_STR_RESULT_LEN:
            raise ValueError("string too long")

    def eval_node(node: ast.AST) -> Any:
        # Constants
        if isinstance(node, ast.Constant):
            val = node.value
            if isinstance(val, (bytes, bytearray)):
                raise ValueError("bytes not allowed")
            if val is None:
                raise ValueError("None not allowed")
            if isinstance(val, bool):
                return val
            if isinstance(val, int):
                if abs(val) > MAX_CONST_INT_ABS:
                    raise ValueError("int constant too large")
                return val
            if isinstance(val, float):
                if not math.isfinite(val):
                    raise ValueError("float constant not finite")
                if abs(val) > MAX_CONST_INT_ABS:
                    raise ValueError("float constant too large")
                return val
            if isinstance(val, str):
                if len(val) > MAX_STR_CONST_LEN:
                    raise ValueError("string constant too long")
                return val
            # Any other constant types are disallowed
            raise ValueError("unsupported constant")

        # Python <3.8 compatibility (NameConstant/Num/Str)
        if isinstance(node, ast.NameConstant):
            if node.value in (True, False):
                return node.value
            raise ValueError("disallowed constant")
        if isinstance(node, ast.Num):
            val = node.n
            if isinstance(val, int) and abs(val) <= MAX_CONST_INT_ABS:
                return val
            if isinstance(val, float) and math.isfinite(val) and abs(val) <= MAX_CONST_INT_ABS:
                return val
            raise ValueError("numeric constant too large")
        if isinstance(node, ast.Str):
            if len(node.s) <= MAX_STR_CONST_LEN:
                return node.s
            raise ValueError("string constant too long")

        # Allow True/False names (depending on Python version AST)
        if isinstance(node, ast.Name):
            if node.id in ("True", "False"):
                return node.id == "True"
            raise ValueError("names not allowed")

        # Parent expression
        if isinstance(node, ast.Expression):
            return eval_node(node.body)

        # Unary operations: +x, -x, not x
        if isinstance(node, ast.UnaryOp) and type(node.op) in allowed_unary_ops:
            operand = eval_node(node.operand)
            op = allowed_unary_ops[type(node.op)]
            if op is operator.not_:
                if not isinstance(operand, bool):
                    raise ValueError("not expects boolean")
                result = op(operand)
            else:
                if not is_number(operand):
                    raise ValueError("unary expects number")
                result = op(operand)
            check_number_bounds(result)
            return result

        # Binary operations
        if isinstance(node, ast.BinOp) and type(node.op) in allowed_bin_ops:
            left = eval_node(node.left)
            right = eval_node(node.right)
            op = allowed_bin_ops[type(node.op)]

            # Power restrictions
            if isinstance(node.op, ast.Pow):
                if not is_number(left) or not is_number(right):
                    raise ValueError("pow expects numbers")
                if abs(left) > MAX_POW_BASE_ABS:
                    raise ValueError("pow base too large")
                if not float(right).is_integer():
                    raise ValueError("pow exponent must be integer")
                if abs(int(right)) > MAX_POW_EXP_ABS:
                    raise ValueError("pow exponent too large")
                result = op(left, int(right))
                check_number_bounds(result)
                return result

            # Division/mod checks
            if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)):
                if not is_number(left) or not is_number(right):
                    raise ValueError("division expects numbers")
                if right == 0:
                    raise ValueError("division by zero")
                result = op(left, right)
                check_number_bounds(result)
                return result

            # Multiplication (including string repetition)
            if isinstance(node.op, ast.Mult):
                # number * number
                if is_number(left) and is_number(right):
                    result = op(left, right)
                    check_number_bounds(result)
                    return result
                # string * int OR int * string
                if isinstance(left, str) and isinstance(right, int):
                    if right < 0 or right > MAX_REPEAT:
                        raise ValueError("excessive repetition")
                    if len(left) * right > MAX_STR_RESULT_LEN:
                        raise ValueError("result string too long")
                    result = left * right
                    check_str_bounds(result)
                    return result
                if isinstance(right, str) and isinstance(left, int):
                    if left < 0 or left > MAX_REPEAT:
                        raise ValueError("excessive repetition")
                    if len(right) * left > MAX_STR_RESULT_LEN:
                        raise ValueError("result string too long")
                    result = right * left
                    check_str_bounds(result)
                    return result
                raise ValueError("unsupported multiplication types")

            # Addition/Subtraction/Modulo of strings or numbers
            if isinstance(node.op, ast.Add):
                if is_number(left) and is_number(right):
                    result = op(left, right)
                    check_number_bounds(result)
                    return result
                if isinstance(left, str) and isinstance(right, str):
                    if len(left) + len(right) > MAX_STR_RESULT_LEN:
                        raise ValueError("result string too long")
                    result = left + right
                    check_str_bounds(result)
                    return result
                raise ValueError("unsupported add types")

            if isinstance(node.op, ast.Sub):
                if is_number(left) and is_number(right):
                    result = op(left, right)
                    check_number_bounds(result)
                    return result
                raise ValueError("unsupported sub types")

            if isinstance(node.op, ast.Mod):
                # Only numeric modulo allowed (string formatting % is disallowed)
                if is_number(left) and is_number(right):
                    if right == 0:
                        raise ValueError("modulo by zero")
                    result = op(left, right)
                    check_number_bounds(result)
                    return result
                raise ValueError("unsupported mod types")

            # Fallback (shouldn't reach due to explicit branches)
            raise ValueError("unsupported binary op")

        # Boolean operations: and/or with booleans
        if isinstance(node, ast.BoolOp):
            if isinstance(node.op, ast.And):
                result = True
                for v in node.values:
                    ev = eval_node(v)
                    if not isinstance(ev, bool):
                        raise ValueError("and expects booleans")
                    result = result and ev
                    if not result:
                        break
                return result
            if isinstance(node.op, ast.Or):
                result = False
                for v in node.values:
                    ev = eval_node(v)
                    if not isinstance(ev, bool):
                        raise ValueError("or expects booleans")
                    result = result or ev
                    if result:
                        break
                return result
            raise ValueError("unsupported boolean op")

        # Comparisons: chainable (a < b <= c)
        if isinstance(node, ast.Compare):
            left = eval_node(node.left)
            # Allow comparisons for numbers and strings only
            prev = left
            for op_node, comp in zip(node.ops, node.comparators):
                right = eval_node(comp)
                if not ((is_number(prev) and is_number(right)) or (isinstance(prev, str) and isinstance(right, str)) or (isinstance(prev, bool) and isinstance(right, bool))):
                    raise ValueError("unsupported comparison types")
                op_type = type(op_node)
                if op_type not in allowed_cmp_ops:
                    raise ValueError("unsupported comparison operator")
                if not allowed_cmp_ops[op_type](prev, right):
                    return False
                prev = right
            return True

        # Explicitly disallow all other nodes:
        raise ValueError("disallowed syntax")

    try:
        tree = ast.parse(query, mode="eval")
        result = eval_node(tree)
        # Final result checks
        check_number_bounds(result)
        check_str_bounds(result)

        # Convert to string for output
        return str(result)
    except Exception:
        return ERROR
