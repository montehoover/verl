import ast
import operator as _op
from typing import Any

# Safety limits
_MAX_NODES = 1000
_MAX_SEQUENCE_LENGTH = 100_000
_MAX_STRING_LENGTH = 100_000
_MAX_INT_BITS = 1_000_000
_MAX_POWER_EXPONENT = 10_000
_MAX_SHIFT = 100_000


def process_user_query(user_input: str) -> str:
    """
    Evaluate a user-provided Python expression in a restricted, safe manner.

    Args:
        user_input: str - The user-provided string to be interpreted and computed as a Python expression.

    Returns:
        str - The string representation of the computed result, or 'Error!' if evaluation fails or input is untrusted.
    """
    try:
        if not isinstance(user_input, str) or not user_input.strip():
            return 'Error!'

        # Parse strictly as an expression
        tree = ast.parse(user_input, mode='eval')

        # Basic structural safety checks
        if _count_nodes(tree) > _MAX_NODES:
            return 'Error!'

        result = _safe_eval(tree.body)

        # Final sanity checks on result size
        if isinstance(result, int) and result.bit_length() > _MAX_INT_BITS:
            return 'Error!'
        if isinstance(result, (str, bytes)) and len(result) > _MAX_STRING_LENGTH:
            return 'Error!'
        if isinstance(result, (list, tuple, set, dict)) and _len_safe(result) > _MAX_SEQUENCE_LENGTH:
            return 'Error!'

        return str(result)
    except Exception:
        return 'Error!'


def _count_nodes(tree: ast.AST) -> int:
    return sum(1 for _ in ast.walk(tree))


# Allowed operations
_BIN_OPS = {
    ast.Add: _op.add,
    ast.Sub: _op.sub,
    ast.Mult: _op.mul,
    ast.Div: _op.truediv,
    ast.FloorDiv: _op.floordiv,
    ast.Mod: _op.mod,
    ast.Pow: _op.pow,
    ast.BitOr: _op.or_,
    ast.BitAnd: _op.and_,
    ast.BitXor: _op.xor,
    ast.LShift: _op.lshift,
    ast.RShift: _op.rshift,
}

_UNARY_OPS = {
    ast.UAdd: _op.pos,
    ast.USub: _op.neg,
    ast.Invert: _op.invert,
    ast.Not: _op.not_,
}

_CMP_OPS = {
    ast.Eq: _op.eq,
    ast.NotEq: _op.ne,
    ast.Lt: _op.lt,
    ast.LtE: _op.le,
    ast.Gt: _op.gt,
    ast.GtE: _op.ge,
    ast.In: lambda a, b: _contains(b, a),
    ast.NotIn: lambda a, b: not _contains(b, a),
    ast.Is: _op.is_,
    ast.IsNot: _op.is_not,
}


def _safe_eval(node: ast.AST) -> Any:
    # Constants
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float, complex, str, bytes, bool, type(None))):
            return node.value
        # Disallow other constant types (e.g., Ellipsis)
        raise ValueError("Untrusted constant")

    # Containers
    if isinstance(node, ast.Tuple):
        elts = [_safe_eval(e) for e in node.elts]
        _ensure_sequence_limits(elts)
        return tuple(elts)

    if isinstance(node, ast.List):
        elts = [_safe_eval(e) for e in node.elts]
        _ensure_sequence_limits(elts)
        return elts

    if isinstance(node, ast.Set):
        elts = {_safe_eval(e) for e in node.elts}
        _ensure_sequence_limits(elts)
        return elts

    if isinstance(node, ast.Dict):
        keys = [_safe_eval(k) if k is not None else None for k in node.keys]
        values = [_safe_eval(v) for v in node.values]
        _ensure_sequence_limits(range(len(values)))
        return {k: v for k, v in zip(keys, values)}

    # Arithmetic / bitwise binary operations
    if isinstance(node, ast.BinOp):
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)

        if type(node.op) not in _BIN_OPS:
            raise ValueError("Operator not allowed")

        # Size-aware handling
        if isinstance(node.op, ast.Pow):
            return _safe_pow(left, right)

        if isinstance(node.op, (ast.LShift, ast.RShift)):
            return _safe_shift(left, right, node.op)

        if isinstance(node.op, ast.Mult):
            return _safe_mult(left, right)

        if isinstance(node.op, ast.Add):
            return _safe_add(left, right)

        # Default safe operations
        res = _BIN_OPS[type(node.op)](left, right)
        return _post_check_numeric(res)

    # Unary operations (including 'not')
    if isinstance(node, ast.UnaryOp):
        if type(node.op) not in _UNARY_OPS:
            raise ValueError("Unary operator not allowed")
        operand = _safe_eval(node.operand)
        return _UNARY_OPS[type(node.op)](operand)

    # Boolean operations (and/or) with short-circuit
    if isinstance(node, ast.BoolOp):
        if not isinstance(node.op, (ast.And, ast.Or)):
            raise ValueError("Bool operator not allowed")
        values = node.values
        if not values:
            raise ValueError("Empty BoolOp")
        if isinstance(node.op, ast.And):
            result = True
            for v in values:
                result = _safe_eval(v)
                if not result:
                    break
            return result
        else:  # Or
            result = False
            for v in values:
                result = _safe_eval(v)
                if result:
                    break
            return result

    # Comparisons
    if isinstance(node, ast.Compare):
        left = _safe_eval(node.left)
        rights = [ _safe_eval(c) for c in node.comparators ]
        ops = node.ops

        if len(ops) != len(rights):
            raise ValueError("Malformed comparison")

        cur_left = left
        for op_node, r in zip(ops, rights):
            if type(op_node) not in _CMP_OPS:
                raise ValueError("Comparison operator not allowed")
            ok = _CMP_OPS[type(op_node)](cur_left, r)
            if not isinstance(ok, bool):
                ok = bool(ok)
            if not ok:
                return False
            cur_left = r
        return True

    # Subscript and slicing
    if isinstance(node, ast.Subscript):
        value = _safe_eval(node.value)
        index = _safe_eval_slice(node.slice)
        try:
            result = value[index]
        except Exception as e:
            raise ValueError("Invalid subscript") from e
        return result

    # Parenthesized expressions are just nested; handled implicitly

    # Disallowed constructs
    if isinstance(node, (ast.Name, ast.Attribute, ast.Call, ast.Lambda,
                         ast.IfExp, ast.ListComp, ast.SetComp, ast.DictComp,
                         ast.GeneratorExp, ast.Await, ast.Yield, ast.YieldFrom,
                         ast.FormattedValue, ast.JoinedStr, ast.Starred)):
        raise ValueError("Untrusted construct")

    # If we encounter any other node type, reject
    raise ValueError(f"Untrusted or unsupported expression: {type(node).__name__}")


def _safe_pow(left: Any, right: Any) -> Any:
    # Restrict exponentiation for safety
    if isinstance(right, (int, float)) and isinstance(left, (int, float, complex)):
        if isinstance(right, int):
            if abs(right) > _MAX_POWER_EXPONENT:
                raise ValueError("Exponent too large")
            if isinstance(left, int) and right >= 0:
                # Pre-check approximate bit size
                base_bits = max(1, abs(left).bit_length())
                if base_bits * right > _MAX_INT_BITS:
                    raise ValueError("Result too large")
        # Perform pow, then check
        res = pow(left, right)
        return _post_check_numeric(res)
    # Disallow other pow combinations (e.g., sequences)
    raise ValueError("Invalid operands for power")


def _safe_shift(left: Any, right: Any, op_node: ast.AST) -> Any:
    if not isinstance(left, int) or not isinstance(right, int):
        raise ValueError("Shift operands must be ints")
    if right < 0 or right > _MAX_SHIFT:
        raise ValueError("Shift too large")
    if isinstance(op_node, ast.LShift):
        # Pre-check bit growth
        if left != 0 and (abs(left).bit_length() + right) > _MAX_INT_BITS:
            raise ValueError("Result too large")
        res = _op.lshift(left, right)
    else:
        res = _op.rshift(left, right)
    return _post_check_numeric(res)


def _safe_mult(left: Any, right: Any) -> Any:
    # Sequence repetition checks
    if isinstance(left, int) and isinstance(right, (str, bytes, list, tuple)):
        return _safe_repeat(right, left)
    if isinstance(right, int) and isinstance(left, (str, bytes, list, tuple)):
        return _safe_repeat(left, right)

    # Numeric multiplication (or other types that Python supports)
    res = _op.mul(left, right)
    return _post_check_numeric(res)


def _safe_add(left: Any, right: Any) -> Any:
    # Sequence concatenation size limits
    if isinstance(left, (str, bytes)) and isinstance(right, type(left)):
        if len(left) + len(right) > _MAX_STRING_LENGTH:
            raise ValueError("String too large")
        return left + right
    if isinstance(left, list) and isinstance(right, list):
        if len(left) + len(right) > _MAX_SEQUENCE_LENGTH:
            raise ValueError("List too large")
        return left + right
    if isinstance(left, tuple) and isinstance(right, tuple):
        if len(left) + len(right) > _MAX_SEQUENCE_LENGTH:
            raise ValueError("Tuple too large")
        return left + right

    # Fallback to Python behavior (e.g., numeric addition)
    res = _op.add(left, right)
    return _post_check_numeric(res)


def _safe_repeat(seq: Any, n: int) -> Any:
    if n < 0:
        # Python allows negative which results in empty or reversed semantics for some; allow but bounded
        n = max(n, -_MAX_SEQUENCE_LENGTH)
    # Compute resulting size
    if isinstance(seq, (str, bytes)):
        if len(seq) * abs(n) > _MAX_STRING_LENGTH:
            raise ValueError("String too large")
    elif isinstance(seq, (list, tuple)):
        if len(seq) * abs(n) > _MAX_SEQUENCE_LENGTH:
            raise ValueError("Sequence too large")
    else:
        raise ValueError("Invalid sequence for repetition")
    return seq * n


def _safe_eval_slice(slice_node: ast.AST) -> Any:
    # In recent Python versions, slice can be ast.Slice or any expression
    if isinstance(slice_node, ast.Slice):
        lower = _safe_eval(slice_node.lower) if slice_node.lower is not None else None
        upper = _safe_eval(slice_node.upper) if slice_node.upper is not None else None
        step = _safe_eval(slice_node.step) if slice_node.step is not None else None
        return slice(lower, upper, step)
    else:
        # Evaluate general index expression (including tuples for multi-index)
        return _safe_eval(slice_node)


def _post_check_numeric(value: Any) -> Any:
    if isinstance(value, int):
        if value.bit_length() > _MAX_INT_BITS:
            raise ValueError("Integer too large")
    return value


def _ensure_sequence_limits(seq_like: Any) -> None:
    # Works with sequences and sets and ranges (we only need length)
    try:
        length = _len_safe(seq_like)
    except Exception:
        # If length not available, ignore here; later checks will apply on final result
        return
    if length > _MAX_SEQUENCE_LENGTH:
        raise ValueError("Sequence too large")


def _len_safe(obj: Any) -> int:
    # Safely get length for builtins we manage
    if isinstance(obj, (list, tuple, set, dict, range, str, bytes)):
        return len(obj)
    # Fallback: try len, else zero (for non-sized results)
    try:
        return len(obj)  # type: ignore
    except Exception:
        return 0


def _contains(container: Any, item: Any) -> bool:
    try:
        return item in container
    except Exception:
        # Some containers may error on membership test; treat as False
        return False
