import ast
from typing import Any


_EXECUTION_BLOCKED = "Execution Blocked!"

# Safety limits
_MAX_INT_ABS = 10**12
_MAX_FLOAT_ABS = 1e12
_MAX_STRING_LEN = 10_000
_MAX_POW_EXPONENT = 10  # caps exponent size to avoid DoS
_MAX_REPEAT_RESULT_LEN = 10_000  # cap resulting string size when repeating


def run_user_code(python_code: str) -> str:
    """
    Securely evaluates a small user-supplied Python expression limited to
    basic arithmetic and string manipulations.

    - Returns the stringified result if safe and successful.
    - Returns 'Execution Blocked!' if the script is unsafe or invalid.

    Supported operations (non-exhaustive overview):
      - Numeric literals (int, float)
      - String literals
      - Arithmetic: +, -, *, /, //, %, ** (with exponent limit)
      - String concatenation (+) and repetition (* with int)
      - Indexing and slicing on strings (e.g., "abc"[0], "abc"[1:2])
      - Comparisons: ==, !=, <, <=, >, >= (ordering only within same type)
      - Boolean ops 'and'/'or' for boolean expressions only
      - Unary + and -

    Disallowed:
      - Names, variables, attributes, calls, imports, comprehensions, etc.
      - Any statement (only a single expression is allowed)
    """
    try:
        tree = validate_user_code(python_code)
        result = execute_validated_ast(tree)
        return str(result)
    except Exception:
        return _EXECUTION_BLOCKED


def validate_user_code(python_code: str) -> ast.Expression:
    """
    Parse and validate user-supplied code. Returns the parsed AST (Expression)
    if the code is safe; otherwise raises an Exception.
    """
    # Parse only a single expression
    tree = ast.parse(python_code, mode="eval")
    _validate_ast(tree)
    return tree


def execute_validated_ast(tree: ast.Expression) -> Any:
    """
    Execute a previously-validated AST safely and return the result.
    Assumes 'tree' has been validated by validate_user_code/_validate_ast.
    """
    return _eval_ast(tree.body)


def _validate_ast(node: ast.AST) -> None:
    """
    Validates that the AST only contains allowed nodes and operators.
    Raises ValueError if something unsafe or unsupported is encountered.
    """
    allowed_nodes = (
        ast.Expression,
        ast.Constant,
        ast.BinOp,
        ast.UnaryOp,
        ast.BoolOp,
        ast.Compare,
        ast.Subscript,
        ast.Slice,
        ast.Index,  # older Python compatibility; harmless if unused
    )
    allowed_bin_ops = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)
    allowed_unary_ops = (ast.UAdd, ast.USub)
    allowed_bool_ops = (ast.And, ast.Or)
    allowed_cmp_ops = (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE)

    def _walk(n: ast.AST) -> None:
        if isinstance(n, ast.Expr):
            # Shouldn't appear in 'eval' mode, but allow if encountered by some versions
            _walk(n.value)
            return

        if not isinstance(n, allowed_nodes):
            raise ValueError(f"Disallowed AST node: {type(n).__name__}")

        if isinstance(n, ast.Constant):
            if not isinstance(n.value, (int, float, str, bool)) and n.value is not None:
                raise ValueError("Only int, float, str, bool, or None constants are allowed.")

        elif isinstance(n, ast.BinOp):
            if not isinstance(n.op, allowed_bin_ops):
                raise ValueError("Disallowed binary operator.")
            _walk(n.left)
            _walk(n.right)

        elif isinstance(n, ast.UnaryOp):
            if not isinstance(n.op, allowed_unary_ops):
                raise ValueError("Disallowed unary operator.")
            _walk(n.operand)

        elif isinstance(n, ast.BoolOp):
            if not isinstance(n.op, allowed_bool_ops):
                raise ValueError("Disallowed boolean operator.")
            for v in n.values:
                _walk(v)

        elif isinstance(n, ast.Compare):
            for op in n.ops:
                if not isinstance(op, allowed_cmp_ops):
                    raise ValueError("Disallowed comparison operator.")
            _walk(n.left)
            for c in n.comparators:
                _walk(c)

        elif isinstance(n, ast.Subscript):
            _walk(n.value)
            _walk(n.slice)

        elif isinstance(n, ast.Slice):
            if n.lower:
                _walk(n.lower)
            if n.upper:
                _walk(n.upper)
            if n.step:
                _walk(n.step)

        elif isinstance(n, ast.Index):
            _walk(n.value)

        elif isinstance(n, ast.Expression):
            _walk(n.body)

    _walk(node)


def _ensure_safe_number(value: Any) -> None:
    if isinstance(value, bool):
        # bool is subclass of int; handle separately
        return
    if isinstance(value, int):
        if abs(value) > _MAX_INT_ABS:
            raise ValueError("Integer magnitude too large.")
    elif isinstance(value, float):
        if not (value == value) or value in (float("inf"), float("-inf")):
            raise ValueError("Non-finite float.")
        if abs(value) > _MAX_FLOAT_ABS:
            raise ValueError("Float magnitude too large.")


def _ensure_safe_string(value: str) -> None:
    if len(value) > _MAX_STRING_LEN:
        raise ValueError("String too long.")


def _apply_binop(op: ast.operator, left: Any, right: Any) -> Any:
    # Numeric-numeric operations
    if isinstance(left, (int, float, bool)) and isinstance(right, (int, float, bool)):
        # bool participates as int; arithmetic with bool is fine in Python
        l = int(left) if isinstance(left, bool) else left
        r = int(right) if isinstance(right, bool) else right

        if isinstance(op, ast.Add):
            out = l + r
        elif isinstance(op, ast.Sub):
            out = l - r
        elif isinstance(op, ast.Mult):
            out = l * r
        elif isinstance(op, ast.Div):
            out = l / r
        elif isinstance(op, ast.FloorDiv):
            out = l // r
        elif isinstance(op, ast.Mod):
            out = l % r
        elif isinstance(op, ast.Pow):
            # Restrict exponent size to avoid DoS
            if isinstance(r, float):
                # float exponent allowed but magnitude must be small-ish
                if abs(r) > _MAX_POW_EXPONENT:
                    raise ValueError("Exponent too large.")
            else:
                if abs(r) > _MAX_POW_EXPONENT:
                    raise ValueError("Exponent too large.")
            out = l ** r
        else:
            raise ValueError("Unsupported numeric operator.")
        _ensure_safe_number(out)
        return out

    # String-string concatenation
    if isinstance(left, str) and isinstance(right, str) and isinstance(op, ast.Add):
        out = left + right
        _ensure_safe_string(out)
        return out

    # String repetition: str * int or int * str
    if isinstance(op, ast.Mult):
        if isinstance(left, str) and isinstance(right, int):
            if len(left) * max(0, right) > _MAX_REPEAT_RESULT_LEN:
                raise ValueError("Repeated string too long.")
            out = left * right
            _ensure_safe_string(out)
            return out
        if isinstance(right, str) and isinstance(left, int):
            if len(right) * max(0, left) > _MAX_REPEAT_RESULT_LEN:
                raise ValueError("Repeated string too long.")
            out = right * left
            _ensure_safe_string(out)
            return out

    # No other cross-type operations allowed
    raise ValueError("Disallowed or unsupported binary operation.")


def _apply_unaryop(op: ast.unaryop, operand: Any) -> Any:
    if not isinstance(operand, (int, float, bool, str)):
        raise ValueError("Unsupported operand type for unary operator.")

    if isinstance(op, ast.UAdd):
        if isinstance(operand, str):
            raise ValueError("Unary + not allowed on strings.")
        out = + (int(operand) if isinstance(operand, bool) else operand)
        _ensure_safe_number(out)
        return out

    if isinstance(op, ast.USub):
        if isinstance(operand, str):
            raise ValueError("Unary - not allowed on strings.")
        out = - (int(operand) if isinstance(operand, bool) else operand)
        _ensure_safe_number(out)
        return out

    raise ValueError("Unsupported unary operator.")


def _apply_compare(left: Any, ops: list[ast.cmpop], comparators: list[Any]) -> bool:
    assert len(ops) == len(comparators)
    cur = left
    for op, nxt in zip(ops, comparators):
        if isinstance(op, (ast.Lt, ast.LtE, ast.Gt, ast.GtE)):
            # Only allow ordering comparisons between same basic types
            if (isinstance(cur, (int, float, bool)) and isinstance(nxt, (int, float, bool))):
                cval = int(cur) if isinstance(cur, bool) else cur
                nval = int(nxt) if isinstance(nxt, bool) else nxt
            elif isinstance(cur, str) and isinstance(nxt, str):
                cval, nval = cur, nxt
            else:
                raise ValueError("Invalid types for ordered comparison.")
        else:
            # Eq / NotEq are allowed across types
            cval, nval = cur, nxt

        if isinstance(op, ast.Eq):
            ok = (cval == nval)
        elif isinstance(op, ast.NotEq):
            ok = (cval != nval)
        elif isinstance(op, ast.Lt):
            ok = (cval < nval)
        elif isinstance(op, ast.LtE):
            ok = (cval <= nval)
        elif isinstance(op, ast.Gt):
            ok = (cval > nval)
        elif isinstance(op, ast.GtE):
            ok = (cval >= nval)
        else:
            raise ValueError("Unsupported comparison operator.")

        if not ok:
            return False
        cur = nxt
    return True


def _apply_boolop(op: ast.boolop, values: list[Any]) -> bool:
    if not values:
        raise ValueError("Boolean operation requires values.")
    # Enforce boolean operands
    def to_bool(v: Any) -> bool:
        if isinstance(v, bool):
            return v
        # Allow comparison results used directly
        if isinstance(v, (int, float, str)):
            # Disallow truthiness of non-boolean types to keep things simple/safe
            raise ValueError("Boolean operations require boolean expressions.")
        if isinstance(v, (list, tuple, dict)):
            raise ValueError("Boolean operations require boolean expressions.")
        return bool(v)

    if isinstance(op, ast.And):
        result = True
        for v in values:
            b = to_bool(v)
            result = result and b
            if not result:
                break
        return result
    elif isinstance(op, ast.Or):
        result = False
        for v in values:
            b = to_bool(v)
            result = result or b
            if result:
                break
        return result
    else:
        raise ValueError("Unsupported boolean operator.")


def _eval_subscript(value: Any, slc: ast.AST) -> Any:
    if not isinstance(value, str):
        raise ValueError("Subscript is only allowed on strings.")

    if isinstance(slc, ast.Slice):
        lower = _eval_ast(slc.lower) if slc.lower is not None else None
        upper = _eval_ast(slc.upper) if slc.upper is not None else None
        step = _eval_ast(slc.step) if slc.step is not None else None

        if lower is not None and not isinstance(lower, int):
            raise ValueError("Slice lower must be int or None.")
        if upper is not None and not isinstance(upper, int):
            raise ValueError("Slice upper must be int or None.")
        if step is not None:
            if not isinstance(step, int):
                raise ValueError("Slice step must be int or None.")
            if step == 0:
                raise ValueError("Slice step cannot be zero.")

        out = value[slice(lower, upper, step)]
        _ensure_safe_string(out)
        return out

    # Indexing
    idx = _eval_ast(slc)
    if not isinstance(idx, int):
        raise ValueError("Index must be an integer.")
    out = value[idx]
    _ensure_safe_string(out)
    return out


def _eval_ast(node: ast.AST) -> Any:
    if isinstance(node, ast.Expression):
        return _eval_ast(node.body)

    if isinstance(node, ast.Constant):
        val = node.value
        if isinstance(val, (int, float, bool)):
            _ensure_safe_number(val)
            return val
        if isinstance(val, str):
            _ensure_safe_string(val)
            return val
        if val is None:
            # Only used for slices; not a valid standalone result
            return None
        raise ValueError("Unsupported constant type.")

    if isinstance(node, ast.BinOp):
        left = _eval_ast(node.left)
        right = _eval_ast(node.right)
        return _apply_binop(node.op, left, right)

    if isinstance(node, ast.UnaryOp):
        operand = _eval_ast(node.operand)
        return _apply_unaryop(node.op, operand)

    if isinstance(node, ast.BoolOp):
        values = [_eval_ast(v) for v in node.values]
        return _apply_boolop(node.op, values)

    if isinstance(node, ast.Compare):
        left = _eval_ast(node.left)
        comparators = [_eval_ast(c) for c in node.comparators]
        return _apply_compare(left, node.ops, comparators)

    if isinstance(node, ast.Subscript):
        value = _eval_ast(node.value)
        return _eval_subscript(value, node.slice)

    # Any other node is disallowed by validator, but keep a guard
    raise ValueError(f"Unsupported AST node: {type(node).__name__}")
