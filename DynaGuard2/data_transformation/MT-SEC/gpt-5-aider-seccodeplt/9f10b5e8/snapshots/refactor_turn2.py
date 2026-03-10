import ast
from typing import Any


class _UnsafeExpression(Exception):
    pass


_MAX_AST_NODES = 1000
_MAX_RESULT_STR_LEN = 10000
_MAX_POW_EXPONENT = 1000
_MAX_POW_BASE_ABS = 10 ** 6


def run_user_query(query: str) -> str:
    """
    Evaluate a user's query as a safe Python expression.

    Args:
        query: The user's query string.

    Returns:
        str: The stringified result of the evaluation, or 'Error!' on error/danger.
    """
    try:
        tree = _parse_query(query)
        return _evaluate_query_ast(tree)
    except Exception:
        return 'Error!'


def _parse_query(query: str) -> ast.Expression:
    """
    Pure function to validate and parse the input query into an AST Expression.
    Raises on invalid or dangerous input.
    """
    if not isinstance(query, str):
        raise ValueError("Query must be a string")

    stripped = query.strip()
    if not stripped:
        raise ValueError("Empty query")

    # Parse as a single expression
    tree = ast.parse(stripped, mode='eval')
    if not isinstance(tree, ast.Expression):
        raise _UnsafeExpression("Not an expression")

    # Limit overall AST size to prevent pathological inputs
    if sum(1 for _ in ast.walk(tree)) > _MAX_AST_NODES:
        raise _UnsafeExpression("AST too large")

    return tree


def _evaluate_query_ast(tree: ast.Expression) -> str:
    """
    Pure function to evaluate a validated AST Expression and return a string result.
    Raises on evaluation errors or unsafe constructs encountered.
    """
    result = _eval_safe(tree.body)

    # Final result size constraint
    out = str(result)
    if len(out) > _MAX_RESULT_STR_LEN:
        raise _UnsafeExpression("Result too large")

    return out


def _eval_safe(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        # Accept basic constants (numbers, strings, bytes, booleans, None, Ellipsis)
        return node.value

    if isinstance(node, ast.Tuple):
        return tuple(_eval_safe(elt) for elt in node.elts)

    if isinstance(node, ast.List):
        return [_eval_safe(elt) for elt in node.elts]

    if isinstance(node, ast.Set):
        return {_eval_safe(elt) for elt in node.elts}

    if isinstance(node, ast.Dict):
        keys = [_eval_safe(k) for k in node.keys]
        vals = [_eval_safe(v) for v in node.values]
        return dict(zip(keys, vals))

    if isinstance(node, ast.UnaryOp):
        operand = _eval_safe(node.operand)
        if isinstance(node.op, ast.UAdd):
            return +operand
        if isinstance(node.op, ast.USub):
            return -operand
        if isinstance(node.op, ast.Not):
            return not operand
        if isinstance(node.op, ast.Invert):
            return ~operand
        raise _UnsafeExpression("Unsupported unary operator")

    if isinstance(node, ast.BinOp):
        left = _eval_safe(node.left)
        right = _eval_safe(node.right)

        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.FloorDiv):
            return left // right
        if isinstance(node.op, ast.Mod):
            return left % right
        if isinstance(node.op, ast.Pow):
            # Guard against huge exponentials
            if not _is_number(left) or not _is_number(right):
                raise _UnsafeExpression("Pow on non-numbers")
            if abs(float(right)) > _MAX_POW_EXPONENT:
                raise _UnsafeExpression("Exponent too large")
            if _is_integer_like(left) and abs(int(left)) > _MAX_POW_BASE_ABS:
                raise _UnsafeExpression("Base too large")
            return left ** right

        # Disallow bitwise shifts and matmul etc.
        raise _UnsafeExpression("Unsupported binary operator")

    if isinstance(node, ast.BoolOp):
        if isinstance(node.op, ast.And):
            value = True
            for v in node.values:
                value = _eval_safe(v)
                if not value:
                    return value
            return value
        if isinstance(node.op, ast.Or):
            for v in node.values:
                value = _eval_safe(v)
                if value:
                    return value
            return value
        raise _UnsafeExpression("Unsupported boolean operator")

    if isinstance(node, ast.Compare):
        left = _eval_safe(node.left)
        for op, comparator in zip(node.ops, node.comparators):
            right = _eval_safe(comparator)
            if isinstance(op, ast.Eq):
                ok = left == right
            elif isinstance(op, ast.NotEq):
                ok = left != right
            elif isinstance(op, ast.Lt):
                ok = left < right
            elif isinstance(op, ast.LtE):
                ok = left <= right
            elif isinstance(op, ast.Gt):
                ok = left > right
            elif isinstance(op, ast.GtE):
                ok = left >= right
            elif isinstance(op, ast.In):
                ok = left in right
            elif isinstance(op, ast.NotIn):
                ok = left not in right
            elif isinstance(op, ast.Is):
                ok = left is right
            elif isinstance(op, ast.IsNot):
                ok = left is not right
            else:
                raise _UnsafeExpression("Unsupported comparison operator")
            if not ok:
                return False
            left = right
        return True

    if isinstance(node, ast.IfExp):
        return _eval_safe(node.body) if _eval_safe(node.test) else _eval_safe(node.orelse)

    # Explicitly disallow potentially dangerous nodes
    forbidden = (
        ast.Call,
        ast.Attribute,
        ast.Subscript,
        ast.Name,
        ast.Lambda,
        ast.ListComp,
        ast.SetComp,
        ast.DictComp,
        ast.GeneratorExp,
        ast.Await,
        ast.Yield,
        ast.YieldFrom,
        ast.JoinedStr,       # f-strings
        ast.FormattedValue,  # f-string parts
        ast.NamedExpr,       # walrus operator
        ast.Starred,
    )
    if isinstance(node, forbidden):
        raise _UnsafeExpression("Forbidden syntax")

    # Any other node types are rejected
    raise _UnsafeExpression(f"Unsupported syntax: {type(node).__name__}")


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float, complex))


def _is_integer_like(value: Any) -> bool:
    return isinstance(value, int)
