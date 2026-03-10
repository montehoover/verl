import ast
import operator
import math

# TASKMANAGER safe expression processor

_MAX_EXPR_LEN = 500
_MAX_DEPTH = 25
_MAX_ABS_INT = 10 ** 12
_MAX_ABS_FLOAT = 1e12
_MAX_POW_EXPONENT = 10
_MAX_POW_BASE_ABS = 10 ** 6


class _UnsafeExpression(Exception):
    pass


def process_user_expression(expr: str) -> str:
    """
    Safely process a user-provided logical or mathematical expression.
    Returns the result as a string, or 'Processing Failed!' on any error or unsafe input.
    """
    try:
        if not isinstance(expr, str):
            return 'Processing Failed!'
        if len(expr) == 0 or len(expr) > _MAX_EXPR_LEN:
            return 'Processing Failed!'

        tree = ast.parse(expr, mode='eval')
        result = _eval_node(tree.body, depth=0)

        if not isinstance(result, (int, float, bool)):
            return 'Processing Failed!'

        # Final sanity checks for numeric ranges
        if isinstance(result, bool):
            pass  # booleans are fine
        elif isinstance(result, int):
            if abs(result) > _MAX_ABS_INT:
                return 'Processing Failed!'
        elif isinstance(result, float):
            if not math.isfinite(result) or abs(result) > _MAX_ABS_FLOAT:
                return 'Processing Failed!'

        return str(result)
    except Exception:
        return 'Processing Failed!'


# Allowed operator mappings
_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_NUMERIC = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

# Allowed comparison operators
_CMP_OPS = (
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
)


def _is_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _check_number_bounds(x):
    if isinstance(x, int):
        if abs(x) > _MAX_ABS_INT:
            raise _UnsafeExpression("Integer out of bounds")
    elif isinstance(x, float):
        if not math.isfinite(x) or abs(x) > _MAX_ABS_FLOAT:
            raise _UnsafeExpression("Float out of bounds")
    else:
        raise _UnsafeExpression("Non-numeric value where numeric required")


def _eval_node(node: ast.AST, depth: int):
    if depth > _MAX_DEPTH:
        raise _UnsafeExpression("Expression too deep")

    # Constants (numbers and booleans only)
    if isinstance(node, ast.Constant):
        val = node.value
        if isinstance(val, bool):
            return val
        if isinstance(val, int):
            _check_number_bounds(val)
            return val
        if isinstance(val, float):
            _check_number_bounds(val)
            return val
        raise _UnsafeExpression("Unsupported literal type")

    # Legacy numeric nodes (for older Python ASTs)
    if isinstance(node, ast.Num):  # type: ignore[attr-defined]
        val = node.n  # type: ignore[attr-defined]
        if isinstance(val, (int, float)):
            _check_number_bounds(val)
            return val
        raise _UnsafeExpression("Unsupported numeric literal")

    # Parenthesized tuples should not be allowed (avoid sequences)
    if isinstance(node, (ast.Tuple, ast.List, ast.Set, ast.Dict)):
        raise _UnsafeExpression("Sequences are not allowed")

    # Binary operations
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _BIN_OPS or op_type is ast.MatMult:
            raise _UnsafeExpression("Operator not allowed")

        left = _eval_node(node.left, depth + 1)
        right = _eval_node(node.right, depth + 1)

        # For numeric ops, enforce numeric operands
        if op_type is ast.Pow:
            if not _is_number(left) or not _is_number(right):
                raise _UnsafeExpression("Power requires numeric operands")
            # Limit exponentiation to avoid huge results
            if abs(float(left)) > _MAX_POW_BASE_ABS:
                raise _UnsafeExpression("Base too large for exponentiation")
            if abs(float(right)) > _MAX_POW_EXPONENT:
                raise _UnsafeExpression("Exponent too large")
        else:
            # All other numeric binops require numeric operands
            if not (
                _is_number(left) or isinstance(left, bool)
            ) or not (_is_number(right) or isinstance(right, bool)):
                raise _UnsafeExpression("Operands must be numeric or boolean")

        # Compute and validate
        try:
            res = _BIN_OPS[op_type](left, right)
        except Exception as e:
            raise _UnsafeExpression(str(e))
        # Post-check on numbers
        if isinstance(res, (int, float)):
            _check_number_bounds(res)
        return res

    # Unary operations
    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        operand = _eval_node(node.operand, depth + 1)
        if op_type in _UNARY_NUMERIC:
            if not (_is_number(operand) or isinstance(operand, bool)):
                raise _UnsafeExpression("Unary operand must be numeric or boolean")
            res = _UNARY_NUMERIC[op_type](operand)
            if isinstance(res, (int, float)):
                _check_number_bounds(res)
            return res
        if op_type is ast.Not:
            if not isinstance(operand, bool):
                raise _UnsafeExpression("Logical not requires boolean")
            return not operand
        raise _UnsafeExpression("Unary operator not allowed")

    # Boolean operations (and/or) with strict boolean operands
    if isinstance(node, ast.BoolOp):
        if isinstance(node.op, ast.And):
            current = _eval_node(node.values[0], depth + 1)
            if not isinstance(current, bool):
                raise _UnsafeExpression("Boolean operations require booleans")
            for sub in node.values[1:]:
                if not current:
                    return False
                nxt = _eval_node(sub, depth + 1)
                if not isinstance(nxt, bool):
                    raise _UnsafeExpression("Boolean operations require booleans")
                current = current and nxt
            return current
        elif isinstance(node.op, ast.Or):
            current = _eval_node(node.values[0], depth + 1)
            if not isinstance(current, bool):
                raise _UnsafeExpression("Boolean operations require booleans")
            for sub in node.values[1:]:
                if current:
                    return True
                nxt = _eval_node(sub, depth + 1)
                if not isinstance(nxt, bool):
                    raise _UnsafeExpression("Boolean operations require booleans")
                current = current or nxt
            return current
        else:
            raise _UnsafeExpression("Boolean operator not allowed")

    # Comparisons
    if isinstance(node, ast.Compare):
        left = _eval_node(node.left, depth + 1)
        comparators = [ _eval_node(c, depth + 1) for c in node.comparators ]
        ops = node.ops

        values = [left] + comparators
        for i, op in enumerate(ops):
            a = values[i]
            b = values[i + 1]
            if not (
                isinstance(a, (int, float, bool)) and isinstance(b, (int, float, bool))
            ):
                raise _UnsafeExpression("Comparisons require numeric or boolean values")
            if not isinstance(op, _CMP_OPS):
                raise _UnsafeExpression("Comparison operator not allowed")

            if isinstance(op, ast.Eq):
                ok = a == b
            elif isinstance(op, ast.NotEq):
                ok = a != b
            elif isinstance(op, ast.Lt):
                ok = a < b
            elif isinstance(op, ast.LtE):
                ok = a <= b
            elif isinstance(op, ast.Gt):
                ok = a > b
            elif isinstance(op, ast.GtE):
                ok = a >= b
            else:
                raise _UnsafeExpression("Comparison operator not allowed")

            if not isinstance(ok, bool):
                raise _UnsafeExpression("Invalid comparison result")

            if not ok:
                return False
        return True

    # Conditional expression (ternary): a if cond else b
    if isinstance(node, ast.IfExp):
        test_val = _eval_node(node.test, depth + 1)
        if not isinstance(test_val, bool):
            raise _UnsafeExpression("Ternary condition must be boolean")
        if test_val:
            return _eval_node(node.body, depth + 1)
        else:
            return _eval_node(node.orelse, depth + 1)

    # Disallow all names, calls, attributes, subscripts, lambdas, comprehensions, etc.
    if isinstance(
        node,
        (
            ast.Name,
            ast.Call,
            ast.Attribute,
            ast.Subscript,
            ast.Lambda,
            ast.ListComp,
            ast.SetComp,
            ast.DictComp if hasattr(ast, "DictComp") else ast.AST,  # safeguard
            ast.GeneratorExp,
            ast.Await,
            ast.Yield,
            ast.YieldFrom,
            ast.Assign,
            ast.AnnAssign,
            ast.AugAssign,
            ast.If,
            ast.For,
            ast.While,
            ast.With,
            ast.Return,
            ast.Delete,
            ast.Try,
            ast.Raise,
            ast.Import,
            ast.ImportFrom,
            ast.Global,
            ast.Nonlocal,
            ast.Assert,
            ast.Pass,
            ast.Break,
            ast.Continue,
        ),
    ):
        raise _UnsafeExpression("Unsupported or unsafe expression element")

    # If we encounter an AST node we didn't explicitly handle, reject it.
    raise _UnsafeExpression("Unsupported expression")
