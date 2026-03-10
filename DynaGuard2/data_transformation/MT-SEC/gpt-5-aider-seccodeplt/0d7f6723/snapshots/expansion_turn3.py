import ast
import operator
from typing import Union


_ALLOWED_NUMERIC_TYPES = (int, float)


def _is_allowed_expr(node: ast.AST) -> bool:
    # Literals
    if isinstance(node, ast.Constant):
        return isinstance(node.value, (_ALLOWED_NUMERIC_TYPES + (str,)))
    # Binary operations: arithmetic and string concatenation/repetition
    if isinstance(node, ast.BinOp):
        if not isinstance(
            node.op,
            (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow),
        ):
            return False
        return _is_allowed_expr(node.left) and _is_allowed_expr(node.right)
    # Unary operations: +x, -x
    if isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, (ast.UAdd, ast.USub)):
            return False
        return _is_allowed_expr(node.operand)
    # Parentheses do not create nodes; tuples (with commas) should be disallowed
    if isinstance(node, (ast.Tuple, ast.List, ast.Set, ast.Dict)):
        return False
    # Disallow everything that could be harmful or evaluate external code/IO
    if isinstance(
        node,
        (
            ast.Call,
            ast.Attribute,
            ast.Subscript,
            ast.Name,
            ast.Lambda,
            ast.IfExp,
            ast.Compare,
            ast.BoolOp,
            ast.Await,
            ast.Yield,
            ast.YieldFrom,
            ast.Starred,
            ast.ListComp,
            ast.SetComp,
            ast.DictComp,
            ast.GeneratorExp,
            ast.JoinedStr,       # f-strings
            ast.FormattedValue,  # f-strings
        ),
    ):
        return False
    # Unknown/unsupported node type
    return False


def check_script_syntax(script: str) -> bool:
    """
    Returns True if the provided script contains only simple arithmetic
    (using numeric literals with +, -, *, /, //, %, **, unary +/-)
    or simple string operations (string literals with + for concatenation
    or * for repetition), and contains no names, calls, attributes, imports,
    or any other potentially harmful constructs.
    """
    try:
        tree = ast.parse(script, mode="exec")
    except SyntaxError:
        return False

    # Allow empty script
    if not tree.body:
        return True

    # Only allow expression statements where the expressions pass the whitelist
    for stmt in tree.body:
        # Disallow any non-expression statements (e.g., assignments, imports, etc.)
        if not isinstance(stmt, ast.Expr):
            return False
        if not _is_allowed_expr(stmt.value):
            return False

    return True


class _SecurityError(Exception):
    pass


_BIN_OPS_NUMERIC = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPS_NUMERIC = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _is_numeric(value):
    return isinstance(value, _ALLOWED_NUMERIC_TYPES)


def _eval_allowed_expr(node: ast.AST):
    # Literals
    if isinstance(node, ast.Constant):
        val = node.value
        if isinstance(val, (_ALLOWED_NUMERIC_TYPES + (str,))):
            return val
        raise _SecurityError("SecurityError: disallowed literal type")

    # Unary operations (only on numeric)
    if isinstance(node, ast.UnaryOp):
        if type(node.op) not in _UNARY_OPS_NUMERIC:
            raise _SecurityError("SecurityError: disallowed unary operation")
        operand = _eval_allowed_expr(node.operand)
        if not _is_numeric(operand):
            raise _SecurityError("SecurityError: unary operation on non-numeric")
        return _UNARY_OPS_NUMERIC[type(node.op)](operand)

    # Binary operations
    if isinstance(node, ast.BinOp):
        left = _eval_allowed_expr(node.left)
        right = _eval_allowed_expr(node.right)

        # Addition: numeric + numeric or str + str
        if isinstance(node.op, ast.Add):
            if _is_numeric(left) and _is_numeric(right):
                return operator.add(left, right)
            if isinstance(left, str) and isinstance(right, str):
                return left + right
            raise _SecurityError("SecurityError: disallowed + operation")

        # Multiplication:
        # - numeric * numeric
        # - str * int or int * str (repetition)
        if isinstance(node.op, ast.Mult):
            if _is_numeric(left) and _is_numeric(right):
                return operator.mul(left, right)
            if isinstance(left, str) and isinstance(right, int):
                return left * right
            if isinstance(right, str) and isinstance(left, int):
                return left * right
            raise _SecurityError("SecurityError: disallowed * operation")

        # Other arithmetic ops only for numeric types
        if type(node.op) in _BIN_OPS_NUMERIC:
            if _is_numeric(left) and _is_numeric(right):
                return _BIN_OPS_NUMERIC[type(node.op)](left, right)
            raise _SecurityError("SecurityError: arithmetic on non-numeric types")

        # Any other bin op is disallowed
        raise _SecurityError("SecurityError: disallowed binary operation")

    # Parentheses don't produce nodes; ensure sequences/dicts are disallowed
    if isinstance(node, (ast.Tuple, ast.List, ast.Set, ast.Dict)):
        raise _SecurityError("SecurityError: disallowed container literal")

    # Everything else is disallowed
    raise _SecurityError("SecurityError: disallowed expression")


def evaluate_safe_script(script: str):
    """
    Evaluate a script that contains only allowed operations:
      - numeric literals and arithmetic (+, -, *, /, //, %, **, unary +/-)
      - string literals with + (concatenation) and * (repetition with int)
    Returns the value of the last expression (None for empty script).
    If a disallowed construct or operation is detected, returns a security error message.
    """
    try:
        tree = ast.parse(script, mode="exec")
    except SyntaxError:
        return "SecurityError: invalid syntax"

    # Empty script
    if not tree.body:
        return None

    last_result = None
    try:
        for stmt in tree.body:
            # Only permit bare expressions
            if not isinstance(stmt, ast.Expr):
                raise _SecurityError("SecurityError: disallowed statement")
            # Quick structural validation using the existing checker
            if not _is_allowed_expr(stmt.value):
                raise _SecurityError("SecurityError: disallowed expression structure")
            # Evaluate with strict type/operator checks
            last_result = _eval_allowed_expr(stmt.value)
        return last_result
    except _SecurityError as se:
        return str(se)
    except Exception as e:
        return f"ExecutionError: {e.__class__.__name__}: {e}"


def parse_user_script(script_str: str):
    """
    Validate and safely evaluate a user-provided script string.

    - Returns the evaluation result for allowed scripts.
    - Returns 'Execution Blocked!' if the script contains unsafe/forbidden parts.
    - May return 'ExecutionError: ...' if a runtime error occurs during safe evaluation.
    """
    # Syntactic and structural validation
    if not check_script_syntax(script_str):
        return "Execution Blocked!"

    # Evaluate with strict checker
    result = evaluate_safe_script(script_str)

    # Map any security-related findings to the unified blocked message
    if isinstance(result, str) and result.startswith("SecurityError:"):
        return "Execution Blocked!"

    return result
