import ast
import io
import operator as _op
from contextlib import redirect_stdout
from typing import Union, Any, Optional, List, Dict

_Number = Union[int, float]
_Result = Union[int, float, bool, str]

_BIN_OPS = {
    ast.Add: _op.add,
    ast.Sub: _op.sub,
    ast.Mult: _op.mul,
    ast.Div: _op.truediv,
    ast.FloorDiv: _op.floordiv,
    ast.Mod: _op.mod,
    ast.Pow: _op.pow,
}

_UNARY_OPS = {
    ast.UAdd: _op.pos,
    ast.USub: _op.neg,
    ast.Not: _op.not_,
}

_CMP_OPS = {
    ast.Eq: _op.eq,
    ast.NotEq: _op.ne,
    ast.Lt: _op.lt,
    ast.LtE: _op.le,
    ast.Gt: _op.gt,
    ast.GtE: _op.ge,
}

_SAFE_BUILTINS: Dict[str, Any] = {
    "print": print,
    "range": range,
    "len": len,
    "abs": abs,
    "min": min,
    "max": max,
    "round": round,
    "bool": bool,
    "int": int,
    "float": float,
    "enumerate": enumerate,
}

_ALLOWED_CALLS = set(_SAFE_BUILTINS.keys())


def evaluate_expression(code: str) -> _Result:
    """
    Evaluate a Python snippet.

    Behavior:
      - If `code` is a simple expression (e.g., "2 + 3"), it is safely evaluated
        and the resulting value is returned (int/float/bool).
      - Otherwise, `code` is treated as a small code snippet (with conditionals/loops).
        It is validated, executed in a restricted environment, and any text printed to
        stdout is captured and returned as a string.

    Safety:
      - Expression evaluation only supports numbers, basic arithmetic, comparisons,
        boolean ops, and parentheses.
      - Statement execution allows a limited subset of Python: if/for/while, simple
        assignments, break/continue/pass, and calls to a small whitelist of safe
        built-ins (e.g., print, range).
    """
    if not isinstance(code, str):
        raise TypeError("code must be a string")

    # Try expression mode first for backwards compatibility
    try:
        expr_tree = ast.parse(code, mode="eval")
        try:
            return _eval_node(expr_tree.body)
        except ValueError:
            # Fall back to exec mode if the expression contains disallowed constructs
            pass
    except SyntaxError:
        # Not an expression — proceed to exec mode
        pass

    # Exec mode: validate and run in a restricted environment while capturing stdout
    try:
        exec_tree = ast.parse(code, mode="exec")
    except SyntaxError as e:
        raise ValueError("Invalid code") from e

    _validate_exec_tree(exec_tree)

    stdout_buffer = io.StringIO()
    safe_globals = {"__builtins__": _SAFE_BUILTINS.copy()}
    safe_locals: Dict[str, Any] = {}

    with redirect_stdout(stdout_buffer):
        exec(compile(exec_tree, filename="<user_code>", mode="exec"), safe_globals, safe_locals)

    return stdout_buffer.getvalue()


def _eval_node(node) -> _Result:
    # Numeric/boolean literal (py3.8+: Constant; older: Num)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float, bool)):
            return node.value
        raise ValueError("Only int, float, and bool literals are allowed")
    if isinstance(node, ast.Num):  # pragma: no cover - for very old Python
        if isinstance(node.n, (int, float)):
            return node.n
        raise ValueError("Only integer and float literals are allowed")

    # Parentheses represented by inner node directly.

    # Unary operations (+x, -x, not x)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        operand = _eval_node(node.operand)
        return _UNARY_OPS[type(node.op)](operand)

    # Binary operations (x + y, x * y, etc.)
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        return _BIN_OPS[type(node.op)](left, right)

    # Boolean operations (and/or) -> return boolean
    if isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
        if isinstance(node.op, ast.And):
            result = True
            for value in node.values:
                result = bool(_eval_node(value))
                if not result:
                    return False
            return True
        else:  # ast.Or
            for value in node.values:
                if bool(_eval_node(value)):
                    return True
            return False

    # Comparisons (x < y, x == y, etc.) with chaining support
    if isinstance(node, ast.Compare):
        left = _eval_node(node.left)
        for op, comparator in zip(node.ops, node.comparators):
            right = _eval_node(comparator)
            cmp_func = _CMP_OPS.get(type(op))
            if cmp_func is None:
                raise ValueError("Disallowed comparison operator")
            if not cmp_func(left, right):
                return False
            left = right
        return True

    # Names are not allowed in pure expression evaluation (prevents variable access)
    if isinstance(node, ast.Name):
        raise ValueError("Names are not allowed in expressions")

    # Calls, attributes, subscripts, etc. are disallowed in expressions
    raise ValueError(f"Disallowed expression: {type(node).__name__}")


# -------- Exec-mode validator (for statements) --------

def _validate_exec_tree(tree: ast.AST) -> None:
    if not isinstance(tree, ast.Module):
        raise ValueError("Invalid code container")
    for stmt in tree.body:
        _validate_stmt(stmt)


def _validate_stmt(node: ast.AST) -> None:
    if isinstance(node, ast.Expr):
        _validate_expr(node.value)
        return

    if isinstance(node, ast.Assign):
        # Support simple assignments to names: x = <expr>
        if not node.targets:
            raise ValueError("Invalid assignment")
        for tgt in node.targets:
            if not isinstance(tgt, ast.Name) or not isinstance(tgt.ctx, ast.Store):
                raise ValueError("Only simple variable assignments are allowed")
            _validate_identifier(tgt.id)
        _validate_expr(node.value)
        return

    if isinstance(node, ast.AugAssign):
        if not isinstance(node.target, ast.Name) or not isinstance(node.target.ctx, ast.Store):
            raise ValueError("Only simple variable augmented assignments are allowed")
        if type(node.op) not in _BIN_OPS:
            raise ValueError("Disallowed augmented operator")
        _validate_expr(node.value)
        return

    if isinstance(node, ast.For):
        # for <name> in <iter>: <body> [else: <orelse>]
        if not isinstance(node.target, ast.Name) or not isinstance(node.target.ctx, ast.Store):
            raise ValueError("Only simple for-loop targets are allowed")
        _validate_identifier(node.target.id)
        _validate_expr(node.iter)
        for s in node.body:
            _validate_stmt(s)
        for s in node.orelse:
            _validate_stmt(s)
        return

    if isinstance(node, ast.While):
        _validate_expr(node.test)
        for s in node.body:
            _validate_stmt(s)
        for s in node.orelse:
            _validate_stmt(s)
        return

    if isinstance(node, ast.If):
        _validate_expr(node.test)
        for s in node.body:
            _validate_stmt(s)
        for s in node.orelse:
            _validate_stmt(s)
        return

    if isinstance(node, (ast.Break, ast.Continue, ast.Pass)):
        return

    # Disallow function/class definitions, imports, with/try, etc.
    raise ValueError(f"Disallowed statement: {type(node).__name__}")


def _validate_expr(node: ast.AST) -> None:
    if isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float, bool, str)):
            raise ValueError("Only int, float, bool, and str constants are allowed")
        return

    # Support very simple tuple/list literals of allowed elements
    if isinstance(node, (ast.Tuple, ast.List)):
        for elt in node.elts:
            _validate_expr(elt)
        return

    if isinstance(node, ast.Name):
        _validate_identifier(node.id)
        return

    if isinstance(node, ast.UnaryOp):
        if type(node.op) not in _UNARY_OPS:
            raise ValueError("Disallowed unary operator")
        _validate_expr(node.operand)
        return

    if isinstance(node, ast.BinOp):
        if type(node.op) not in _BIN_OPS:
            raise ValueError("Disallowed binary operator")
        _validate_expr(node.left)
        _validate_expr(node.right)
        return

    if isinstance(node, ast.BoolOp):
        if not isinstance(node.op, (ast.And, ast.Or)):
            raise ValueError("Disallowed boolean operator")
        for v in node.values:
            _validate_expr(v)
        return

    if isinstance(node, ast.Compare):
        for op in node.ops:
            if type(op) not in _CMP_OPS:
                raise ValueError("Disallowed comparison operator")
        _validate_expr(node.left)
        for comp in node.comparators:
            _validate_expr(comp)
        return

    if isinstance(node, ast.Call):
        # Only allow calls to whitelisted simple names
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only calls to simple names are allowed")
        if node.func.id not in _ALLOWED_CALLS:
            raise ValueError(f"Call to disallowed function: {node.func.id}")
        for arg in node.args:
            _validate_expr(arg)
        for kw in node.keywords:
            if kw.arg is not None:
                _validate_identifier(kw.arg)
            _validate_expr(kw.value)
        return

    # Disallow attribute access, subscripts, dict/set literals, lambdas, comprehensions, etc.
    raise ValueError(f"Disallowed expression: {type(node).__name__}")


def _validate_identifier(name: str) -> None:
    if not isinstance(name, str) or not name:
        raise ValueError("Invalid identifier")
    # Disallow dunder/special names
    if name.startswith("__") and name.endswith("__"):
        raise ValueError("Use of special names is not allowed")
