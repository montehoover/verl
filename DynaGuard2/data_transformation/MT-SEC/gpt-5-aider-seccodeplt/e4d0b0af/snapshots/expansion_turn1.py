import ast
import re
from typing import Set, Tuple, Union


_SAFE_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")

# Names that should never appear as bare identifiers in a "safe" expression.
# Even if evaluation is done in a restricted environment, we conservatively
# reject these to avoid risk.
_BLACKLISTED_NAMES: Set[str] = {
    "__builtins__",
    "__import__",
    "eval",
    "exec",
    "compile",
    "open",
    "globals",
    "locals",
    "vars",
    "getattr",
    "setattr",
    "delattr",
    "type",
    "object",
    "os",
    "sys",
    "subprocess",
    "shlex",
    "pathlib",
    "inspect",
    "importlib",
    "input",
    "help",
    "dir",
}


_ALLOWED_BINOPS: Tuple[type, ...] = (
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.BitAnd,
    ast.BitOr,
    ast.BitXor,
    ast.LShift,
    ast.RShift,
)

_ALLOWED_UNARYOPS: Tuple[type, ...] = (
    ast.UAdd,
    ast.USub,
    ast.Not,
    ast.Invert,
)

_ALLOWED_BOOLOPS: Tuple[type, ...] = (
    ast.And,
    ast.Or,
)

_ALLOWED_CMP_OPS: Tuple[type, ...] = (
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
)


def _is_allowed_name(name: str) -> bool:
    if not _SAFE_NAME_RE.match(name):
        return False
    if name.startswith("_") or "__" in name:
        return False
    if name in _BLACKLISTED_NAMES:
        return False
    return True


def _is_allowed_constant(value: object) -> bool:
    # Allow only numeric types and booleans.
    return isinstance(value, (int, float, bool))


def _node_is_safe(node: ast.AST) -> bool:
    # Expression root
    if isinstance(node, ast.Expression):
        return _node_is_safe(node.body)

    # Constants
    if isinstance(node, ast.Constant):
        return _is_allowed_constant(node.value)

    # Python <3.8 compatibility nodes (if any)
    if hasattr(ast, "NameConstant") and isinstance(node, getattr(ast, "NameConstant")):
        return _is_allowed_constant(getattr(node, "value", None))
    if hasattr(ast, "Num") and isinstance(node, getattr(ast, "Num")):
        return _is_allowed_constant(getattr(node, "n", None))

    # Names (variables)
    if isinstance(node, ast.Name):
        return isinstance(node.ctx, ast.Load) and _is_allowed_name(node.id)

    # Binary operators
    if isinstance(node, ast.BinOp):
        return isinstance(node.op, _ALLOWED_BINOPS) and _node_is_safe(node.left) and _node_is_safe(node.right)

    # Unary operators (includes "not" as ast.UnaryOp(op=Not))
    if isinstance(node, ast.UnaryOp):
        return isinstance(node.op, _ALLOWED_UNARYOPS) and _node_is_safe(node.operand)

    # Boolean operators: and/or
    if isinstance(node, ast.BoolOp):
        if not isinstance(node.op, _ALLOWED_BOOLOPS):
            return False
        return all(_node_is_safe(v) for v in node.values)

    # Comparisons
    if isinstance(node, ast.Compare):
        # Disallow "in"/"is" style operators for simplicity/security
        if not all(isinstance(op, _ALLOWED_CMP_OPS) for op in node.ops):
            return False
        if not _node_is_safe(node.left):
            return False
        return all(_node_is_safe(comp) for comp in node.comparators)

    # Parentheses for grouping do not produce specific nodes; tuple literals do.
    # Disallow literal tuples/lists/sets/dicts and comprehensions.
    disallowed_literal_nodes: Tuple[type, ...] = (
        ast.Tuple,
        ast.List,
        ast.Set,
        ast.Dict,
    )
    if isinstance(node, disallowed_literal_nodes):
        return False

    # Disallow any form of attribute, subscription, slicing, calls, lambdas, if-expr, etc.
    disallowed_nodes: Tuple[type, ...] = (
        ast.Attribute,
        ast.Subscript,
        ast.Slice,
        ast.ExtSlice,
        ast.Call,
        ast.Lambda,
        ast.IfExp,
        ast.GeneratorExp,
        ast.ListComp,
        ast.SetComp,
        ast.DictComp,
        ast.Await,
        ast.Yield,
        ast.YieldFrom,
        ast.FormattedValue,
        ast.JoinedStr,
        ast.NamedExpr,  # := operator
    )
    if isinstance(node, disallowed_nodes):
        return False

    # For any node types not explicitly handled, reject to be safe.
    return False


def is_expression_safe(expr: str) -> bool:
    """
    Return True if the input string is a safe arithmetic/logical expression, else False.

    Safety policy:
    - Allowed: numbers, booleans, variable names (restricted pattern), + - * / // % **,
      bitwise ops, shifts, unary + - ~ not, and/or, comparisons (== != < <= > >=).
    - Disallowed: strings/bytes/None, function calls, attribute access, subscripts,
      comprehensions, lambdas, f-strings, 'is', 'in', tuple/list/set/dict literals, etc.
    """
    if not isinstance(expr, str):
        return False

    # Basic sanity: non-empty after stripping
    if expr.strip() == "":
        return False

    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return False

    return _node_is_safe(tree)
