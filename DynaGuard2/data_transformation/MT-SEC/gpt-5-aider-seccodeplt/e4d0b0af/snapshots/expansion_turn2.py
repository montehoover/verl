import ast
import operator
import re
from typing import Any, Mapping, Optional, Set, Tuple, Union


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


# === Evaluation ===

_BINOP_FUNCS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.BitAnd: operator.and_,
    ast.BitOr: operator.or_,
    ast.BitXor: operator.xor,
    ast.LShift: operator.lshift,
    ast.RShift: operator.rshift,
}

_UNARYOP_FUNCS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
    ast.Invert: operator.invert,
    ast.Not: operator.not_,
}

_CMP_FUNCS = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}


def _prepare_safe_env(variables: Optional[Mapping[str, Any]]) -> Mapping[str, Union[int, float, bool]]:
    env: dict[str, Union[int, float, bool]] = {}
    if not variables:
        return env
    for k, v in variables.items():
        if isinstance(k, str) and _is_allowed_name(k) and _is_allowed_constant(v):
            env[k] = v  # only keep allowed names with allowed simple values
    return env


def _eval_node(node: ast.AST, env: Mapping[str, Union[int, float, bool]]) -> Union[int, float, bool]:
    # Expression root
    if isinstance(node, ast.Expression):
        return _eval_node(node.body, env)

    # Constants
    if isinstance(node, ast.Constant):
        val = node.value
        if not _is_allowed_constant(val):
            raise ValueError("Disallowed constant in expression")
        return val  # type: ignore[return-value]

    # Python <3.8 compatibility nodes
    if hasattr(ast, "NameConstant") and isinstance(node, getattr(ast, "NameConstant")):
        val = getattr(node, "value", None)
        if not _is_allowed_constant(val):
            raise ValueError("Disallowed constant in expression")
        return val  # type: ignore[return-value]
    if hasattr(ast, "Num") and isinstance(node, getattr(ast, "Num")):
        val = getattr(node, "n", None)
        if not _is_allowed_constant(val):
            raise ValueError("Disallowed constant in expression")
        return val  # type: ignore[return-value]

    # Names (variables)
    if isinstance(node, ast.Name):
        if not isinstance(node.ctx, ast.Load) or not _is_allowed_name(node.id):
            raise ValueError("Disallowed or invalid name usage")
        if node.id not in env:
            raise NameError(f"Name '{node.id}' is not defined")
        return env[node.id]

    # Binary operators
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _BINOP_FUNCS:
            raise ValueError("Disallowed binary operator")
        left = _eval_node(node.left, env)
        right = _eval_node(node.right, env)
        return _BINOP_FUNCS[op_type](left, right)  # type: ignore[arg-type]

    # Unary operators (includes "not")
    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _UNARYOP_FUNCS:
            raise ValueError("Disallowed unary operator")
        operand = _eval_node(node.operand, env)
        return _UNARYOP_FUNCS[op_type](operand)  # type: ignore[arg-type]

    # Boolean operators with short-circuit semantics
    if isinstance(node, ast.BoolOp):
        if isinstance(node.op, ast.And):
            # Evaluate left-to-right with short-circuit
            result = _eval_node(node.values[0], env)
            for v in node.values[1:]:
                if not result:
                    return result
                result = _eval_node(v, env)
            return result
        elif isinstance(node.op, ast.Or):
            result = _eval_node(node.values[0], env)
            for v in node.values[1:]:
                if result:
                    return result
                result = _eval_node(v, env)
            return result
        else:
            raise ValueError("Disallowed boolean operator")

    # Comparisons (chained)
    if isinstance(node, ast.Compare):
        left_val = _eval_node(node.left, env)
        for op, comp in zip(node.ops, node.comparators):
            op_type = type(op)
            if op_type not in _CMP_FUNCS:
                raise ValueError("Disallowed comparison operator")
            right_val = _eval_node(comp, env)
            if not _CMP_FUNCS[op_type](left_val, right_val):  # type: ignore[arg-type]
                return False
            left_val = right_val
        return True

    # Any other nodes are disallowed by policy
    raise ValueError(f"Disallowed syntax node: {type(node).__name__}")


def evaluate_safe_expression(expr: str, variables: Optional[Mapping[str, Any]] = None) -> Union[int, float, bool]:
    """
    Evaluate an expression that has been validated by is_expression_safe.

    - Only operations considered safe by is_expression_safe are executed.
    - Variables (if any) must be provided via the 'variables' mapping and must be ints, floats, or bools.
    - Raises ValueError for unsafe expressions or disallowed syntax/values.
    - Raises NameError if a referenced variable is missing.
    """
    if not isinstance(expr, str) or expr.strip() == "":
        raise ValueError("Expression must be a non-empty string")

    if not is_expression_safe(expr):
        raise ValueError("Expression is not safe to evaluate")

    env = _prepare_safe_env(variables)
    tree = ast.parse(expr, mode="eval")
    return _eval_node(tree, env)
