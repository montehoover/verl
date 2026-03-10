import re
import ast

# Supported operators mapping for evaluation
_BIN_OPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
    ast.Pow: lambda a, b: a ** b,
}

_UNARY_OPS = {
    ast.UAdd: lambda a: +a,
    ast.USub: lambda a: -a,
}

_UNSAFE_NODES = (
    ast.Call,
    ast.Attribute,
    ast.Subscript,
    ast.Name,
    ast.Lambda,
    ast.IfExp,
    ast.Compare,
    ast.BoolOp,
    ast.Dict,
    ast.List,
    ast.Set,
    ast.Tuple,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp if hasattr(ast, "DictComp") else tuple(),  # older versions may not have DictComp
    ast.GeneratorExp,
    ast.Await,
    ast.Yield,
    ast.YieldFrom,
    ast.Assign,
    ast.AugAssign,
    ast.NamedExpr,
    ast.With,
    ast.Import,
    ast.ImportFrom,
    ast.Try,
    ast.For,
    ast.While,
    ast.FunctionDef,
    ast.ClassDef,
)

def compute_expression(input_expr: str):
    """
    Safely compute a simple arithmetic expression from a user-provided string.

    Args:
        input_expr (str): The arithmetic expression to evaluate.

    Returns:
        The numeric result of the expression (int or float).

    Raises:
        ValueError: If unsupported characters, unsafe commands, or invalid operations are detected.
    """
    if not isinstance(input_expr, str):
        raise ValueError("Expression must be a string.")
    expr = input_expr.strip()
    if not expr:
        raise ValueError("Empty expression is not allowed.")

    # Parse expression into an AST in eval mode
    try:
        node = ast.parse(expr, mode="eval")
    except SyntaxError:
        raise ValueError("Invalid expression syntax.")

    # Walk the AST to reject any unsafe or unsupported constructs early
    for sub in ast.walk(node):
        # Disallow unsafe node types
        if isinstance(sub, _UNSAFE_NODES):
            raise ValueError("Unsupported or unsafe expression element detected.")

        # Disallow any constants that are not plain numbers (int/float)
        if isinstance(sub, ast.Constant):
            val = sub.value
            # Disallow booleans, strings, bytes, complex numbers, None, Ellipsis, etc.
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise ValueError("Unsupported literal in expression.")
        # For Python versions <3.8, numeric literals may appear as ast.Num
        if hasattr(ast, "Num") and isinstance(sub, ast.Num):
            # ast.Num is numeric; nothing to do here, just acknowledge
            pass
        # Disallow any operator types not in our allowed sets
        if isinstance(sub, ast.BinOp) and type(sub.op) not in _BIN_OPS:
            raise ValueError("Unsupported operator in expression.")
        if isinstance(sub, ast.UnaryOp) and type(sub.op) not in _UNARY_OPS:
            raise ValueError("Unsupported unary operator in expression.")

    def _eval(n):
        if isinstance(n, ast.Expression):
            return _eval(n.body)
        if isinstance(n, ast.BinOp):
            left = _eval(n.left)
            right = _eval(n.right)
            op_type = type(n.op)
            if op_type not in _BIN_OPS:
                raise ValueError("Unsupported operator in expression.")
            # Guard against division/modulo by zero
            if op_type in (ast.Div, ast.FloorDiv, ast.Mod) and right == 0:
                raise ValueError("Division or modulo by zero.")
            try:
                return _BIN_OPS[op_type](left, right)
            except OverflowError as e:
                raise ValueError("Numeric overflow in computation.") from e
            except ZeroDivisionError as e:
                # Extra guard in case of float quirks
                raise ValueError("Division or modulo by zero.") from e
        if isinstance(n, ast.UnaryOp):
            op_type = type(n.op)
            if op_type not in _UNARY_OPS:
                raise ValueError("Unsupported unary operator in expression.")
            val = _eval(n.operand)
            try:
                return _UNARY_OPS[op_type](val)
            except OverflowError as e:
                raise ValueError("Numeric overflow in computation.") from e
        # Numeric literals
        if isinstance(n, ast.Constant):
            val = n.value
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise ValueError("Unsupported literal in expression.")
            return val
        if hasattr(ast, "Num") and isinstance(n, ast.Num):
            return n.n
        # Parentheses are handled implicitly by AST structure; no separate node exists.

        # Any other node is not supported
        raise ValueError(f"Unsupported expression element: {type(n).__name__}")

    return _eval(node)
