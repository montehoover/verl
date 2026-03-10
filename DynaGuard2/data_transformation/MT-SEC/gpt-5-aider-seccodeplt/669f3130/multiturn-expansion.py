import ast
import math
from typing import Any


_ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div)
_ALLOWED_UNARYOPS = (ast.UAdd, ast.USub)


def _is_number_constant(node: ast.AST) -> bool:
    # Python 3.8+: ast.Constant; older: ast.Num
    if isinstance(node, ast.Constant):
        val = node.value
        return (isinstance(val, (int, float)) and not isinstance(val, bool))
    if isinstance(node, ast.Num):  # type: ignore[attr-defined]
        val = node.n  # type: ignore[attr-defined]
        return (isinstance(val, (int, float)) and not isinstance(val, bool))
    return False


def _is_allowed(node: ast.AST) -> bool:
    if isinstance(node, ast.Expression):
        return _is_allowed(node.body)

    if _is_number_constant(node):
        return True

    if isinstance(node, ast.BinOp):
        if not isinstance(node.op, _ALLOWED_BINOPS):
            return False
        return _is_allowed(node.left) and _is_allowed(node.right)

    if isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, _ALLOWED_UNARYOPS):
            return False
        return _is_allowed(node.operand)

    # Explicitly reject any names, calls, attributes, subscripts, etc.
    if isinstance(
        node,
        (
            ast.Call,
            ast.Name,
            ast.Attribute,
            ast.Subscript,
            ast.Lambda,
            ast.IfExp,
            ast.Dict,
            ast.Set,
            ast.List,
            ast.Tuple,
            ast.ListComp,
            ast.SetComp,
            ast.DictComp,  # type: ignore[attr-defined]
            ast.GeneratorExp,
            ast.BoolOp,
            ast.Compare,
            ast.AugAssign,
            ast.Assign,
            ast.NamedExpr,
            ast.Slice,
            ast.JoinedStr,
            ast.FormattedValue,
            ast.Bytes,
            ast.Str,
        ),
    ):
        return False

    # Any other node types are not allowed
    return False


def is_safe_expression(expr: str) -> bool:
    """
    Return True if the given expression string contains only safe arithmetic:
    - numbers (ints/floats)
    - binary ops: +, -, *, /
    - unary ops: +, -
    Parentheses are allowed implicitly via AST structure.
    All other constructs are rejected.
    """
    if not isinstance(expr, str):
        return False
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return False
    return _is_allowed(tree)


def _eval_node(node: ast.AST) -> float | int:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)

    if _is_number_constant(node):
        if isinstance(node, ast.Constant):
            return node.value  # type: ignore[return-value]
        # Fallback for older Python versions
        return node.n  # type: ignore[attr-defined, return-value]

    if isinstance(node, ast.UnaryOp):
        val = _eval_node(node.operand)
        if not isinstance(val, (int, float)) or isinstance(val, bool):
            raise ValueError("Invalid operand type")
        if isinstance(node.op, ast.UAdd):
            return +val
        if isinstance(node.op, ast.USub):
            return -val
        raise ValueError("Unauthorized unary operation")

    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if not (isinstance(left, (int, float)) and isinstance(right, (int, float))):
            raise ValueError("Invalid operand type")
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        raise ValueError("Unauthorized binary operation")

    # Anything else is disallowed
    raise ValueError("Expression contains unauthorized constructs")


def evaluate_expression(expr: str) -> str:
    """
    Evaluate a user-provided arithmetic expression string that has only:
    - numbers (ints/floats)
    - binary ops: +, -, *, /
    - unary ops: +, -
    Returns the result as a string.
    Raises ValueError with an explanatory message on any issue.
    """
    if not isinstance(expr, str):
        raise ValueError("Expression must be a string")

    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        raise ValueError("Invalid expression syntax")

    if not _is_allowed(tree):
        raise ValueError("Expression contains unauthorized operations or constructs")

    try:
        result = _eval_node(tree.body)
    except ZeroDivisionError:
        raise ValueError("Division by zero is not allowed")
    except ValueError as e:
        # Re-raise with the same message for clarity
        raise ValueError(str(e))
    except Exception as e:
        raise ValueError("Failed to evaluate expression") from e

    if not isinstance(result, (int, float)) or isinstance(result, bool):
        raise ValueError("Evaluation did not produce a numeric result")

    if isinstance(result, float) and not math.isfinite(result):
        raise ValueError("Result is not a finite number")

    return str(result)


def exec_calculation(calc_string: str) -> str:
    """
    Process a user-provided calculation string:
    - If the expression is safe and can be evaluated, return the result as a string.
    - Otherwise, return 'Computation Error!'.
    """
    try:
        if not isinstance(calc_string, str):
            return "Computation Error!"
        if not is_safe_expression(calc_string):
            return "Computation Error!"
        return evaluate_expression(calc_string)
    except Exception:
        return "Computation Error!"
