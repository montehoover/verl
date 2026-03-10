import ast
import math
from typing import Union

# Safety limits to prevent resource exhaustion
_MAX_EXPR_LENGTH = 1000
_MAX_AST_NODES = 1000
_MAX_RECURSION_DEPTH = 200
_MAX_POWER_EXPONENT = 10000  # absolute value limit for exponent in power operations

_Number = Union[int, float]


def _is_allowed_constant(value) -> bool:
    # Allow only int and float (reject bool, complex, strings, etc.)
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return True
    return False


def _is_safe_ast(tree: ast.AST) -> bool:
    # Validate that only a restricted set of nodes/operators are present
    for node in ast.walk(tree):
        if isinstance(node, ast.Expression):
            continue
        elif isinstance(node, ast.BinOp):
            if not isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)):
                return False
        elif isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, (ast.UAdd, ast.USub)):
                return False
        elif isinstance(node, ast.Constant):
            if not _is_allowed_constant(node.value):
                return False
        # Explicitly reject all other node types
        elif isinstance(node, (ast.Load,)):
            # Load context can appear but is harmless only if there are no Names; still allow Load
            continue
        else:
            return False
    return True


def _eval_ast(node: ast.AST, depth: int = 0) -> _Number:
    if depth > _MAX_RECURSION_DEPTH:
        raise ValueError("Too deep")

    if isinstance(node, ast.Constant):
        value = node.value
        if not _is_allowed_constant(value):
            raise ValueError("Disallowed constant")
        return value  # type: ignore[return-value]

    if isinstance(node, ast.UnaryOp):
        operand = _eval_ast(node.operand, depth + 1)
        if not isinstance(operand, (int, float)):
            raise ValueError("Non-numeric operand")
        if isinstance(node.op, ast.UAdd):
            return +operand
        if isinstance(node.op, ast.USub):
            return -operand
        raise ValueError("Disallowed unary operator")

    if isinstance(node, ast.BinOp):
        left = _eval_ast(node.left, depth + 1)
        right = _eval_ast(node.right, depth + 1)
        if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
            raise ValueError("Non-numeric operand")

        op = node.op
        try:
            if isinstance(op, ast.Add):
                return left + right
            if isinstance(op, ast.Sub):
                return left - right
            if isinstance(op, ast.Mult):
                return left * right
            if isinstance(op, ast.Div):
                return left / right
            if isinstance(op, ast.FloorDiv):
                return left // right
            if isinstance(op, ast.Mod):
                return left % right
            if isinstance(op, ast.Pow):
                # Basic guard against massive exponentiation
                if isinstance(right, (int, float)) and abs(right) > _MAX_POWER_EXPONENT:
                    raise ValueError("Exponent too large")
                return left ** right
        except Exception as e:
            raise ValueError(str(e))

        raise ValueError("Disallowed binary operator")

    # If it's an Expression node, evaluate its body
    if isinstance(node, ast.Expression):
        return _eval_ast(node.body, depth + 1)

    raise ValueError("Disallowed syntax")


def evaluate_expression(expr_string: str) -> str:
    """
    Evaluate a basic arithmetic expression given as a Python-syntax string.
    Returns the result as a string, or 'Computation Error!' on any failure
    or if suspicious/unsafe input is detected.
    """
    try:
        if not isinstance(expr_string, str):
            return "Computation Error!"
        s = expr_string.strip()
        if not s:
            return "Computation Error!"
        if len(s) > _MAX_EXPR_LENGTH:
            return "Computation Error!"

        # Parse expression
        tree = ast.parse(s, mode="eval")

        # Basic structural limits
        if sum(1 for _ in ast.walk(tree)) > _MAX_AST_NODES:
            return "Computation Error!"

        # Validate AST for safety
        if not _is_safe_ast(tree):
            return "Computation Error!"

        # Evaluate safely
        result = _eval_ast(tree.body, 0)

        # Ensure result is numeric and finite if float
        if isinstance(result, float) and not math.isfinite(result):
            return "Computation Error!"
        if not isinstance(result, (int, float)):
            return "Computation Error!"

        return str(result)
    except Exception:
        return "Computation Error!"
