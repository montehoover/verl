import ast
import math
from typing import Any, Dict


def _build_allowed_env() -> Dict[str, Any]:
    """
    Build a whitelist of safe names (functions and constants) that are allowed
    to be referenced by user-provided math expressions.

    The environment includes:
    - A curated subset of functions from the math module (when available)
    - Common math constants (pi, e, tau, inf, nan)
    - Safe built-ins like abs and round
    """
    allowed: Dict[str, Any] = {}

    # Allow common math functions (add only if present on this Python version)
    function_names = [
        "sin", "cos", "tan",
        "asin", "acos", "atan", "atan2",
        "sinh", "cosh", "tanh",
        "asinh", "acosh", "atanh",
        "exp", "expm1",
        "log", "log10", "log1p", "log2",
        "sqrt", "hypot",
        "degrees", "radians",
        "ceil", "floor", "trunc",
        "fabs", "fmod", "remainder",
        "gcd", "lcm",
        "isfinite", "isinf", "isnan",
        "gamma", "lgamma", "erf", "erfc",
        "factorial", "comb", "perm",
        "fsum", "prod",
    ]
    for name in function_names:
        fn = getattr(math, name, None)
        if fn is not None:
            allowed[name] = fn

    # Allow common math constants
    constant_names = ["pi", "e", "tau", "inf", "nan"]
    for name in constant_names:
        val = getattr(math, name, None)
        if val is not None:
            allowed[name] = val

    # Safe built-ins
    allowed["abs"] = abs
    allowed["round"] = round

    return allowed


# Whitelisted operators that are safe for numeric expressions
_ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)
_ALLOWED_UNARYOPS = (ast.UAdd, ast.USub)


def _is_safe_ast(node: ast.AST, allowed_names: Dict[str, Any]) -> bool:
    """
    Recursively validate that the AST only contains safe numeric operations.

    Security policy:
    - Only numeric literals are allowed (no strings, bytes, bools, etc.)
    - Arithmetic operations are limited to +, -, *, /, //, %, ** and unary +/-.
    - Names must reference entries in the allowed environment.
    - Function calls are only permitted for whitelisted names; attribute access,
      keyword splats (**kwargs), starargs, and other complex constructs are forbidden.
    - All other node types (attributes, subscripts, comprehensions, conditionals,
      lambdas, statements, etc.) are disallowed.
    """
    if isinstance(node, ast.Expression):
        return _is_safe_ast(node.body, allowed_names)

    if isinstance(node, ast.Constant):
        # Only numeric constants; explicitly reject bools and other types
        return isinstance(node.value, (int, float, complex)) and not isinstance(node.value, bool)

    if isinstance(node, ast.BinOp):
        return (
            isinstance(node.op, _ALLOWED_BINOPS) and
            _is_safe_ast(node.left, allowed_names) and
            _is_safe_ast(node.right, allowed_names)
        )

    if isinstance(node, ast.UnaryOp):
        return isinstance(node.op, _ALLOWED_UNARYOPS) and _is_safe_ast(node.operand, allowed_names)

    if isinstance(node, ast.Name):
        # Only allow names from our safe environment
        return node.id in allowed_names

    if isinstance(node, ast.Call):
        # Only allow calling safe names; no attribute access, no **kwargs
        if not isinstance(node.func, ast.Name):
            return False
        func_name = node.func.id
        if func_name not in allowed_names or not callable(allowed_names[func_name]):
            return False
        # Validate all args and keyword values; disallow **kwargs (kw.arg is None)
        for arg in node.args:
            if not _is_safe_ast(arg, allowed_names):
                return False
        for kw in node.keywords:
            if kw.arg is None:
                return False
            if not _is_safe_ast(kw.value, allowed_names):
                return False
        return True

    # Disallow everything else: attributes, subscripts, comprehensions, conditionals, etc.
    return False


def _parse_math_expression(math_input: str) -> ast.Expression:
    """
    Parse a user-provided string strictly as a Python expression.

    Returns:
        ast.Expression: The parsed AST for the expression.

    Raises:
        SyntaxError: If the input cannot be parsed in 'eval' mode.
        TypeError: If the input is not a string.
    """
    if not isinstance(math_input, str):
        raise TypeError("Input must be a string.")
    return ast.parse(math_input, mode='eval')  # type: ignore[return-value]


def is_expression_safe(tree: ast.AST, allowed_env: Dict[str, Any]) -> bool:
    """
    Public wrapper to check if an AST is safe under the defined policy.

    Returns:
        bool: True if the AST satisfies all security constraints, else False.
    """
    return _is_safe_ast(tree, allowed_env)


def evaluate_safe_expression(tree: ast.AST, allowed_env: Dict[str, Any]) -> str:
    """
    Evaluate a previously validated AST safely.

    Execution is done with:
    - An empty builtins dict to prevent access to Python built-ins.
    - A restricted local environment containing only allowed names.

    Returns:
        str: The stringified result of the evaluation.

    Raises:
        Exception: If compilation or evaluation fails for any reason.
    """
    compiled = compile(tree, '<expr>', 'eval')
    result = eval(compiled, {'__builtins__': {}}, allowed_env)
    return str(result)


def math_expression_safety_check(math_input: str) -> str:
    """
    Safely evaluate a Python mathematical expression string.

    Returns the stringified result of the evaluation, or 'Unsafe Code Detected!'
    if any unsafe content is found or evaluation fails.
    """
    try:
        # Parse the input strictly as an expression
        tree = _parse_math_expression(math_input)

        # Build the allowed environment and validate the AST against our whitelist
        allowed_env = _build_allowed_env()
        if not is_expression_safe(tree, allowed_env):
            return 'Unsafe Code Detected!'

        # Evaluate the safe expression within the restricted environment
        return evaluate_safe_expression(tree, allowed_env)
    except Exception:
        return 'Unsafe Code Detected!'
