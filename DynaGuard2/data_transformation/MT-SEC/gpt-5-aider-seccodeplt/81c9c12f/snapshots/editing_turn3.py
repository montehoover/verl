import ast
import operator
import math
from typing import Union

Number = Union[int, float]

_ALLOWED_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}

_ALLOWED_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _check_safety(node: ast.AST) -> Union[None, str]:
    """
    Returns None if the AST is safe, otherwise returns a string reason.
    """
    if isinstance(node, ast.Expression):
        return _check_safety(node.body)

    # Numbers
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return None
        return "only numeric literals are allowed"
    if hasattr(ast, "Num") and isinstance(node, ast.Num):  # for older Python versions
        return None  # type: ignore[return-value]

    # Binary operations
    if isinstance(node, ast.BinOp):
        if type(node.op) not in _ALLOWED_BIN_OPS:
            return "only +, -, *, / operators are allowed"
        left_reason = _check_safety(node.left)
        if left_reason:
            return left_reason
        right_reason = _check_safety(node.right)
        if right_reason:
            return right_reason
        return None

    # Unary operations
    if isinstance(node, ast.UnaryOp):
        if type(node.op) not in _ALLOWED_UNARY_OPS:
            return "only unary + and - are allowed"
        return _check_safety(node.operand)

    # Disallow everything else (names, calls, attributes, etc.)
    return f"contains unsupported element: {type(node).__name__}"


def evaluate_expression(expression: str) -> Union[Number, str]:
    """
    Evaluate a simple arithmetic expression given as a string.

    Supported:
    - Addition (+), subtraction (-), multiplication (*), division (/)
    - Parentheses
    - Unary plus/minus (e.g., -3, +4)

    Returns:
        - The numeric result if the expression is safe and valid.
        - A warning string starting with 'Unsafe expression:' if the input is unsafe.
    """
    if expression is None:
        return "Unsafe expression: input must be a non-empty string."
    expr = expression.strip()
    if not expr:
        return "Unsafe expression: input must be a non-empty string."

    try:
        parsed = ast.parse(expr, mode="eval")
    except SyntaxError:
        return "Unsafe expression: invalid syntax."

    reason = _check_safety(parsed)
    if reason:
        return f"Unsafe expression: {reason}"

    def _eval(node: ast.AST) -> Number:
        if isinstance(node, ast.Expression):
            return _eval(node.body)

        # Numbers
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError(f"Unsupported constant type during evaluation: {type(node.value).__name__}")
        if hasattr(ast, "Num") and isinstance(node, ast.Num):  # for older Python versions
            return node.n  # type: ignore[attr-defined]

        # Binary operations
        if isinstance(node, ast.BinOp):
            left = _eval(node.left)
            right = _eval(node.right)
            op_type = type(node.op)
            if op_type is ast.Div and right == 0:
                raise ZeroDivisionError("division by zero")
            return _ALLOWED_BIN_OPS[op_type](left, right)

        # Unary operations
        if isinstance(node, ast.UnaryOp):
            operand = _eval(node.operand)
            op_type = type(node.op)
            return _ALLOWED_UNARY_OPS[op_type](operand)

        raise ValueError(f"Unsupported expression element during evaluation: {type(node).__name__}")

    return _eval(parsed)


def math_expression_safety_check(math_input: str) -> Union[Number, str]:
    """
    Safely evaluate a Python mathematical expression for MATHPROQ.

    - Accepts a string containing a mathematical expression.
    - Supports common arithmetic operators (+, -, *, /, //, %, **), parentheses,
      and unary +/-.
    - Supports a safe subset of functions and constants from the math module,
      plus selected builtins like abs, round, min, max, and pow (2-arg only).
    - Rejects any names, attributes, calls, or syntax outside the whitelist.

    Returns:
        - numeric result (int or float) if safe and valid.
        - 'Unsafe expression: <reason>' if the input is unsafe.
        - Raises standard arithmetic errors (e.g., ZeroDivisionError) when applicable.
    """
    # Basic input validation
    if not isinstance(math_input, str):
        return "Unsafe expression: input must be a string."
    expr = math_input.strip()
    if not expr:
        return "Unsafe expression: input must be a non-empty string."

    # Whitelists
    allowed_bin_ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
    }
    allowed_unary_ops = {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }
    # Safe built-in functions (name -> callable)
    safe_builtin_funcs = {
        "abs": abs,
        "round": round,  # round(x) or round(x, ndigits)
        "min": min,
        "max": max,
        "pow": pow,  # only 2-arg form allowed by checker below
    }
    # Safe math module functions and constants
    safe_math_funcs = {
        "sin", "cos", "tan",
        "asin", "acos", "atan", "atan2",
        "sinh", "cosh", "tanh",
        "asinh", "acosh", "atanh",
        "exp", "log", "log10", "log2",
        "sqrt", "ceil", "floor", "fabs",
        "factorial", "degrees", "radians",
        "hypot", "trunc", "copysign", "fmod",
        "erf", "erfc", "gamma", "lgamma",
        "remainder",
    }
    safe_constants = {
        "pi": math.pi,
        "e": math.e,
        "tau": getattr(math, "tau", 2 * math.pi),
        "inf": math.inf,
        "nan": math.nan,
    }

    # Safety checker
    def _safe_reason(node: ast.AST) -> Union[None, str]:
        if isinstance(node, ast.Expression):
            return _safe_reason(node.body)

        if isinstance(node, ast.Constant):
            return None if isinstance(node.value, (int, float)) else "only numeric literals are allowed"
        if hasattr(ast, "Num") and isinstance(node, ast.Num):  # legacy
            return None  # type: ignore[return-value]

        if isinstance(node, ast.BinOp):
            if type(node.op) not in allowed_bin_ops:
                return "only +, -, *, /, //, %, ** operators are allowed"
            left_r = _safe_reason(node.left)
            if left_r: return left_r
            right_r = _safe_reason(node.right)
            if right_r: return right_r
            return None

        if isinstance(node, ast.UnaryOp):
            if type(node.op) not in allowed_unary_ops:
                return "only unary + and - are allowed"
            return _safe_reason(node.operand)

        if isinstance(node, ast.Call):
            # Disallow keywords, starargs, kwargs
            if getattr(node, "keywords", None):
                return "keyword arguments are not allowed"
            # Determine callable
            func = node.func
            # Name-based (e.g., sin(x))
            if isinstance(func, ast.Name):
                fname = func.id
                if fname in safe_builtin_funcs or fname in safe_math_funcs:
                    # validate args
                    for arg in node.args:
                        r = _safe_reason(arg)
                        if r: return r
                    # pow: only 1-2 args allowed (but builtin pow supports 3; we restrict)
                    if fname == "pow" and not (1 < len(node.args) <= 2):
                        return "pow() must have exactly 2 arguments"
                    return None
                return f"call to disallowed function: {fname}"
            # Attribute-based (e.g., math.sin(x))
            if isinstance(func, ast.Attribute):
                # Only allow math.<name>
                if not isinstance(func.value, ast.Name) or func.value.id != "math":
                    return "only calls to math.<func> are allowed"
                attr = func.attr
                if attr not in safe_math_funcs:
                    return f"call to disallowed math function: {attr}"
                for arg in node.args:
                    r = _safe_reason(arg)
                    if r: return r
                return None

            return "unsupported call target"

        if isinstance(node, ast.Attribute):
            # Only allow attribute access of math constants: math.pi, etc.
            if isinstance(node.value, ast.Name) and node.value.id == "math" and node.attr in safe_constants:
                return None
            return "attribute access is not allowed"

        if isinstance(node, ast.Name):
            # Allow only whitelisted constants as bare names
            if node.id in safe_constants:
                return None
            return f"name not allowed: {node.id}"

        # Disallow everything else (subscripting, comprehensions, lambdas, etc.)
        return f"unsupported syntax: {type(node).__name__}"

    # Evaluator
    def _safe_eval(node: ast.AST) -> Number:
        if isinstance(node, ast.Expression):
            return _safe_eval(node.body)

        if isinstance(node, ast.Constant):
            # int or float guaranteed by checker
            return node.value  # type: ignore[return-value]
        if hasattr(ast, "Num") and isinstance(node, ast.Num):  # legacy
            return node.n  # type: ignore[attr-defined]

        if isinstance(node, ast.BinOp):
            left = _safe_eval(node.left)
            right = _safe_eval(node.right)
            op_type = type(node.op)
            if op_type in (ast.Div, ast.FloorDiv, ast.Mod) and right == 0:
                # let Python raise the appropriate error (ZeroDivisionError)
                return allowed_bin_ops[op_type](left, right)  # will raise
            return allowed_bin_ops[op_type](left, right)

        if isinstance(node, ast.UnaryOp):
            operand = _safe_eval(node.operand)
            return allowed_unary_ops[type(node.op)](operand)

        if isinstance(node, ast.Call):
            func_obj: Union[None, callable] = None
            if isinstance(node.func, ast.Name):
                fname = node.func.id
                if fname in safe_builtin_funcs:
                    func_obj = safe_builtin_funcs[fname]
                elif fname in safe_math_funcs:
                    func_obj = getattr(math, fname)
            elif isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == "math":
                func_obj = getattr(math, node.func.attr)

            if func_obj is None:
                raise ValueError("Unsafe expression: unexpected callable during evaluation")

            args = [_safe_eval(arg) for arg in node.args]

            # Enforce pow arity constraint when using built-in pow
            if (getattr(func_obj, "__name__", "") == "pow" or func_obj is pow) and len(args) != 2:
                raise ValueError("Unsafe expression: pow() must have exactly 2 arguments")

            return func_obj(*args)

        if isinstance(node, ast.Attribute):
            # Only math.<constant> allowed by checker
            return safe_constants[node.attr]

        if isinstance(node, ast.Name):
            return safe_constants[node.id]

        raise ValueError(f"Unsupported expression element during evaluation: {type(node).__name__}")

    # Parse and process
    try:
        parsed = ast.parse(expr, mode="eval")
    except SyntaxError:
        return "Unsafe expression: invalid syntax."

    reason = _safe_reason(parsed)
    if reason:
        return f"Unsafe expression: {reason}"

    return _safe_eval(parsed)
