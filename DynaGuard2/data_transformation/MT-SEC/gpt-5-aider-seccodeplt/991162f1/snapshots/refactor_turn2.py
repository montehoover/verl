import ast
import math
import operator
from typing import Any

# Safety limits to avoid excessive computation
_MAX_EXPR_LENGTH = 10000
_MAX_NODES = 1000
_MAX_DEPTH = 50
_MAX_ABS_VALUE = 1e308  # Close to float max
_MAX_POW_EXP = 1000
_MAX_POW_BASE_ABS = 1_000_000


def _get_allowed_environment() -> dict[str, Any]:
    # Whitelisted operators
    allowed_binops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,  # handled via safe_pow
    }
    allowed_unaryops = {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }

    # Whitelisted math functions and constants
    math_func_names = {
        'sin', 'cos', 'tan',
        'asin', 'acos', 'atan', 'atan2',
        'sinh', 'cosh', 'tanh',
        'asinh', 'acosh', 'atanh',
        'log', 'log10', 'log2',
        'exp', 'sqrt', 'pow',
        'fabs', 'floor', 'ceil', 'trunc',
        'factorial', 'fmod', 'hypot',
        'degrees', 'radians',
        'gamma', 'lgamma',
        'erf', 'erfc',
        'copysign', 'remainder',
        'fsum', 'prod', 'dist',
        'isfinite', 'isinf', 'isnan',
    }
    math_const_names = {'pi', 'e', 'tau', 'inf', 'nan'}

    allowed_math_objects: dict[str, Any] = {}
    for name in math_func_names | math_const_names:
        if hasattr(math, name):
            allowed_math_objects[name] = getattr(math, name)

    # Allow calling common functions directly without "math."
    allowed_direct_funcs: dict[str, Any] = {
        'abs': abs,
        'round': round,
        'pow': pow,  # will be safely handled
    }
    for name in math_func_names:
        obj = allowed_math_objects.get(name)
        if callable(obj):
            allowed_direct_funcs[name] = obj

    # Allow direct access to constants (e.g., "pi") as names
    allowed_direct_names: dict[str, Any] = {k: v for k, v in allowed_math_objects.items() if not callable(v)}

    return {
        'allowed_binops': allowed_binops,
        'allowed_unaryops': allowed_unaryops,
        'allowed_math_objects': allowed_math_objects,
        'allowed_direct_funcs': allowed_direct_funcs,
        'allowed_direct_names': allowed_direct_names,
    }


def _parse_expression(expr: str) -> ast.AST:
    return ast.parse(expr, mode='eval')


def _is_number(x: Any) -> bool:
    # Exclude bool as it's a subclass of int
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _assert_number(x: Any) -> None:
    if not _is_number(x):
        raise ValueError("Non-numeric result")
    if math.isfinite(x) and abs(x) > _MAX_ABS_VALUE:
        raise ValueError("Result too large")


def _safe_pow(a: Any, b: Any) -> float:
    if not _is_number(a) or not _is_number(b):
        raise ValueError("Invalid operands to pow")
    # If integer exponent, limit size
    if float(b).is_integer():
        if abs(a) > _MAX_POW_BASE_ABS or abs(b) > _MAX_POW_EXP:
            raise ValueError("Exponent/base too large")
    result = operator.pow(a, b)
    _assert_number(result)
    return result


def _validate_ast(tree: ast.AST, env: dict[str, Any]) -> None:
    allowed_binops = env['allowed_binops']
    allowed_unaryops = env['allowed_unaryops']
    allowed_math_objects = env['allowed_math_objects']
    allowed_direct_funcs = env['allowed_direct_funcs']
    allowed_direct_names = env['allowed_direct_names']

    node_count = 0

    def visit(node: ast.AST, parent: ast.AST | None = None, depth: int = 0) -> None:
        nonlocal node_count
        node_count += 1
        if node_count > _MAX_NODES or depth > _MAX_DEPTH:
            raise ValueError("Expression too complex")

        if isinstance(node, ast.Expression):
            visit(node.body, node, depth + 1)
            return

        if isinstance(node, ast.Constant):  # Python 3.8+
            if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
                return
            raise ValueError("Only numeric constants allowed")

        # Python <3.8 compatibility (Num is deprecated but keep for completeness)
        if hasattr(ast, 'Num') and isinstance(node, getattr(ast, 'Num')):
            if isinstance(node.n, (int, float)) and not isinstance(node.n, bool):
                return
            raise ValueError("Only numeric constants allowed")

        if isinstance(node, ast.UnaryOp):
            if type(node.op) not in allowed_unaryops:
                raise ValueError("Unary op not allowed")
            visit(node.operand, node, depth + 1)
            return

        if isinstance(node, ast.BinOp):
            if type(node.op) not in allowed_binops:
                raise ValueError("Binary op not allowed")
            visit(node.left, node, depth + 1)
            visit(node.right, node, depth + 1)
            return

        if isinstance(node, ast.Name):
            # If used as a function (part of Call.func), validate in Call handler
            if isinstance(parent, ast.Call) and parent.func is node:
                # Only certain functions are allowed
                if node.id not in allowed_direct_funcs:
                    raise ValueError("Function not allowed")
                return
            # Otherwise, only direct constants are allowed (no function objects)
            if node.id in allowed_direct_names:
                return
            # Prevent referencing 'math' or functions as bare names
            raise ValueError(f"Name '{node.id}' not allowed")
        
        if isinstance(node, ast.Attribute):
            # Only allow math.<allowed>
            if not (isinstance(node.value, ast.Name) and node.value.id == 'math'):
                raise ValueError("Attribute access not allowed")
            attr = node.attr
            if isinstance(parent, ast.Call) and parent.func is node:
                # Called function must be callable
                if attr not in allowed_math_objects or not callable(allowed_math_objects[attr]):
                    raise ValueError("Function not allowed")
                return
            # As a value, only constants (non-callables) are allowed
            if attr in allowed_math_objects and not callable(allowed_math_objects[attr]):
                return
            raise ValueError("Function object usage not allowed")

        if isinstance(node, ast.Call):
            if node.keywords:
                raise ValueError("Keywords not allowed in calls")
            # Validate function target
            if isinstance(node.func, ast.Name):
                if node.func.id not in allowed_direct_funcs:
                    raise ValueError("Function not allowed")
            elif isinstance(node.func, ast.Attribute):
                if not (isinstance(node.func.value, ast.Name) and node.func.value.id == 'math'):
                    raise ValueError("Attribute access not allowed")
                attr = node.func.attr
                if attr not in allowed_math_objects or not callable(allowed_math_objects[attr]):
                    raise ValueError("Function not allowed")
            else:
                raise ValueError("Function not allowed")
            # Validate arguments
            for arg in node.args:
                visit(arg, node, depth + 1)
            return

        # Disallow everything else
        raise ValueError("Disallowed expression construct")

    visit(tree)


def _evaluate_ast(tree: ast.AST, env: dict[str, Any]) -> Any:
    allowed_binops = env['allowed_binops']
    allowed_unaryops = env['allowed_unaryops']
    allowed_math_objects = env['allowed_math_objects']
    allowed_direct_funcs = env['allowed_direct_funcs']
    allowed_direct_names = env['allowed_direct_names']

    node_count = 0

    def eval_call(func_obj: Any, args_nodes: list[ast.AST], depth: int) -> Any:
        if any(not isinstance(a, ast.AST) for a in args_nodes):
            raise ValueError("Invalid args")
        args = [_eval(a, depth + 1) for a in args_nodes]
        # Special handling for built-in pow
        if func_obj is pow:
            if not (2 <= len(args) <= 3):
                raise ValueError("Invalid pow arity")
            if len(args) == 2:
                return _safe_pow(args[0], args[1])
            # 3-arg pow: only allow integers and positive modulus to keep it safe/finite
            if not all(isinstance(x, int) and not isinstance(x, bool) for x in args):
                raise ValueError("pow with mod requires integer args")
            if args[2] == 0:
                raise ValueError("pow modulo 0")
            result = pow(args[0], args[1], args[2])
            _assert_number(result)
            return result
        # For math functions, just call after args evaluation
        result = func_obj(*args)
        # Functions like isfinite/isinf/isnan return bool; allow them, but convert to int for numeric-only policy
        if isinstance(result, bool):
            result = 1 if result else 0
        _assert_number(result)
        return result

    def _eval(node: ast.AST, depth: int = 0) -> Any:
        nonlocal node_count
        node_count += 1
        if node_count > _MAX_NODES or depth > _MAX_DEPTH:
            raise ValueError("Expression too complex")

        if isinstance(node, ast.Expression):
            return _eval(node.body, depth + 1)

        if isinstance(node, ast.Constant):  # Python 3.8+
            if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
                return node.value
            raise ValueError("Only numeric constants allowed")

        # Python <3.8 compatibility (Num is deprecated but keep for completeness)
        if hasattr(ast, 'Num') and isinstance(node, getattr(ast, 'Num')):
            if isinstance(node.n, (int, float)) and not isinstance(node.n, bool):
                return node.n
            raise ValueError("Only numeric constants allowed")

        if isinstance(node, ast.UnaryOp):
            if type(node.op) not in allowed_unaryops:
                raise ValueError("Unary op not allowed")
            operand = _eval(node.operand, depth + 1)
            result = allowed_unaryops[type(node.op)](operand)
            _assert_number(result)
            return result

        if isinstance(node, ast.BinOp):
            if type(node.op) not in allowed_binops:
                raise ValueError("Binary op not allowed")
            left = _eval(node.left, depth + 1)
            right = _eval(node.right, depth + 1)
            if isinstance(node.op, ast.Pow):
                return _safe_pow(left, right)
            result = allowed_binops[type(node.op)](left, right)
            _assert_number(result)
            return result

        if isinstance(node, ast.Name):
            # resolve direct names (constants only)
            if node.id in allowed_direct_names:
                return allowed_direct_names[node.id]
            if node.id in allowed_direct_funcs:
                # Disallow using function objects as values; must be called
                raise ValueError("Function object usage not allowed")
            raise ValueError(f"Name '{node.id}' not allowed")

        if isinstance(node, ast.Attribute):
            # Only allow math.<allowed>
            if isinstance(node.value, ast.Name) and node.value.id == 'math':
                attr = node.attr
                if attr in allowed_math_objects:
                    obj = allowed_math_objects[attr]
                    if callable(obj):
                        # Referencing function object without calling is not allowed
                        raise ValueError("Function object usage not allowed")
                    return obj
            raise ValueError("Attribute access not allowed")

        if isinstance(node, ast.Call):
            if node.keywords:
                raise ValueError("Keywords not allowed in calls")
            func_obj = None
            if isinstance(node.func, ast.Name):
                func_obj = allowed_direct_funcs.get(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                if isinstance(node.func.value, ast.Name) and node.func.value.id == 'math':
                    func_obj = allowed_math_objects.get(node.func.attr)
            if func_obj is None or not callable(func_obj):
                raise ValueError("Function not allowed")
            return eval_call(func_obj, node.args, depth)

        # Disallow everything else
        raise ValueError("Disallowed expression construct")

    return _eval(tree)


def evaluate_math_expression(math_expression: str) -> str:
    """
    Safely evaluate a mathematical expression string without using eval/exec.

    Returns:
        str: The computed result as a string, or 'Unsafe Code Detected!' on any
             unsafe construct or error.
    """
    try:
        if not isinstance(math_expression, str):
            return 'Unsafe Code Detected!'

        if len(math_expression) > _MAX_EXPR_LENGTH:
            return 'Unsafe Code Detected!'

        env = _get_allowed_environment()

        try:
            tree = _parse_expression(math_expression)
        except Exception:
            return 'Unsafe Code Detected!'

        # Validate structure and constructs first
        _validate_ast(tree, env)

        # Evaluate if validation passes
        result = _evaluate_ast(tree, env)

        return str(result)

    except Exception:
        return 'Unsafe Code Detected!'
