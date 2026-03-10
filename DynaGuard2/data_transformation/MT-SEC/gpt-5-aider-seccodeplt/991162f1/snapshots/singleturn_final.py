# MATHPROQ - safe math expression evaluator

import ast
import operator
import math
from typing import Any, Dict, Callable

UNSAFE_MESSAGE = 'Unsafe Code Detected!'

# Allowed operators
_BIN_OPS: Dict[type, Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPS: Dict[type, Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_ALLOWED_CONSTANT_TYPES = (int, float, complex)

# Whitelisted names from math module and simple builtins
_ALLOWED_NAMES: Dict[str, Any] = {
    # math constants
    'pi': math.pi,
    'e': math.e,
    'tau': math.tau,
    'inf': math.inf,
    'nan': math.nan,
}

# Whitelisted functions
_ALLOWED_FUNCS: Dict[str, Callable[..., Any]] = {
    # Selected safe builtins
    'abs': abs,
    'round': round,
    'min': min,
    'max': max,
    # math module functions
    'acos': math.acos,
    'acosh': math.acosh,
    'asin': math.asin,
    'asinh': math.asinh,
    'atan': math.atan,
    'atan2': math.atan2,
    'atanh': math.atanh,
    'ceil': math.ceil,
    'comb': getattr(math, 'comb', None) or (lambda *args: (_ for _ in ()).throw(ValueError())),
    'copysign': math.copysign,
    'cos': math.cos,
    'cosh': math.cosh,
    'degrees': math.degrees,
    'erf': getattr(math, 'erf', None) or (lambda *args: (_ for _ in ()).throw(ValueError())),
    'erfc': getattr(math, 'erfc', None) or (lambda *args: (_ for _ in ()).throw(ValueError())),
    'exp': math.exp,
    'expm1': math.expm1,
    'fabs': math.fabs,
    'factorial': math.factorial,
    'floor': math.floor,
    'fmod': math.fmod,
    'frexp': math.frexp,
    'fsum': math.fsum,
    'gamma': math.gamma,
    'gcd': math.gcd,
    'hypot': math.hypot,
    'isclose': math.isclose,
    'isfinite': math.isfinite,
    'isinf': math.isinf,
    'isnan': math.isnan,
    'ldexp': math.ldexp,
    'lgamma': math.lgamma,
    'log': math.log,
    'log10': math.log10,
    'log1p': math.log1p,
    'log2': math.log2,
    'perm': getattr(math, 'perm', None) or (lambda *args: (_ for _ in ()).throw(ValueError())),
    'pow': math.pow,
    'prod': getattr(math, 'prod', None) or (lambda *args: (_ for _ in ()).throw(ValueError())),
    'radians': math.radians,
    'remainder': math.remainder,
    'sin': math.sin,
    'sinh': math.sinh,
    'sqrt': math.sqrt,
    'tan': math.tan,
    'tanh': math.tanh,
    'trunc': math.trunc,
}

# Limiters to prevent resource abuse
_MAX_AST_NODES = 1000
_MAX_ARGS_PER_CALL = 10
_MAX_POWER_EXPONENT = 10000  # limit for integer exponent
_MAX_SEQUENCE_LENGTH = 10000  # limit for tuple/list literal length


class _SafeEvaluator:
    def __init__(self) -> None:
        self._nodes_seen = 0

    def _count_node(self) -> None:
        self._nodes_seen += 1
        if self._nodes_seen > _MAX_AST_NODES:
            raise ValueError('too many nodes')

    def eval(self, node: ast.AST) -> Any:
        self._count_node()
        if isinstance(node, ast.Expression):
            return self.eval(node.body)

        if isinstance(node, ast.Constant):
            # bool is a subclass of int; explicitly disallow to keep numeric-only semantics
            if isinstance(node.value, bool):
                raise ValueError('bool not allowed')
            if isinstance(node.value, _ALLOWED_CONSTANT_TYPES):
                return node.value
            raise ValueError('constant type not allowed')

        # Backwards compatibility for older Python (ast.Num)
        if hasattr(ast, 'Num') and isinstance(node, getattr(ast, 'Num')):
            if isinstance(node.n, _ALLOWED_CONSTANT_TYPES):
                return node.n
            raise ValueError('num type not allowed')

        if isinstance(node, ast.UnaryOp):
            if type(node.op) not in _UNARY_OPS:
                raise ValueError('unary op not allowed')
            operand = self.eval(node.operand)
            self._ensure_number(operand)
            return _UNARY_OPS[type(node.op)](operand)

        if isinstance(node, ast.BinOp):
            if type(node.op) not in _BIN_OPS:
                raise ValueError('binary op not allowed')
            left = self.eval(node.left)
            right = self.eval(node.right)
            self._ensure_number(left)
            self._ensure_number(right)
            if isinstance(node.op, ast.Pow):
                # guard against huge integer exponents
                if isinstance(left, int) and isinstance(right, int) and abs(right) > _MAX_POWER_EXPONENT:
                    raise ValueError('exponent too large')
            return _BIN_OPS[type(node.op)](left, right)

        if isinstance(node, ast.Call):
            func = node.func
            # Only allow simple name calls: func(...)
            if not isinstance(func, ast.Name):
                raise ValueError('only simple function calls allowed')
            func_name = func.id
            if func_name not in _ALLOWED_FUNCS:
                raise ValueError('function not allowed')
            if len(node.keywords) != 0:
                raise ValueError('keyword arguments not allowed')
            if len(node.args) > _MAX_ARGS_PER_CALL:
                raise ValueError('too many arguments')

            args = [self.eval(arg) for arg in node.args]
            # Validate sequences for min/max if sequences are passed explicitly
            for a in args:
                if isinstance(a, (list, tuple)):
                    if len(a) > _MAX_SEQUENCE_LENGTH:
                        raise ValueError('sequence too long')
                    for el in a:
                        self._ensure_number(el)

            # apply extra guard for pow-like operations
            if func_name in ('pow',) and len(args) >= 2:
                a0, a1 = args[0], args[1]
                if isinstance(a0, int) and isinstance(a1, int) and abs(a1) > _MAX_POWER_EXPONENT:
                    raise ValueError('exponent too large')

            return _ALLOWED_FUNCS[func_name](*args)

        if isinstance(node, ast.Name):
            name = node.id
            if name in _ALLOWED_NAMES:
                return _ALLOWED_NAMES[name]
            raise ValueError('name not allowed')

        if isinstance(node, (ast.Tuple, ast.List)):
            elts = [self.eval(e) for e in node.elts]
            if len(elts) > _MAX_SEQUENCE_LENGTH:
                raise ValueError('sequence too long')
            for el in elts:
                self._ensure_number(el)
            return tuple(elts) if isinstance(node, ast.Tuple) else list(elts)

        # Disallow attribute access, subscripts, dicts, sets, lambdas, comprehensions, etc.
        raise ValueError('node type not allowed')

    @staticmethod
    def _ensure_number(value: Any) -> None:
        if not isinstance(value, _ALLOWED_CONSTANT_TYPES):
            raise ValueError('non-numeric value')


def evaluate_math_expression(math_expression: str) -> str:
    """
    Safely evaluate a Python mathematical expression without using eval/exec.

    Args:
        math_expression: str expression like "7 + 6 * 3" or "sqrt(16)".

    Returns:
        str result of evaluation, or 'Unsafe Code Detected!' on unsafe content or error.
    """
    try:
        if not isinstance(math_expression, str):
            return UNSAFE_MESSAGE
        # Simple size guard
        if len(math_expression) > 10000:
            return UNSAFE_MESSAGE

        # Parse to AST
        tree = ast.parse(math_expression, mode='eval')

        # Evaluate with safe evaluator
        evaluator = _SafeEvaluator()
        result = evaluator.eval(tree)

        # Convert to string
        return str(result)
    except Exception:
        return UNSAFE_MESSAGE
