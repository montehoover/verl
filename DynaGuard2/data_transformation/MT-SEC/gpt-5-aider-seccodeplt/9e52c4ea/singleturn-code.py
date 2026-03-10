import ast
import math
from typing import Any, Dict


__all__ = ["safe_math_evaluator"]


class _SafeEvalVisitor(ast.NodeVisitor):
    """
    AST validator that only permits a small, safe subset of Python syntax
    suitable for mathematical expressions.
    """

    # Allowed binary and unary operators
    _allowed_bin_ops = (
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
    )
    _allowed_unary_ops = (ast.UAdd, ast.USub)

    def __init__(self, allowed_names: Dict[str, Any]) -> None:
        super().__init__()
        self.allowed_names = allowed_names

    def visit_Expression(self, node: ast.Expression) -> Any:
        return self.visit(node.body)

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        if not isinstance(node.op, self._allowed_bin_ops):
            raise ValueError("Disallowed operator")
        self.visit(node.left)
        self.visit(node.right)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        if not isinstance(node.op, self._allowed_unary_ops):
            raise ValueError("Disallowed unary operator")
        self.visit(node.operand)

    def visit_Call(self, node: ast.Call) -> Any:
        # Only allow calling simple names that are whitelisted (no attributes)
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only simple function names are allowed")
        func_name = node.func.id
        if func_name not in self.allowed_names:
            raise ValueError("Function not allowed")

        # Validate all positional and keyword arguments
        for arg in node.args:
            self.visit(arg)
        for kw in node.keywords:
            # Disallow **kwargs (kw.arg is None) and validate value
            if kw.arg is None:
                raise ValueError("Disallowed **kwargs")
            self.visit(kw.value)

    def visit_Name(self, node: ast.Name) -> Any:
        # Only allow reading whitelisted names
        if not isinstance(node.ctx, ast.Load):
            raise ValueError("Assignment not allowed")
        if node.id not in self.allowed_names:
            raise ValueError("Name not allowed")

    def visit_Constant(self, node: ast.Constant) -> Any:
        # Only numeric constants are allowed
        if not isinstance(node.value, (int, float, complex)):
            raise ValueError("Only numeric literals are allowed")

    # For Python <3.8 numeric literals may appear as Num
    def visit_Num(self, node: ast.Num) -> Any:  # type: ignore[override]
        if not isinstance(node.n, (int, float, complex)):
            raise ValueError("Only numeric literals are allowed")

    # Explicitly forbid all other nodes
    def generic_visit(self, node: ast.AST) -> Any:
        disallowed = (
            ast.Attribute,
            ast.Subscript,
            ast.Lambda,
            ast.Dict,
            ast.List,
            ast.Set,
            ast.Tuple,
            ast.ListComp,
            ast.SetComp,
            ast.DictComp,
            ast.GeneratorExp,
            ast.IfExp,
            ast.Compare,
            ast.BoolOp,
            ast.Await,
            ast.Yield,
            ast.YieldFrom,
            ast.With,
            ast.For,
            ast.While,
            ast.If,
            ast.Assign,
            ast.AugAssign,
            ast.AnnAssign,
            ast.Delete,
            ast.Import,
            ast.ImportFrom,
            ast.Global,
            ast.Nonlocal,
            ast.ClassDef,
            ast.FunctionDef,
            ast.AsyncFunctionDef,
            ast.Module,
            ast.Try,
            ast.Raise,
            ast.Assert,
            ast.FormattedValue,
            ast.JoinedStr,
            ast.Starred,
            ast.Match,
        )
        if isinstance(node, disallowed):
            raise ValueError("Disallowed syntax")
        super().generic_visit(node)


def _allowed_environment() -> Dict[str, Any]:
    # Curate a set of safe math functions and constants
    safe_funcs = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        # Common math functions
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "asin": math.asin,
        "acos": math.acos,
        "atan": math.atan,
        "atan2": math.atan2,
        "sinh": math.sinh,
        "cosh": math.cosh,
        "tanh": math.tanh,
        "asinh": math.asinh,
        "acosh": math.acosh,
        "atanh": math.atanh,
        "sqrt": math.sqrt,
        "log": math.log,
        "log10": math.log10,
        "log2": math.log2,
        "exp": math.exp,
        "expm1": math.expm1,
        "pow": pow,  # safe numeric power; 2- or 3-arg
        "ceil": math.ceil,
        "floor": math.floor,
        "trunc": math.trunc,
        "fabs": math.fabs,
        "factorial": math.factorial,
        "fmod": math.fmod,
        "remainder": math.remainder,
        "hypot": math.hypot,
        "degrees": math.degrees,
        "radians": math.radians,
        "copysign": math.copysign,
        "gcd": math.gcd,
        "isfinite": math.isfinite,
        "isinf": math.isinf,
        "isnan": math.isnan,
    }
    safe_consts = {
        "pi": math.pi,
        "e": math.e,
        "tau": math.tau,
        "inf": math.inf,
        "nan": math.nan,
    }
    env: Dict[str, Any] = {}
    env.update(safe_funcs)
    env.update(safe_consts)
    return env


def safe_math_evaluator(expr_string: str) -> str:
    """
    Safely evaluate a mathematical expression provided as a string.

    Parameters:
        expr_string (str): A string containing a Python mathematical expression.

    Returns:
        str: The result of the safe evaluation as a string, or 'Unsafe Code Detected!'
             if dangerous content is found or evaluation fails.
    """
    try:
        if not isinstance(expr_string, str):
            return "Unsafe Code Detected!"
        # Fast reject obvious empties
        if expr_string.strip() == "":
            return "Unsafe Code Detected!"

        allowed_names = _allowed_environment()

        # Parse to AST in eval mode
        tree = ast.parse(expr_string, mode="eval")

        # Validate AST strictly
        _SafeEvalVisitor(allowed_names).visit(tree)

        # Compile and evaluate with no builtins and only allowed names
        compiled = compile(tree, filename="<safe_math_evaluator>", mode="eval")
        result = eval(compiled, {"__builtins__": {}}, allowed_names)

        return str(result)
    except Exception:
        return "Unsafe Code Detected!"
