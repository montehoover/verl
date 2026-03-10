import ast
import math
from typing import Any, Dict


# Build a whitelist of safe functions/constants available to the expression.
_SAFE_BUILTINS: Dict[str, Any] = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "pow": pow,
}

_SAFE_MATH_NAMES = [
    "acos", "acosh", "asin", "asinh", "atan", "atan2", "atanh",
    "ceil", "comb", "copysign", "cos", "cosh", "degrees", "dist",
    "erf", "erfc", "exp", "expm1", "fabs", "factorial", "floor",
    "fmod", "frexp", "fsum", "gamma", "gcd", "hypot", "isclose",
    "isfinite", "isinf", "isnan", "ldexp", "lgamma", "log", "log10",
    "log1p", "log2", "modf", "perm", "prod", "radians", "remainder",
    "sin", "sinh", "sqrt", "tan", "tanh", "trunc",
    "pi", "e", "tau", "inf", "nan",
]

_SAFE_GLOBALS: Dict[str, Any] = {name: getattr(math, name) for name in _SAFE_MATH_NAMES if hasattr(math, name)}
_SAFE_GLOBALS.update(_SAFE_BUILTINS)
_ALLOWED_NAMES = set(_SAFE_GLOBALS.keys())


class _SafeExpressionValidator(ast.NodeVisitor):
    allowed_bin_ops = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)
    allowed_unary_ops = (ast.UAdd, ast.USub)

    def __init__(self, allowed_names: set[str]) -> None:
        self.allowed_names = allowed_names

    def generic_visit(self, node: ast.AST) -> Any:
        raise ValueError("Unsafe node encountered")

    def visit_Expression(self, node: ast.Expression) -> Any:
        return self.visit(node.body)

    def visit_Name(self, node: ast.Name) -> Any:
        if node.id not in self.allowed_names:
            raise ValueError("Name not allowed")
        return None

    def visit_Constant(self, node: ast.Constant) -> Any:
        if not isinstance(node.value, (int, float, bool)):
            raise ValueError("Constant type not allowed")
        return None

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        if not isinstance(node.op, self.allowed_unary_ops):
            raise ValueError("Unary operator not allowed")
        return self.visit(node.operand)

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        if not isinstance(node.op, self.allowed_bin_ops):
            raise ValueError("Binary operator not allowed")
        self.visit(node.left)
        self.visit(node.right)
        return None

    def visit_Call(self, node: ast.Call) -> Any:
        # Only allow calls to whitelisted names, no attributes, no starargs / kwargs splats
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only direct function calls are allowed")
        func_name = node.func.id
        if func_name not in self.allowed_names:
            raise ValueError("Function not allowed")

        # Disallow starred args (e.g., *args) and keyword splats (e.g., **kwargs)
        for arg in node.args:
            if isinstance(arg, ast.Starred):
                raise ValueError("Starred arguments not allowed")
            self.visit(arg)
        for kw in node.keywords:
            if kw.arg is None:
                # This represents **kwargs
                raise ValueError("Keyword argument splat not allowed")
            self.visit(kw.value)
        return None

    # Explicitly disallow a wide range of nodes by mapping to generic_visit behavior
    def visit_Attribute(self, node: ast.Attribute) -> Any:
        raise ValueError("Attribute access not allowed")

    def visit_Subscript(self, node: ast.Subscript) -> Any:
        raise ValueError("Subscript not allowed")

    def visit_List(self, node: ast.List) -> Any:
        raise ValueError("List literals not allowed")

    def visit_Dict(self, node: ast.Dict) -> Any:
        raise ValueError("Dict literals not allowed")

    def visit_Set(self, node: ast.Set) -> Any:
        raise ValueError("Set literals not allowed")

    def visit_Tuple(self, node: ast.Tuple) -> Any:
        raise ValueError("Tuple literals not allowed")

    def visit_Compare(self, node: ast.Compare) -> Any:
        raise ValueError("Comparisons not allowed")

    def visit_BoolOp(self, node: ast.BoolOp) -> Any:
        raise ValueError("Boolean operations not allowed")

    def visit_IfExp(self, node: ast.IfExp) -> Any:
        raise ValueError("Ternary expressions not allowed")

    def visit_Lambda(self, node: ast.Lambda) -> Any:
        raise ValueError("Lambda not allowed")


def safely_execute_expression(expression_code: str) -> str:
    """
    Safely evaluates a mathematical Python expression provided by the user.

    Rules:
    - Only a restricted set of numeric constants, arithmetic operators, and math/builtin functions are allowed.
    - No attribute access, indexing, comprehensions, assignments, imports, or other statements are permitted.
    - On any parsing, validation, or runtime error, returns 'Unsafe Code Detected!'.

    :param expression_code: The user-supplied expression to evaluate.
    :return: The stringified result of the evaluation or 'Unsafe Code Detected!' on any issue.
    """
    try:
        if not isinstance(expression_code, str) or not expression_code.strip():
            return "Unsafe Code Detected!"

        # Parse and validate the AST
        expr_ast = ast.parse(expression_code, mode="eval")
        _SafeExpressionValidator(_ALLOWED_NAMES).visit(expr_ast)

        # Evaluate in a restricted environment
        result = eval(compile(expr_ast, "<string>", "eval"), {"__builtins__": {}}, dict(_SAFE_GLOBALS))

        # Ensure the result is a basic numeric or boolean type
        if not isinstance(result, (int, float, bool)):
            return "Unsafe Code Detected!"

        return str(result)
    except Exception:
        return "Unsafe Code Detected!"
