import ast
import logging
import math
from typing import Any, Dict


# Configure module-level logger for MATHPROQ
logger = logging.getLogger("MATHPROQ")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def _format_expression_for_log(expression_code: Any, limit: int = 200) -> str:
    """
    Prepare a compact, safe preview of the expression for logging.
    """
    try:
        if not isinstance(expression_code, str):
            return f"<non-string:{type(expression_code).__name__}>"
        s = expression_code.strip()
        if len(s) <= limit:
            return s
        return f"{s[:limit]}... [len={len(s)}]"
    except Exception:
        return "<unprintable expression>"


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


def _parse_and_validate_expression(expression_code: str, allowed_names: set[str]) -> ast.Expression:
    """
    Parse a string into an AST expression and validate it against the allowed names and nodes.
    Raises ValueError on any issue.
    """
    expr_for_log = _format_expression_for_log(expression_code)
    logger.debug("Parsing and validating expression: '%s'", expr_for_log)

    if not isinstance(expression_code, str) or not expression_code.strip():
        logger.debug("Expression is empty or non-string: '%s'", expr_for_log)
        raise ValueError("Empty or non-string expression")
    expr_ast = ast.parse(expression_code, mode="eval")
    _SafeExpressionValidator(allowed_names).visit(expr_ast)
    logger.debug("Validation succeeded for expression: '%s'", expr_for_log)
    return expr_ast


def _execute_validated_expression(expr_ast: ast.Expression, safe_globals: Dict[str, Any]) -> Any:
    """
    Execute a previously validated AST expression within a restricted environment.
    Raises ValueError if the result is of a disallowed type or on any execution issue.
    """
    result = eval(compile(expr_ast, "<string>", "eval"), {"__builtins__": {}}, dict(safe_globals))
    if not isinstance(result, (int, float, bool)):
        raise ValueError("Result type not allowed")
    return result


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
    expr_for_log = _format_expression_for_log(expression_code)
    logger.info("Attempting execution: expr='%s'", expr_for_log)

    try:
        expr_ast = _parse_and_validate_expression(expression_code, _ALLOWED_NAMES)
        result = _execute_validated_expression(expr_ast, _SAFE_GLOBALS)
        logger.info("Execution succeeded: expr='%s', result=%s", expr_for_log, result)
        return str(result)
    except Exception as e:
        logger.warning("Execution failed or unsafe: expr='%s', error=%s", expr_for_log, e, exc_info=True)
        return "Unsafe Code Detected!"
