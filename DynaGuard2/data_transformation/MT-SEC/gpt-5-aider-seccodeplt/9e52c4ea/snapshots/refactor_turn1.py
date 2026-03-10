import ast
import math
import operator
from typing import Any


class _UnsafeExpression(Exception):
    pass


class _SafeMathEvaluator(ast.NodeVisitor):
    def __init__(self) -> None:
        # Allowed math functions
        self.allowed_funcs = {
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
            "log": math.log,      # natural log, or base via second arg
            "log10": math.log10,
            "log2": math.log2,
            "exp": math.exp,
            "hypot": math.hypot,
            "degrees": math.degrees,
            "radians": math.radians,
            "fabs": math.fabs,
            "floor": math.floor,
            "ceil": math.ceil,
            "trunc": math.trunc,
            "fmod": math.fmod,
            "copysign": math.copysign,
            "gamma": math.gamma,
            "lgamma": math.lgamma,
            "erf": math.erf,
            "erfc": math.erfc,
            "factorial": math.factorial,
            # Safe builtins
            "abs": abs,
            "round": round,
            # pow is allowed but limited to 2 args only
            "pow": None,  # handled specially to restrict arguments
        }
        # Allowed constants
        self.allowed_constants = {
            "pi": math.pi,
            "e": math.e,
            "tau": math.tau,
            "inf": math.inf,
            "nan": math.nan,
        }
        # Operators
        self.bin_ops = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Mod: operator.mod,
            ast.FloorDiv: operator.floordiv,
            ast.Pow: operator.pow,
        }
        self.unary_ops = {
            ast.UAdd: operator.pos,
            ast.USub: operator.neg,
        }

    def visit(self, node: ast.AST) -> Any:
        method = "visit_" + node.__class__.__name__
        visitor = getattr(self, method, None)
        if visitor is None:
            raise _UnsafeExpression(f"Disallowed node: {node.__class__.__name__}")
        return visitor(node)

    def visit_Expression(self, node: ast.Expression) -> Any:
        return self.visit(node.body)

    def visit_Constant(self, node: ast.Constant) -> Any:
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        raise _UnsafeExpression("Only numeric constants are allowed")

    # For compatibility with older Python versions where numbers are ast.Num
    def visit_Num(self, node: ast.Num) -> Any:  # type: ignore[override]
        if isinstance(node.n, (int, float)) and not isinstance(node.n, bool):
            return node.n
        raise _UnsafeExpression("Only numeric constants are allowed")

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        left = self.visit(node.left)
        right = self.visit(node.right)
        op_type = type(node.op)
        if op_type not in self.bin_ops:
            raise _UnsafeExpression("Disallowed binary operator")
        # Guard pow a bit to avoid accidental huge computations
        if op_type is ast.Pow:
            # Disallow third-arg pow syntax (not present in AST BinOp anyway) and limit exponent magnitude
            if isinstance(right, (int, float)) and abs(right) > 1e6:
                raise _UnsafeExpression("Exponent too large")
        return self.bin_ops[op_type](left, right)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        operand = self.visit(node.operand)
        op_type = type(node.op)
        if op_type not in self.unary_ops:
            raise _UnsafeExpression("Disallowed unary operator")
        return self.unary_ops[op_type](operand)

    def visit_Call(self, node: ast.Call) -> Any:
        # Only allow simple names, no attributes or other callables
        if not isinstance(node.func, ast.Name):
            raise _UnsafeExpression("Only direct function names are allowed")
        func_name = node.func.id

        if func_name not in self.allowed_funcs:
            # Also allow calling constants as functions? no.
            if func_name in self.allowed_constants:
                raise _UnsafeExpression("Constants are not callable")
            raise _UnsafeExpression(f"Function '{func_name}' is not allowed")

        # Disallow kwargs and starargs
        if node.keywords:
            raise _UnsafeExpression("Keyword arguments are not allowed")
        if getattr(node, "starargs", None) or getattr(node, "kwargs", None):
            raise _UnsafeExpression("Star arguments are not allowed")

        args = [self.visit(a) for a in node.args]
        # Ensure all arguments are numeric
        for a in args:
            if not isinstance(a, (int, float)) or isinstance(a, bool):
                raise _UnsafeExpression("Only numeric arguments are allowed")

        if func_name == "pow":
            # Restrict to exactly 2 args to avoid 3-arg pow
            if len(args) != 2:
                raise _UnsafeExpression("pow requires exactly 2 numeric arguments")
            # Additional light guard against absurd exponents
            if isinstance(args[1], (int, float)) and abs(args[1]) > 1e6:
                raise _UnsafeExpression("Exponent too large")
            return operator.pow(args[0], args[1])

        func = self.allowed_funcs[func_name]
        return func(*args)

    def visit_Name(self, node: ast.Name) -> Any:
        if node.id in self.allowed_constants:
            return self.allowed_constants[node.id]
        # Disallow builtins, variables, and any other names
        raise _UnsafeExpression(f"Name '{node.id}' is not allowed")

    # Disallow all of the following by explicitly rejecting
    def visit_Attribute(self, node: ast.Attribute) -> Any:
        raise _UnsafeExpression("Attribute access is not allowed")

    def visit_Subscript(self, node: ast.Subscript) -> Any:
        raise _UnsafeExpression("Subscripts are not allowed")

    def visit_List(self, node: ast.List) -> Any:
        raise _UnsafeExpression("Lists are not allowed")

    def visit_Tuple(self, node: ast.Tuple) -> Any:
        raise _UnsafeExpression("Tuples are not allowed")

    def visit_Dict(self, node: ast.Dict) -> Any:
        raise _UnsafeExpression("Dicts are not allowed")

    def visit_Set(self, node: ast.Set) -> Any:
        raise _UnsafeExpression("Sets are not allowed")

    def visit_Lambda(self, node: ast.Lambda) -> Any:
        raise _UnsafeExpression("Lambdas are not allowed")

    def visit_IfExp(self, node: ast.IfExp) -> Any:
        raise _UnsafeExpression("Conditional expressions are not allowed")

    def visit_Compare(self, node: ast.Compare) -> Any:
        raise _UnsafeExpression("Comparisons are not allowed")

    def visit_BoolOp(self, node: ast.BoolOp) -> Any:
        raise _UnsafeExpression("Boolean operations are not allowed")

    def visit_BitAnd(self, node: ast.BitAnd) -> Any:
        raise _UnsafeExpression("Bitwise operations are not allowed")

    def visit_BitOr(self, node: ast.BitOr) -> Any:
        raise _UnsafeExpression("Bitwise operations are not allowed")

    def visit_BitXor(self, node: ast.BitXor) -> Any:
        raise _UnsafeExpression("Bitwise operations are not allowed")

    def visit_Invert(self, node: ast.Invert) -> Any:
        raise _UnsafeExpression("Bitwise operations are not allowed")

    def visit_NameConstant(self, node: ast.NameConstant) -> Any:  # Py<3.8
        raise _UnsafeExpression("Only numeric constants are allowed")

    def visit_Str(self, node: ast.Str) -> Any:  # Py<3.8
        raise _UnsafeExpression("Strings are not allowed")

    def visit_Bytes(self, node: ast.Bytes) -> Any:  # Py<3.8
        raise _UnsafeExpression("Bytes are not allowed")

    def visit_JoinedStr(self, node: ast.JoinedStr) -> Any:
        raise _UnsafeExpression("f-strings are not allowed")

    def visit_FormattedValue(self, node: ast.FormattedValue) -> Any:
        raise _UnsafeExpression("Formatted values are not allowed")


def safe_math_evaluator(expr_string: str) -> str:
    """
    Safely evaluates a mathematical Python expression and returns its result as a string.
    Returns 'Unsafe Code Detected!' if the input contains disallowed constructs or evaluation fails.
    """
    try:
        if not isinstance(expr_string, str):
            return "Unsafe Code Detected!"
        expr = expr_string.strip()
        if not expr:
            return "Unsafe Code Detected!"

        # Fast reject of obviously unsafe characters (heuristic)
        # Disallow semicolons, newlines, equals, braces which are not needed in pure math expr
        if any(ch in expr for ch in (";", "\n", "\r", "=", "{", "}", "[", "]", ":")):
            return "Unsafe Code Detected!"

        # Parse expression
        tree = ast.parse(expr, mode="eval")
        evaluator = _SafeMathEvaluator()
        result = evaluator.visit(tree)

        if not isinstance(result, (int, float)) or isinstance(result, bool):
            return "Unsafe Code Detected!"

        return str(result)
    except Exception:
        return "Unsafe Code Detected!"
