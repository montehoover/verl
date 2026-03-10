import ast
import math
from types import BuiltinFunctionType, FunctionType


def math_expression_safety_check(math_input: str) -> str:
    """
    Safely evaluates a Python mathematical expression given as a string.

    Rules:
    - Only numeric constants, arithmetic operators, and calls to approved math functions/constants are allowed.
    - No attribute access, subscripting, comprehensions, lambdas, conditionals, assignments, or imports.
    - Names must be in a curated allowlist (selected built-ins and math module symbols).
    - Returns the string result of the evaluation.
    - Returns 'Unsafe Code Detected!' if any unsafe content is detected or evaluation fails.
    """

    unsafe_response = "Unsafe Code Detected!"

    if not isinstance(math_input, str):
        return unsafe_response

    expr = math_input.strip()
    if not expr:
        return unsafe_response

    # Build a safe environment of allowed names (functions/constants)
    safe_names = _build_safe_namespace()

    # Parse to AST
    try:
        tree = ast.parse(expr, mode="eval")
    except Exception:
        return unsafe_response

    # Validate AST for only allowed nodes/operations
    try:
        _validate_ast_is_safe(tree, allowed_names=set(safe_names.keys()))
    except Exception:
        return unsafe_response

    # Compile and evaluate with restricted globals/locals
    try:
        code = compile(tree, filename="<math_input>", mode="eval")
        result = eval(code, {"__builtins__": {}}, safe_names)
    except Exception:
        return unsafe_response

    try:
        return str(result)
    except Exception:
        return unsafe_response


def _build_safe_namespace() -> dict:
    """
    Create a dictionary of safe names including selected built-ins and math module
    functions/constants. Excludes anything private or non-numeric.
    """
    safe = {}

    # Selected safe built-ins for numeric math
    safe_builtins = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "pow": pow,  # although ** is allowed, pow() is familiar to some users
    }
    safe.update(safe_builtins)

    # Expose safe math module symbols directly as names (no attribute access)
    for name in dir(math):
        if name.startswith("_"):
            continue
        obj = getattr(math, name)
        # Allow numeric constants
        if isinstance(obj, (int, float)):
            safe[name] = obj
        # Allow callables (math functions)
        elif isinstance(obj, (BuiltinFunctionType, FunctionType)) or callable(obj):
            safe[name] = obj

    return safe


class _SafeEvalVisitor(ast.NodeVisitor):
    """
    AST validator that ensures only a limited, safe subset of Python syntax is used.
    Raises ValueError on any disallowed construct.
    """

    allowed_bin_ops = (
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
    )
    allowed_unary_ops = (ast.UAdd, ast.USub)

    def __init__(self, allowed_names: set[str]):
        super().__init__()
        self.allowed_names = allowed_names

    def visit_Expression(self, node: ast.Expression):
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant):
        # Allow only numeric constants (bool is a subclass of int, so exclude)
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float, complex)):
            raise ValueError("Only numeric constants are allowed")

    def visit_Name(self, node: ast.Name):
        if not isinstance(node.ctx, ast.Load):
            raise ValueError("Only reading names is allowed")
        if node.id not in self.allowed_names:
            raise ValueError(f"Use of name '{node.id}' is not allowed")

    def visit_BinOp(self, node: ast.BinOp):
        if not isinstance(node.op, self.allowed_bin_ops):
            raise ValueError("Disallowed binary operator")
        self.visit(node.left)
        self.visit(node.right)

    def visit_UnaryOp(self, node: ast.UnaryOp):
        if not isinstance(node.op, self.allowed_unary_ops):
            raise ValueError("Disallowed unary operator")
        self.visit(node.operand)

    def visit_Call(self, node: ast.Call):
        # Only allow function calls by simple name, no attributes or complex callables
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only direct function calls by name are allowed")
        if node.func.id not in self.allowed_names:
            raise ValueError(f"Function '{node.func.id}' is not allowed")

        # Disallow keywords and starred arguments for simplicity/safety
        if node.keywords:
            raise ValueError("Keyword arguments are not allowed")
        for arg in node.args:
            if isinstance(arg, ast.Starred):
                raise ValueError("Starred arguments are not allowed")
            self.visit(arg)

    # Explicitly forbid everything else by overriding and raising, or letting
    # generic_visit catch unexpected node types via a custom generic_visit
    def visit_Attribute(self, node: ast.Attribute):
        raise ValueError("Attribute access is not allowed")

    def visit_Subscript(self, node: ast.Subscript):
        raise ValueError("Subscript access is not allowed")

    def visit_Lambda(self, node: ast.Lambda):
        raise ValueError("Lambdas are not allowed")

    def visit_List(self, node: ast.List):
        raise ValueError("List literals are not allowed")

    def visit_Tuple(self, node: ast.Tuple):
        raise ValueError("Tuple literals are not allowed")

    def visit_Dict(self, node: ast.Dict):
        raise ValueError("Dict literals are not allowed")

    def visit_Set(self, node: ast.Set):
        raise ValueError("Set literals are not allowed")

    def visit_IfExp(self, node: ast.IfExp):
        raise ValueError("Conditional expressions are not allowed")

    def visit_BoolOp(self, node: ast.BoolOp):
        raise ValueError("Boolean operations are not allowed")

    def visit_Compare(self, node: ast.Compare):
        raise ValueError("Comparisons are not allowed")

    def visit_ListComp(self, node: ast.ListComp):
        raise ValueError("Comprehensions are not allowed")

    def visit_SetComp(self, node: ast.SetComp):
        raise ValueError("Comprehensions are not allowed")

    def visit_DictComp(self, node: ast.DictComp):
        raise ValueError("Comprehensions are not allowed")

    def visit_GeneratorExp(self, node: ast.GeneratorExp):
        raise ValueError("Comprehensions are not allowed")

    def visit_NamedExpr(self, node: ast.NamedExpr):
        raise ValueError("Assignment expressions are not allowed")

    def visit_Assign(self, node: ast.Assign):
        raise ValueError("Assignments are not allowed")

    def visit_AugAssign(self, node: ast.AugAssign):
        raise ValueError("Assignments are not allowed")

    def visit_CallBlock(self, node):
        # Non-standard / placeholder defense
        raise ValueError("Blocks are not allowed")

    def generic_visit(self, node):
        # Allow traversal only through known safe container nodes
        if isinstance(
            node,
            (
                ast.Expression,
                ast.BinOp,
                ast.UnaryOp,
                ast.Call,
                ast.Name,
                ast.Load,
                ast.Constant,
            ),
        ):
            super().generic_visit(node)
        else:
            # Unknown or disallowed node type
            raise ValueError(f"Disallowed syntax: {type(node).__name__}")


def _validate_ast_is_safe(tree: ast.AST, allowed_names: set[str]) -> None:
    visitor = _SafeEvalVisitor(allowed_names)
    visitor.visit(tree)
