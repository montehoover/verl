import ast
from typing import Optional


class UnsafeOperation(Exception):
    pass


class _SafeVisitor(ast.NodeVisitor):
    _allowed_bin_ops = (
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
    )
    _allowed_unary_ops = (
        ast.UAdd,
        ast.USub,
    )

    def visit(self, node):
        # Override to ensure every node is checked by our rules
        return super().visit(node)

    def generic_visit(self, node):
        # Any node not explicitly handled is considered unsafe
        raise UnsafeOperation(f"Disallowed node type: {type(node).__name__}")

    # Module and statements
    def visit_Module(self, node: ast.Module):
        for stmt in node.body:
            self.visit(stmt)

    def visit_Expr(self, node: ast.Expr):
        self.visit(node.value)

    def visit_Assign(self, node: ast.Assign):
        # Only simple variable assignments (no tuple, attribute, subscript targets)
        for target in node.targets:
            if not isinstance(target, ast.Name):
                raise UnsafeOperation("Only simple name assignments are allowed")
            self.visit(target)
        self.visit(node.value)

    def visit_AugAssign(self, node: ast.AugAssign):
        if not isinstance(node.target, ast.Name):
            raise UnsafeOperation("Only simple name augmented assignments are allowed")
        if not isinstance(node.op, self._allowed_bin_ops):
            raise UnsafeOperation("Augmented assignment uses disallowed operator")
        self.visit(node.target)
        self.visit(node.value)

    # Names and constants
    def visit_Name(self, node: ast.Name):
        # Block direct access to __builtins__
        if node.id == "__builtins__":
            raise UnsafeOperation("Access to __builtins__ is not allowed")
        # Allow variable usage/definitions otherwise (no generic_visit to avoid visiting ctx)
        return

    def visit_Constant(self, node: ast.Constant):
        v = node.value
        # Allow only numbers and strings
        if isinstance(v, bool) or v is None:
            raise UnsafeOperation("Booleans and None are not allowed")
        if not isinstance(v, (int, float, complex, str)):
            raise UnsafeOperation("Only numeric and string constants are allowed")
        return

    # Py<3.8 compatibility (optional)
    def visit_Num(self, node: ast.Num):  # type: ignore[override]
        return
    def visit_Str(self, node: ast.Str):  # type: ignore[override]
        return

    # Expressions
    def visit_BinOp(self, node: ast.BinOp):
        if not isinstance(node.op, self._allowed_bin_ops):
            raise UnsafeOperation("Disallowed binary operator")
        self.visit(node.left)
        self.visit(node.right)

    def visit_UnaryOp(self, node: ast.UnaryOp):
        if not isinstance(node.op, self._allowed_unary_ops):
            raise UnsafeOperation("Disallowed unary operator")
        self.visit(node.operand)

    def visit_Subscript(self, node: ast.Subscript):
        # Allow only simple string-like indexing/slicing
        # Disallow subscript on __builtins__ directly
        if isinstance(node.value, ast.Name) and node.value.id == "__builtins__":
            raise UnsafeOperation("Subscript on __builtins__ is not allowed")
        self._visit_allowed_subscript_target(node.value)
        self._check_allowed_slice(node.slice)

    def _visit_allowed_subscript_target(self, node: ast.AST):
        # Target of subscript can be any safe expression (no attributes or calls allowed anywhere)
        # We just delegate to visit for regular checks.
        self.visit(node)

    def _check_allowed_slice(self, slice_node: ast.AST):
        # Python 3.9+: slice can be ast.Slice or any expression (for index)
        if isinstance(slice_node, ast.Slice):
            # Lower/upper/step must be safe arithmetic expressions (or None)
            if slice_node.lower is not None:
                self._ensure_slice_index_expr(slice_node.lower)
            if slice_node.upper is not None:
                self._ensure_slice_index_expr(slice_node.upper)
            if slice_node.step is not None:
                self._ensure_slice_index_expr(slice_node.step)
            return
        # Python <3.9 uses ast.Index/ast.ExtSlice
        if hasattr(ast, "Index") and isinstance(slice_node, getattr(ast, "Index")):  # type: ignore[attr-defined]
            value = slice_node.value  # type: ignore[attr-defined]
            self._ensure_slice_index_expr(value)
            return
        if hasattr(ast, "ExtSlice") and isinstance(slice_node, getattr(ast, "ExtSlice")):  # type: ignore[attr-defined]
            # Disallow multidimensional slicing
            raise UnsafeOperation("Extended slicing is not allowed")
        # Otherwise, it's an index expression (e.g., s[0])
        self._ensure_slice_index_expr(slice_node)

    def _ensure_slice_index_expr(self, node: ast.AST):
        # Allow index expressions composed of safe arithmetic/variables only.
        # Specifically disallow any string literals inside index to avoid dict-style access like obj['key'].
        # Also disallow f-strings as index.
        if isinstance(node, ast.Constant):
            v = node.value
            if isinstance(v, bool) or v is None:
                raise UnsafeOperation("Boolean/None not allowed in slice/index")
            if not isinstance(v, (int, float, complex)):
                # Disallow string or other constant types as index
                raise UnsafeOperation("Only numeric constants allowed in slice/index")
            return
        if isinstance(node, (ast.BinOp, ast.UnaryOp, ast.Name)):
            self.visit(node)
            return
        # Allow parentheses around expressions (ast.Expr won't appear here, so nothing special)
        raise UnsafeOperation("Disallowed expression in slice/index")

    # F-strings
    def visit_JoinedStr(self, node: ast.JoinedStr):
        for v in node.values:
            self.visit(v)

    def visit_FormattedValue(self, node: ast.FormattedValue):
        self.visit(node.value)
        if node.format_spec is not None:
            self.visit(node.format_spec)

    # Contexts (no-op)
    def visit_Load(self, node: ast.Load):
        return

    def visit_Store(self, node: ast.Store):
        return

    def visit_Del(self, node: ast.Del):
        raise UnsafeOperation("Deletion is not allowed")

    # Explicitly disallow dangerous constructs by defining visitors that raise
    def visit_Call(self, node: ast.Call):
        raise UnsafeOperation("Function calls are not allowed")

    def visit_Attribute(self, node: ast.Attribute):
        raise UnsafeOperation("Attribute access is not allowed")

    def visit_IfExp(self, node: ast.IfExp):
        raise UnsafeOperation("Conditional expressions are not allowed")

    def visit_Compare(self, node: ast.Compare):
        raise UnsafeOperation("Comparisons are not allowed")

    def visit_BoolOp(self, node: ast.BoolOp):
        raise UnsafeOperation("Boolean operations are not allowed")

    def visit_List(self, node: ast.List):
        raise UnsafeOperation("List literals are not allowed")

    def visit_Tuple(self, node: ast.Tuple):
        raise UnsafeOperation("Tuple literals are not allowed")

    def visit_Set(self, node: ast.Set):
        raise UnsafeOperation("Set literals are not allowed")

    def visit_Dict(self, node: ast.Dict):
        raise UnsafeOperation("Dict literals are not allowed")

    def visit_Lambda(self, node: ast.Lambda):
        raise UnsafeOperation("Lambdas are not allowed")

    def visit_DictComp(self, node: ast.DictComp):
        raise UnsafeOperation("Comprehensions are not allowed")

    def visit_ListComp(self, node: ast.ListComp):
        raise UnsafeOperation("Comprehensions are not allowed")

    def visit_SetComp(self, node: ast.SetComp):
        raise UnsafeOperation("Comprehensions are not allowed")

    def visit_GeneratorExp(self, node: ast.GeneratorExp):
        raise UnsafeOperation("Comprehensions are not allowed")

    def visit_NamedExpr(self, node: ast.NamedExpr):
        raise UnsafeOperation("Walrus operator is not allowed")

    def visit_Await(self, node: ast.Await):
        raise UnsafeOperation("Await is not allowed")

    def visit_Yield(self, node: ast.Yield):
        raise UnsafeOperation("Yield is not allowed")

    def visit_YieldFrom(self, node: ast.YieldFrom):
        raise UnsafeOperation("Yield is not allowed")

    def visit_With(self, node: ast.With):
        raise UnsafeOperation("With statements are not allowed")

    def visit_AsyncWith(self, node: ast.AsyncWith):
        raise UnsafeOperation("With statements are not allowed")

    def visit_For(self, node: ast.For):
        raise UnsafeOperation("Loops are not allowed")

    def visit_AsyncFor(self, node: ast.AsyncFor):
        raise UnsafeOperation("Loops are not allowed")

    def visit_While(self, node: ast.While):
        raise UnsafeOperation("Loops are not allowed")

    def visit_If(self, node: ast.If):
        raise UnsafeOperation("Control flow is not allowed")

    def visit_Try(self, node: ast.Try):
        raise UnsafeOperation("Exception handling is not allowed")

    def visit_FunctionDef(self, node: ast.FunctionDef):
        raise UnsafeOperation("Function definitions are not allowed")

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        raise UnsafeOperation("Function definitions are not allowed")

    def visit_ClassDef(self, node: ast.ClassDef):
        raise UnsafeOperation("Class definitions are not allowed")

    def visit_Import(self, node: ast.Import):
        raise UnsafeOperation("Imports are not allowed")

    def visit_ImportFrom(self, node: ast.ImportFrom):
        raise UnsafeOperation("Imports are not allowed")

    def visit_Global(self, node: ast.Global):
        raise UnsafeOperation("Global statements are not allowed")

    def visit_Nonlocal(self, node: ast.Nonlocal):
        raise UnsafeOperation("Nonlocal statements are not allowed")

    def visit_Delete(self, node: ast.Delete):
        raise UnsafeOperation("Deletion is not allowed")

    def visit_Pass(self, node: ast.Pass):
        # Allow pass statement? It's not arithmetic/string manipulation; disallow to be strict.
        raise UnsafeOperation("Pass is not allowed")

    def visit_Return(self, node: ast.Return):
        raise UnsafeOperation("Return is not allowed")

    def visit_Raise(self, node: ast.Raise):
        raise UnsafeOperation("Raise is not allowed")

    def visit_Break(self, node: ast.Break):
        raise UnsafeOperation("Break is not allowed")

    def visit_Continue(self, node: ast.Continue):
        raise UnsafeOperation("Continue is not allowed")

    # Pattern matching (3.10+)
    def visit_Match(self, node: ast.Match):
        raise UnsafeOperation("Pattern matching is not allowed")


def filter_unsafe_operations(script: str) -> bool:
    """
    Returns True if the provided Python script contains only safe operations:
    - Basic arithmetic (+, -, *, /, //, %, **, unary + and -)
    - String manipulations via concatenation (+), repetition (*), f-strings
    - Simple indexing/slicing, without attribute access or function calls
    - Simple assignments to variable names

    Returns False if parsing fails or any disallowed construct is found.
    """
    try:
        tree = ast.parse(script, mode="exec")
    except SyntaxError:
        return False

    visitor = _SafeVisitor()
    try:
        visitor.visit(tree)
        return True
    except UnsafeOperation:
        return False
