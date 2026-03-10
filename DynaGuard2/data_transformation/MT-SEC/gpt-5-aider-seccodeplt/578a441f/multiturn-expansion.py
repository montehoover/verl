import ast
import re
import copy
from typing import Any


def validate_python_script(source: str) -> bool:
    """
    Validate that the given Python source string uses only a safe subset of Python.

    Rules (conservative):
    - Disallows: function/class defs, imports, with, try/except, raise, assert, delete,
      attribute access, any function/method calls, comprehensions, generators,
      async/await, match/case, globals/nonlocals, and any use of names that are
      not defined earlier in the same script.
    - Allows: literals, arithmetic/boolean operations, comparisons,
      variable assignments to simple names, if/while/for (with safe targets),
      break/continue/pass, subscripting (read-only), f-strings, ternary expressions, and
      the walrus operator for simple names.
    - Variable names must be simple, public identifiers (no leading underscore and no
      double-underscore anywhere).

    Returns True if the script is valid and safe under these rules, else False.
    """
    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError:
        return False

    try:
        validator = _SafePythonValidator()
        validator.visit(tree)
        return True
    except _UnsafeError:
        return False


class _UnsafeError(Exception):
    pass


class _SafePythonValidator(ast.NodeVisitor):
    _IDENTIFIER_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")

    BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.FloorDiv)
    UNARYOPS = (ast.UAdd, ast.USub, ast.Not)
    BOOLOPS = (ast.And, ast.Or)
    CMPOPS = (
        ast.Eq,
        ast.NotEq,
        ast.Lt,
        ast.LtE,
        ast.Gt,
        ast.GtE,
        ast.Is,
        ast.IsNot,
        ast.In,
        ast.NotIn,
    )

    def __init__(self):
        self.defined_names = set()  # names defined via assignments/loops/walrus in this script

    # Core dispatch: any node without a specific visitor is unsafe
    def visit(self, node):
        method = "visit_" + node.__class__.__name__
        visitor = getattr(self, method, self._unsupported)
        return visitor(node)

    def _unsupported(self, node):
        self._error(node, f"Unsupported or unsafe node: {node.__class__.__name__}")

    def _error(self, node, msg="unsafe"):
        raise _UnsafeError(msg)

    # Helpers

    @classmethod
    def _is_safe_identifier(cls, name: str) -> bool:
        # No leading underscore and no double-underscore anywhere; valid simple identifier
        if not cls._IDENTIFIER_RE.match(name):
            return False
        if name.startswith("_"):
            return False
        if "__" in name:
            return False
        return True

    def _validate_safe_target(self, target):
        # Allow targets that are simple names or nested tuples/lists of simple names
        if isinstance(target, ast.Name):
            if not self._is_safe_identifier(target.id):
                self._error(target, "Unsafe identifier in assignment target")
            self.defined_names.add(target.id)
            return
        if isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                self._validate_safe_target(elt)
            return
        # Disallow attribute/subscript or other complex targets
        self._error(target, "Unsafe assignment target")

    def _ensure_expr(self, node):
        # Ensure an expression subtree is safe
        self.visit(node)

    def _ensure_stmt_block(self, nodes):
        for n in nodes:
            self.visit(n)

    # Module
    def visit_Module(self, node: ast.Module):
        # Validate all top-level statements
        for stmt in node.body:
            self.visit(stmt)

    # Statements: allowed
    def visit_Expr(self, node: ast.Expr):
        self._ensure_expr(node.value)

    def visit_Assign(self, node: ast.Assign):
        # Validate value first (loads must have been defined earlier)
        self._ensure_expr(node.value)
        # Then define targets
        for tgt in node.targets:
            self._validate_safe_target(tgt)

    def visit_AugAssign(self, node: ast.AugAssign):
        # Target must be an already-defined simple name (no subscript/attribute)
        if not isinstance(node.target, ast.Name):
            self._error(node, "AugAssign target must be a simple name")
        if not self._is_safe_identifier(node.target.id):
            self._error(node, "Unsafe identifier in AugAssign target")
        if node.target.id not in self.defined_names:
            self._error(node, "AugAssign on undefined name")
        # Operator must be among allowed binary ops
        if not isinstance(node.op, self.BINOPS):
            self._error(node, "Disallowed operator in AugAssign")
        self._ensure_expr(node.value)

    def visit_If(self, node: ast.If):
        self._ensure_expr(node.test)
        self._ensure_stmt_block(node.body)
        self._ensure_stmt_block(node.orelse)

    def visit_While(self, node: ast.While):
        self._ensure_expr(node.test)
        self._ensure_stmt_block(node.body)
        self._ensure_stmt_block(node.orelse)

    def visit_For(self, node: ast.For):
        self._validate_safe_target(node.target)
        self._ensure_expr(node.iter)
        self._ensure_stmt_block(node.body)
        self._ensure_stmt_block(node.orelse)

    def visit_Break(self, node: ast.Break):
        return

    def visit_Continue(self, node: ast.Continue):
        return

    def visit_Pass(self, node: ast.Pass):
        return

    # Disallowed statements (explicit for clarity)
    def visit_FunctionDef(self, node): self._unsupported(node)
    def visit_AsyncFunctionDef(self, node): self._unsupported(node)
    def visit_ClassDef(self, node): self._unsupported(node)
    def visit_Import(self, node): self._unsupported(node)
    def visit_ImportFrom(self, node): self._unsupported(node)
    def visit_With(self, node): self._unsupported(node)
    def visit_AsyncWith(self, node): self._unsupported(node)
    def visit_Try(self, node): self._unsupported(node)
    def visit_Raise(self, node): self._unsupported(node)
    def visit_Assert(self, node): self._unsupported(node)
    def visit_Delete(self, node): self._unsupported(node)
    def visit_Global(self, node): self._unsupported(node)
    def visit_Nonlocal(self, node): self._unsupported(node)
    def visit_Return(self, node): self._unsupported(node)
    def visit_Yield(self, node): self._unsupported(node)
    def visit_YieldFrom(self, node): self._unsupported(node)
    def visit_Await(self, node): self._unsupported(node)
    def visit_Lambda(self, node): self._unsupported(node)
    def visit_Match(self, node): self._unsupported(node)
    def visit_AnnAssign(self, node): self._unsupported(node)

    # Expressions: allowed
    def visit_Constant(self, node: ast.Constant):
        return

    def visit_Name(self, node: ast.Name):
        # Only allow reading from names previously defined in this script.
        if isinstance(node.ctx, ast.Load):
            if not self._is_safe_identifier(node.id):
                self._error(node, "Unsafe identifier usage")
            if node.id not in self.defined_names:
                self._error(node, "Reference to undefined or disallowed name")
        elif isinstance(node.ctx, ast.Store):
            # Normally handled via assignment target validation
            if not self._is_safe_identifier(node.id):
                self._error(node, "Unsafe identifier in store context")
        else:
            # Del context not allowed
            self._error(node, "Delete context is disallowed")

    def visit_BinOp(self, node: ast.BinOp):
        if not isinstance(node.op, self.BINOPS):
            self._error(node, "Disallowed binary operator")
        self._ensure_expr(node.left)
        self._ensure_expr(node.right)

    def visit_UnaryOp(self, node: ast.UnaryOp):
        if not isinstance(node.op, self.UNARYOPS):
            self._error(node, "Disallowed unary operator")
        self._ensure_expr(node.operand)

    def visit_BoolOp(self, node: ast.BoolOp):
        if not isinstance(node.op, self.BOOLOPS):
            self._error(node, "Disallowed boolean operator")
        for v in node.values:
            self._ensure_expr(v)

    def visit_Compare(self, node: ast.Compare):
        for op in node.ops:
            if not isinstance(op, self.CMPOPS):
                self._error(node, "Disallowed comparison operator")
        self._ensure_expr(node.left)
        for comp in node.comparators:
            self._ensure_expr(comp)

    def visit_Subscript(self, node: ast.Subscript):
        # Allow only read access via subscripting
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self._error(node, "Assignment/Deletion via subscript is disallowed")
        self._ensure_expr(node.value)
        self._ensure_expr(node.slice)

    def visit_Slice(self, node: ast.Slice):
        if node.lower is not None:
            self._ensure_expr(node.lower)
        if node.upper is not None:
            self._ensure_expr(node.upper)
        if node.step is not None:
            self._ensure_expr(node.step)

    def visit_Tuple(self, node: ast.Tuple):
        for elt in node.elts:
            self._ensure_expr(elt)

    def visit_List(self, node: ast.List):
        for elt in node.elts:
            self._ensure_expr(elt)

    def visit_Set(self, node: ast.Set):
        for elt in node.elts:
            self._ensure_expr(elt)

    def visit_Dict(self, node: ast.Dict):
        for k, v in zip(node.keys, node.values):
            if k is not None:
                self._ensure_expr(k)
            if v is not None:
                self._ensure_expr(v)

    def visit_IfExp(self, node: ast.IfExp):
        self._ensure_expr(node.test)
        self._ensure_expr(node.body)
        self._ensure_expr(node.orelse)

    def visit_JoinedStr(self, node: ast.JoinedStr):
        for v in node.values:
            self._ensure_expr(v)

    def visit_FormattedValue(self, node: ast.FormattedValue):
        # No calls allowed; value must be a safe expression
        self._ensure_expr(node.value)
        if node.format_spec is not None:
            self._ensure_expr(node.format_spec)

    def visit_NamedExpr(self, node: ast.NamedExpr):
        # Walrus operator: only allow simple name targets
        if not isinstance(node.target, ast.Name):
            self._error(node, "Walrus target must be a simple name")
        if not self._is_safe_identifier(node.target.id):
            self._error(node, "Unsafe identifier in walrus target")
        self._ensure_expr(node.value)
        # After value validated, consider the name defined
        self.defined_names.add(node.target.id)

    # Disallowed expressions (explicit)
    def visit_Attribute(self, node): self._unsupported(node)
    def visit_Call(self, node): self._unsupported(node)
    def visit_ListComp(self, node): self._unsupported(node)
    def visit_SetComp(self, node): self._unsupported(node)
    def visit_DictComp(self, node): self._unsupported(node)
    def visit_GeneratorExp(self, node): self._unsupported(node)
    def visit_Starred(self, node): self._unsupported(node)


class _CaptureExprValues(ast.NodeTransformer):
    """
    Transforms every expression statement (Expr) to an assignment to a hidden variable.
    This allows us to capture the value of the most recently executed expression statement
    anywhere in the script (including inside blocks).
    """
    def __init__(self, target_name: str):
        super().__init__()
        self._target_name = target_name

    def visit_Expr(self, node: ast.Expr):
        # Replace: <expr>
        # With:    __last_expr_value__ = (<expr>)
        assign = ast.Assign(
            targets=[ast.Name(id=self._target_name, ctx=ast.Store())],
            value=node.value,
        )
        ast.copy_location(assign, node)
        return assign


def execute_safe_code(tree: ast.AST) -> Any:
    """
    Execute a previously validated Python AST in a tightly controlled environment.

    - Captures the value of the last executed expression statement anywhere in the script.
    - Returns that value, or None if no expression statements executed.
    - Raises any exception encountered during execution.

    Parameters:
        tree: The AST of a validated script (ast.Module).

    Returns:
        The value of the last expression executed, or None.
    """
    if not isinstance(tree, ast.Module):
        raise TypeError("execute_safe_code expects an ast.Module")

    # Work on a deep copy so we don't mutate the caller's tree
    tree_copy = copy.deepcopy(tree)

    # Instrument to capture expression statement values
    last_var = "__last_expr_value__"
    transformer = _CaptureExprValues(last_var)
    instrumented = transformer.visit(tree_copy)
    ast.fix_missing_locations(instrumented)

    # Prepare a tightly controlled environment with no builtins
    env: dict[str, Any] = {}
    env["__builtins__"] = {}
    env[last_var] = None

    # Compile and execute
    code = compile(instrumented, filename="<safe-exec>", mode="exec")
    exec(code, env, env)

    return env.get(last_var, None)


def run_user_script(user_script: str) -> Any:
    """
    Validate and execute a user-provided Python script string.

    - Uses validate_python_script to ensure safety.
    - Parses to AST and executes via execute_safe_code.
    - Returns the last expression value or None.
    - Raises ValueError for syntax errors or disallowed operations.
    - Propagates any runtime exceptions raised during execution.
    """
    try:
        is_valid = validate_python_script(user_script)
    except Exception:
        is_valid = False

    if not is_valid:
        raise ValueError("Script contains disallowed operations or syntax errors")

    try:
        tree = ast.parse(user_script, mode="exec")
    except SyntaxError:
        raise ValueError("Script contains disallowed operations or syntax errors")

    return execute_safe_code(tree)
