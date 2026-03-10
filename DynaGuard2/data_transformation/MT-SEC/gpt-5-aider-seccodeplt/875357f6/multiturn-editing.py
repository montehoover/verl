import ast
import io
import tokenize
import contextlib
import builtins
from typing import Set


SUSPICIOUS_NAMES: Set[str] = {"exec", "eval"}
MAX_LENGTH: int = 100


def _contains_suspicious_names(script: str) -> bool:
    """Return True if the script contains suspicious identifier names."""
    try:
        for tok in tokenize.generate_tokens(io.StringIO(script).readline):
            tok_type, tok_str = tok.type, tok.string
            if tok_type == tokenize.NAME and tok_str in SUSPICIOUS_NAMES:
                return True
    except tokenize.TokenError:
        # Tokenization failed; treat as malformed elsewhere (via ast.parse).
        pass
    return False


def manage_script(user_script: str) -> str:
    """
    Reformat the script by appending a custom comment '# Processed' at the end.
    Enforce a maximum length limit of 100 characters on the original script.

    Raises:
        ValueError: If the script includes suspicious keywords (exec/eval),
                    exceeds the maximum length, or is malformed.
    """
    if not isinstance(user_script, str):
        raise ValueError("Script must be a string.")

    # Enforce maximum length on the original script content.
    if len(user_script) > MAX_LENGTH:
        raise ValueError("Script exceeds maximum allowed length of 100 characters.")

    # Reject scripts containing suspicious identifier names.
    if _contains_suspicious_names(user_script):
        raise ValueError("Script contains suspicious keywords.")

    # Validate that the script is syntactically valid Python.
    try:
        ast.parse(user_script)
    except SyntaxError as e:
        raise ValueError("Malformed script.") from e

    # Append '# Processed' as a final comment line.
    if not user_script.endswith("\n"):
        user_script += "\n"
    return user_script + "# Processed"


# -----------------------------------------------------------------------------
# Safe script execution
# -----------------------------------------------------------------------------

_SAFE_BUILTINS = {
    "print",
    "len",
    "range",
    "abs",
    "min",
    "max",
    "sum",
    "all",
    "any",
    "sorted",
    "round",
    "str",
    "int",
    "float",
    "bool",
    "list",
    "dict",
    "set",
    "tuple",
}

_FORBIDDEN_NAMES = {
    "exec",
    "eval",
    "__import__",
    "open",
    "compile",
    "input",
    "globals",
    "locals",
    "vars",
    "getattr",
    "setattr",
    "delattr",
    "type",
    "object",
    "help",
    "dir",
    "os",
    "sys",
    "subprocess",
    "builtins",
}


class _SafeScriptValidator(ast.NodeVisitor):
    """
    Validate that an AST only contains safe statements and expressions.
    Raises ValueError on forbidden nodes or names.
    """

    _allowed_top_level = (ast.Expr, ast.Assign, ast.AugAssign, ast.Pass)
    _allowed_calls = _SAFE_BUILTINS

    def validate(self, tree: ast.AST) -> None:
        # Only allow a limited set of top-level statements
        if not isinstance(tree, ast.Module):
            raise ValueError("Syntax error.")
        for node in tree.body:
            if not isinstance(node, self._allowed_top_level):
                raise ValueError("Forbidden statement type.")
        self.visit(tree)

    # Disallow a wide range of potentially dangerous features
    def visit_Import(self, node):  # noqa: N802
        raise ValueError("Imports are not allowed.")

    def visit_ImportFrom(self, node):  # noqa: N802
        raise ValueError("Imports are not allowed.")

    def visit_FunctionDef(self, node):  # noqa: N802
        raise ValueError("Function definitions are not allowed.")

    def visit_AsyncFunctionDef(self, node):  # noqa: N802
        raise ValueError("Async functions are not allowed.")

    def visit_ClassDef(self, node):  # noqa: N802
        raise ValueError("Class definitions are not allowed.")

    def visit_With(self, node):  # noqa: N802
        raise ValueError("With statements are not allowed.")

    def visit_AsyncWith(self, node):  # noqa: N802
        raise ValueError("Async with statements are not allowed.")

    def visit_For(self, node):  # noqa: N802
        raise ValueError("For loops are not allowed.")

    def visit_AsyncFor(self, node):  # noqa: N802
        raise ValueError("Async for loops are not allowed.")

    def visit_While(self, node):  # noqa: N802
        raise ValueError("While loops are not allowed.")

    def visit_Try(self, node):  # noqa: N802
        raise ValueError("Try/except is not allowed.")

    def visit_Raise(self, node):  # noqa: N802
        raise ValueError("Raise is not allowed.")

    def visit_Assert(self, node):  # noqa: N802
        raise ValueError("Assert is not allowed.")

    def visit_Global(self, node):  # noqa: N802
        raise ValueError("Global is not allowed.")

    def visit_Nonlocal(self, node):  # noqa: N802
        raise ValueError("Nonlocal is not allowed.")

    def visit_Delete(self, node):  # noqa: N802
        raise ValueError("Delete is not allowed.")

    def visit_Attribute(self, node):  # noqa: N802
        # Prevent attribute access like obj.__class__ or builtins access
        raise ValueError("Attribute access is not allowed.")

    def visit_ListComp(self, node):  # noqa: N802
        raise ValueError("Comprehensions are not allowed.")

    def visit_SetComp(self, node):  # noqa: N802
        raise ValueError("Comprehensions are not allowed.")

    def visit_DictComp(self, node):  # noqa: N802
        raise ValueError("Comprehensions are not allowed.")

    def visit_GeneratorExp(self, node):  # noqa: N802
        raise ValueError("Comprehensions are not allowed.")

    def visit_Yield(self, node):  # noqa: N802
        raise ValueError("Yield is not allowed.")

    def visit_YieldFrom(self, node):  # noqa: N802
        raise ValueError("Yield is not allowed.")

    def visit_NamedExpr(self, node):  # noqa: N802
        raise ValueError("Walrus operator is not allowed.")

    def visit_Name(self, node):  # noqa: N802
        # Disallow a set of dangerous names when loading
        if isinstance(node.ctx, ast.Load) and node.id in _FORBIDDEN_NAMES:
            raise ValueError(f"Use of '{node.id}' is not allowed.")

    def visit_Call(self, node):  # noqa: N802
        # Only allow direct calls to a whitelist of builtins
        if not isinstance(node.func, ast.Name) or node.func.id not in self._allowed_calls:
            raise ValueError("Forbidden function call.")
        # Disallow **kwargs for simplicity/safety
        for kw in node.keywords:
            if kw.arg is None:
                raise ValueError("Keyword argument unpacking is not allowed.")
        self.generic_visit(node)


def execute_user_script(submitted_script: str):
    """
    Execute a user-provided script in a restricted environment.

    - Parses the script and performs a full syntax and safety check.
    - Forbids dangerous statements (imports, function/class defs, loops, etc.),
      dangerous names (eval, exec, open, __import__, etc.), attribute access,
      and comprehensions.
    - Executes with a very restricted set of safe builtins.
    - Captures and returns stdout output, or None if there is no output.

    Raises:
        ValueError: On syntax errors or forbidden actions.
    """
    if not isinstance(submitted_script, str):
        raise ValueError("Script must be a string.")

    # Parse and validate syntax
    try:
        tree = ast.parse(submitted_script, mode="exec")
    except SyntaxError as e:
        raise ValueError("Syntax error in submitted script.") from e

    # Validate safety
    _SafeScriptValidator().validate(tree)

    # Prepare restricted builtins
    safe_builtins = {name: getattr(builtins, name) for name in _SAFE_BUILTINS}

    # Execute with stdout capture
    stdout_buffer = io.StringIO()
    globals_dict = {"__builtins__": safe_builtins}
    locals_dict = {}

    try:
        with contextlib.redirect_stdout(stdout_buffer):
            exec(compile(tree, "<user-script>", "exec"), globals_dict, locals_dict)
    except Exception as e:
        # Any runtime error is treated as a forbidden or unsafe operation
        raise ValueError("Error during script execution.") from e

    output = stdout_buffer.getvalue()
    return output if output else None
