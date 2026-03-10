import ast
import io
from contextlib import redirect_stdout


class _SafetyChecker(ast.NodeVisitor):
    """
    Visits the AST and raises ValueError if any forbidden construct is found.
    """

    # Builtins that are safe to expose and allow calling
    ALLOWED_BUILTIN_CALLS = {
        "print",
        "len",
        "range",
        "sum",
        "min",
        "max",
        "sorted",
        "enumerate",
        "any",
        "all",
        "abs",
        "round",
        "bool",
        "int",
        "float",
        "str",
        "list",
        "dict",
        "set",
        "tuple",
        "zip",
    }

    # Names that should never be callable or referenced for safety
    FORBIDDEN_NAMES = {
        "__import__",
        "eval",
        "exec",
        "open",
        "compile",
        "input",
        "globals",
        "locals",
        "vars",
        "dir",
        "getattr",
        "setattr",
        "delattr",
        "help",
        "memoryview",
        "super",
        "type",  # block calling type() to prevent introspection tricks
    }

    # Node types that are outright forbidden
    FORBIDDEN_NODES = (
        ast.Import,
        ast.ImportFrom,
        ast.Attribute,        # obj.attr access can break sandboxing
        ast.With,
        ast.AsyncWith,
        ast.Try,
        ast.Raise,
        ast.ClassDef,
        ast.Global,
        ast.Nonlocal,
        ast.Delete,
        ast.Await,
        ast.AsyncFunctionDef,
        ast.Yield,
        ast.YieldFrom,
    )

    def __init__(self):
        self.defined_functions = set()

    def visit(self, node):
        if isinstance(node, self.FORBIDDEN_NODES):
            raise ValueError(f"Forbidden construct: {type(node).__name__}")
        return super().visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        # Disallow decorators for safety
        if node.decorator_list:
            raise ValueError("Decorators are not allowed.")
        # Record function name to allow calling it later
        self.defined_functions.add(node.name)
        # Validate arguments default values and body
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name):
        # Disallow double-underscore names (dunder)
        if node.id.startswith("__") and node.id.endswith("__"):
            raise ValueError("Access to dunder names is forbidden.")
        # Disallow referencing forbidden names at all
        if node.id in self.FORBIDDEN_NAMES:
            raise ValueError(f"Use of forbidden name: {node.id}")

    def visit_Call(self, node: ast.Call):
        # Only allow calls to:
        # - whitelisted builtins
        # - functions defined in the script itself
        func = node.func
        if isinstance(func, ast.Name):
            name = func.id
            if name in self.FORBIDDEN_NAMES:
                raise ValueError(f"Calling forbidden function: {name}")
            if (name not in self.ALLOWED_BUILTIN_CALLS) and (name not in self.defined_functions):
                raise ValueError(f"Calls are restricted. '{name}' is not allowed.")
        else:
            # Disallow calling via attributes, subscripts, lambdas, etc.
            raise ValueError("Only direct calls to allowed function names are permitted.")
        # Visit arguments
        self.generic_visit(node)

    # Ensure comprehensions and f-strings are still validated deeply
    def visit_JoinedStr(self, node: ast.JoinedStr):
        self.generic_visit(node)

    def visit_FormattedValue(self, node: ast.FormattedValue):
        self.generic_visit(node)


def execute_user_script(submitted_script: str):
    """
    Safely execute a user-provided Python script with restricted operations.

    Args:
        submitted_script (string): The script that the user submits for execution.

    Returns:
        The output produced by the script via stdout, if any; otherwise None.

    Raises:
        ValueError: If the script contains syntax errors or forbidden actions.
        Other exceptions raised by the script at runtime will propagate.
    """
    if not isinstance(submitted_script, str):
        raise ValueError("Submitted script must be a string.")

    # Parse and validate AST
    try:
        tree = ast.parse(submitted_script, mode="exec")
    except SyntaxError as e:
        raise ValueError(f"Syntax error in script: {e}") from e

    checker = _SafetyChecker()
    checker.visit(tree)

    # Prepare a restricted set of builtins
    safe_builtins = {
        "print": print,
        "len": len,
        "range": range,
        "sum": sum,
        "min": min,
        "max": max,
        "sorted": sorted,
        "enumerate": enumerate,
        "any": any,
        "all": all,
        "abs": abs,
        "round": round,
        "bool": bool,
        "int": int,
        "float": float,
        "str": str,
        "list": list,
        "dict": dict,
        "set": set,
        "tuple": tuple,
        "zip": zip,
        # True/False/None are constants in code, but ensure they exist too
        "True": True,
        "False": False,
        "None": None,
    }

    # Create a sandboxed global/locals namespace
    sandbox_globals = {"__builtins__": safe_builtins}
    # Use the same dict for locals so definitions persist within the same namespace
    sandbox_locals = sandbox_globals

    # Execute while capturing stdout
    buf = io.StringIO()
    with redirect_stdout(buf):
        exec(compile(tree, filename="<user_script>", mode="exec"), sandbox_globals, sandbox_locals)

    out = buf.getvalue()
    return out if out else None
