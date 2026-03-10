import ast
import io
import contextlib


class _SafetyChecker(ast.NodeVisitor):
    PROHIBITED_NAMES = {
        "__import__",
        "open",
        "eval",
        "exec",
        "compile",
        "getattr",
        "setattr",
        "delattr",
        "vars",
        "globals",
        "locals",
        "input",
        "help",
        "dir",
        "type",
        "object",
        "super",
        "memoryview",
        "bytearray",
        "bytes",
    }

    def visit_Import(self, node):
        raise ValueError("Import statements are not allowed.")

    def visit_ImportFrom(self, node):
        raise ValueError("Import statements are not allowed.")

    def visit_Global(self, node):
        raise ValueError("Global statement is not allowed.")

    def visit_Nonlocal(self, node):
        raise ValueError("Nonlocal statement is not allowed.")

    def visit_ClassDef(self, node):
        # Prevent class creation which relies on __build_class__ and metaclass tricks
        raise ValueError("Class definitions are not allowed.")

    def visit_Attribute(self, node):
        # Block access to double-underscore attributes like __class__, __dict__, etc.
        if isinstance(node.attr, str) and node.attr.startswith("__"):
            raise ValueError("Access to dunder attributes is not allowed.")
        self.generic_visit(node)

    def visit_Name(self, node):
        # Block direct usage of dangerous builtins by name
        if node.id in self.PROHIBITED_NAMES:
            raise ValueError(f"Use of '{node.id}' is not allowed.")
        self.generic_visit(node)

    def visit_Call(self, node):
        # Block calling dangerous functions via direct name or attribute
        func = node.func
        if isinstance(func, ast.Name) and func.id in self.PROHIBITED_NAMES:
            raise ValueError(f"Calling '{func.id}' is not allowed.")
        if isinstance(func, ast.Attribute):
            if isinstance(func.attr, str) and func.attr.startswith("__"):
                raise ValueError("Calling dunder attributes is not allowed.")
        self.generic_visit(node)


def _get_safe_builtins():
    """
    Return a fresh mapping of allowed builtins for executing user code.
    """
    return {
        # Simple types and constructors
        "bool": bool,
        "int": int,
        "float": float,
        "str": str,
        "list": list,
        "dict": dict,
        "set": set,
        "tuple": tuple,
        # Safe utilities
        "abs": abs,
        "all": all,
        "any": any,
        "enumerate": enumerate,
        "len": len,
        "max": max,
        "min": min,
        "range": range,
        "round": round,
        "sorted": sorted,
        "sum": sum,
        "zip": zip,
        "print": print,
        "repr": repr,
    }


def _validate_script(run_code):
    """
    Parse and statically validate the provided script.

    Returns:
        ast.AST: The parsed AST if validation passes.

    Raises:
        ValueError: If input type is invalid, parsing fails, or prohibited actions are detected.
    """
    if not isinstance(run_code, str):
        raise ValueError("Script must be a string.")

    try:
        tree = ast.parse(run_code, mode="exec")
    except SyntaxError as e:
        raise ValueError(f"Syntax error: {e.msg}") from None

    _SafetyChecker().visit(tree)
    return tree


def _execute_validated_ast(tree, builtins_map=None):
    """
    Execute a pre-validated AST with restricted builtins and capture stdout.

    Args:
        tree (ast.AST): Pre-validated AST to execute.
        builtins_map (dict | None): Custom builtins to use; defaults to a safe set.

    Returns:
        str | None: Captured stdout if any output was produced; otherwise None.

    Raises:
        ValueError: If a runtime error occurs during execution.
    """
    if builtins_map is None:
        builtins_map = _get_safe_builtins()

    safe_globals = {"__builtins__": builtins_map}

    stdout_buffer = io.StringIO()
    code_obj = compile(tree, "<user_code>", "exec")

    try:
        with contextlib.redirect_stdout(stdout_buffer):
            exec(code_obj, safe_globals, safe_globals)
    except Exception as e:
        raise ValueError(f"Runtime error: {e}") from None

    output = stdout_buffer.getvalue()
    return output if output != "" else None


def restricted_run_code(run_code):
    """
    Execute a user-provided Python script under restricted conditions.

    Args:
        run_code (str): The user's Python script.

    Returns:
        str | None: Captured stdout from the script if any was produced; otherwise None.

    Raises:
        ValueError: If the script contains prohibited actions, has syntax issues,
                    or encounters runtime errors during execution.
    """
    tree = _validate_script(run_code)
    return _execute_validated_ast(tree)
