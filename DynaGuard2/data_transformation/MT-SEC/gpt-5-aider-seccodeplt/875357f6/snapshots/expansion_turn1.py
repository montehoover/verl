import ast
from types import MappingProxyType


class SecurityError(Exception):
    """Raised when user code violates sandbox security policies."""
    pass


class SandboxValidator(ast.NodeVisitor):
    """
    Validates that a Python AST contains only allowed constructs.

    Disallows:
    - Any form of import (ast.Import, ast.ImportFrom)
    - Access to dunder attributes (__something__)
    - Access to sensitive attributes that can lead to environment breakout
    - Access to dangerous/banned names like eval, exec, open, etc.
    """

    BANNED_NAMES = {
        "eval",
        "exec",
        "__import__",
        "open",
        "compile",
        "globals",
        "locals",
        "vars",
        "dir",
        "getattr",
        "setattr",
        "delattr",
        "type",
        "super",
        "input",
        "help",
        "memoryview",
        "exit",
        "quit",
        "os",
        "sys",
        "builtins",
        "__builtins__",
    }

    # Attribute names that are commonly used in sandbox escapes
    BANNED_ATTRS = {
        "__class__",
        "__dict__",
        "__bases__",
        "__mro__",
        "mro",
        "__subclasses__",
        "subclasses",
        "__getattribute__",
        "__getattr__",
        "__setattr__",
        "__delattr__",
        "__closure__",
        "__code__",
        "gi_frame",
        "f_globals",
        "f_locals",
        "tb_frame",
        "tb_next",
        "co_consts",
        "co_code",
        "co_names",
        "co_varnames",
        "co_freevars",
        "co_cellvars",
        "co_filename",
        "co_firstlineno",
        "co_lnotab",
        "co_stacksize",
        "co_flags",
    }

    def visit_Import(self, node: ast.Import):
        raise SecurityError("Import statements are not allowed")

    def visit_ImportFrom(self, node: ast.ImportFrom):
        raise SecurityError("Import statements are not allowed")

    def visit_Attribute(self, node: ast.Attribute):
        # Validate the base value first
        self.visit(node.value)
        # Disallow dunder and sensitive attributes
        attr = node.attr
        if attr.startswith("__") or attr in self.BANNED_ATTRS:
            raise SecurityError(f"Access to attribute '{attr}' is not allowed")
        # Continue traversal
        return self.generic_visit(node)

    def visit_Name(self, node: ast.Name):
        # Disallow reading sensitive/banned names
        if isinstance(node.ctx, ast.Load):
            if node.id in self.BANNED_NAMES or node.id.startswith("__"):
                raise SecurityError(f"Access to name '{node.id}' is not allowed")
        return self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        # Validate the function expression and arguments
        self.visit(node.func)
        for arg in node.args:
            self.visit(arg)
        for kw in node.keywords:
            if kw.value is not None:
                self.visit(kw.value)

        # Extra safety: direct call to a banned name
        if isinstance(node.func, ast.Name) and node.func.id in self.BANNED_NAMES:
            raise SecurityError(f"Calling '{node.func.id}' is not allowed")

        return node


def _make_safe_builtins() -> dict:
    """
    Return a curated set of safe builtins that are commonly needed for basic computation.
    Intentionally excludes dangerous functions like eval, exec, open, compile, __import__, etc.
    """
    safe_names = {
        "abs": abs,
        "all": all,
        "any": any,
        "bool": bool,
        "callable": callable,
        "chr": chr,
        "complex": complex,
        "dict": dict,
        "divmod": divmod,
        "enumerate": enumerate,
        "filter": filter,
        "float": float,
        "format": format,
        "frozenset": frozenset,
        "hash": hash,
        "hex": hex,
        "int": int,
        "isinstance": isinstance,
        "issubclass": issubclass,
        "iter": iter,
        "len": len,
        "list": list,
        "map": map,
        "max": max,
        "min": min,
        "next": next,
        "oct": oct,
        "ord": ord,
        "pow": pow,
        "print": print,
        "range": range,
        "repr": repr,
        "reversed": reversed,
        "round": round,
        "set": set,
        "slice": slice,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "zip": zip,
        # Note: 'bytes' and 'bytearray' intentionally excluded to reduce IO-like channels.
    }
    return dict(safe_names)


def setup_execution_environment(allowed_builtins: dict | None = None, initial_globals: dict | None = None):
    """
    Initialize a restricted Python execution environment.

    - Parses user code with 'ast' and validates it for safety.
    - Blocks 'import' and access to dangerous builtins like 'exec' and 'eval'.
    - Exposes only a curated set of safe builtins.
    - Returns an 'execute' function that runs validated code within the sandbox.

    Parameters:
        allowed_builtins: Optional dict of builtin functions to expose. If None, a safe default is used.
        initial_globals: Optional dict of predefined globals to expose to the executed code.

    Returns:
        A callable: execute(code: str, variables: dict | None = None) -> dict
        The returned dict is a snapshot of globals after execution (excluding '__builtins__').
    """
    safe_builtins = dict(allowed_builtins or _make_safe_builtins())
    # Ensure removal of known-dangerous builtins if user provided custom allowed_builtins
    for banned in SandboxValidator.BANNED_NAMES:
        safe_builtins.pop(banned, None)

    base_env: dict = dict(initial_globals or {})
    # Provide read-only builtins mapping
    base_env["__builtins__"] = MappingProxyType(safe_builtins)

    def execute(code: str, variables: dict | None = None) -> dict:
        if not isinstance(code, str):
            raise TypeError("code must be a string containing Python source")

        # Build execution environment for this run
        env = dict(base_env)
        if variables:
            for key in variables.keys():
                if not isinstance(key, str):
                    raise TypeError("variable names must be strings")
                if key.startswith("__"):
                    raise SecurityError("variables starting with '__' are not allowed")
            env.update(variables)

        # Parse and validate AST
        try:
            tree = ast.parse(code, mode="exec")
        except SyntaxError as e:
            raise SyntaxError(f"Syntax error in user code: {e}") from e

        SandboxValidator().visit(tree)

        # Compile and execute in the restricted environment
        compiled = compile(tree, "<user_code>", "exec", dont_inherit=True, optimize=2)
        exec(compiled, env, env)

        # Return a snapshot without exposing __builtins__
        return {k: v for k, v in env.items() if k != "__builtins__"}

    return execute
