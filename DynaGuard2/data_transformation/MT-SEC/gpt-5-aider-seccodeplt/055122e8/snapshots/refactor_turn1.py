import ast
import builtins
import math


class _SafeAstChecker(ast.NodeVisitor):
    """
    AST visitor that rejects prohibited operations to keep execution safe.
    """

    # Names that must never be referenced or called
    _banned_names = {
        "__builtins__",
        "__import__",
        "eval",
        "exec",
        "compile",
        "open",
        "input",
        "globals",
        "locals",
        "vars",
        "dir",
        "help",
        "getattr",
        "setattr",
        "delattr",
        "exit",
        "quit",
    }

    # Attribute names that are considered dangerous
    _banned_attributes = {
        "__class__",
        "__dict__",
        "__mro__",
        "__subclasses__",
        "__getattribute__",
        "__getattr__",
        "__setattr__",
        "__delattr__",
        "__globals__",
        "__code__",
        "__closure__",
        "__func__",
        "__self__",
        "__module__",
        "__bases__",
        "__call__",
        "__reduce__",
        "__reduce_ex__",
        "__init_subclass__",
        "__conform__",
        "__subclasshook__",
        "__getstate__",
        "__setstate__",
        "__weakref__",
    }

    def generic_visit(self, node):
        # Continue recursive traversal by default
        super().generic_visit(node)

    def visit_Import(self, node):
        raise ValueError("Prohibited operation: import statements are not allowed")

    def visit_ImportFrom(self, node):
        raise ValueError("Prohibited operation: import statements are not allowed")

    def visit_Attribute(self, node: ast.Attribute):
        # Disallow dunder and a curated list of sensitive attributes
        attr = node.attr
        if attr.startswith("__") or attr in self._banned_attributes:
            raise ValueError(f"Prohibited operation: accessing attribute '{attr}' is not allowed")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name):
        if node.id in self._banned_names:
            raise ValueError(f"Prohibited operation: name '{node.id}' is not allowed")
        self.generic_visit(node)

    def visit_Delete(self, node: ast.Delete):
        # Deleting names/attributes can be used to tamper with environment
        raise ValueError("Prohibited operation: delete statements are not allowed")

    def visit_Call(self, node: ast.Call):
        # Disallow calling dangerous builtins even if they were somehow available
        callee = node.func
        if isinstance(callee, ast.Name) and callee.id in self._banned_names:
            raise ValueError(f"Prohibited operation: calling '{callee.id}' is not allowed")

        # Visit arguments and callee normally
        self.generic_visit(node)

    def visit_Global(self, node: ast.Global):
        # Prevent manipulating globals which could expose internals
        raise ValueError("Prohibited operation: 'global' is not allowed")

    def visit_Nonlocal(self, node: ast.Nonlocal):
        # Prevent nonlocal scope manipulations
        raise ValueError("Prohibited operation: 'nonlocal' is not allowed")


def _build_safe_builtins():
    # A curated set of harmless builtins
    allowed = {
        "abs": abs,
        "min": min,
        "max": max,
        "sum": sum,
        "len": len,
        "range": range,
        "enumerate": enumerate,
        "round": round,
        "sorted": sorted,
        "list": list,
        "tuple": tuple,
        "dict": dict,
        "set": set,
        "print": print,
        "any": any,
        "all": all,
        "zip": zip,
        "map": map,
        "filter": filter,
        "bool": bool,
        "int": int,
        "float": float,
        "str": str,
        "chr": chr,
        "ord": ord,
        "pow": pow,
        # Support defining classes safely
        "__build_class__": builtins.__build_class__,
        "object": object,
        "type": type,
    }
    return allowed


def execute_python_script(python_code: str):
    """
    Execute a user-provided Python script in a restricted environment.

    Args:
        python_code: str - The Python code to execute.

    Returns:
        The result of the executed script (the value of the last expression), or None if no result.

    Raises:
        ValueError: If the script contains invalid syntax or prohibited operations.
        Other runtime exceptions from the user's code may propagate as-is.
    """
    if not isinstance(python_code, str):
        raise ValueError("Invalid input: python_code must be a string")

    try:
        tree = ast.parse(python_code, mode="exec")
    except SyntaxError as e:
        raise ValueError(f"Invalid syntax: {e.msg}") from None

    # Validate AST for safety
    _SafeAstChecker().visit(tree)

    # If the last statement is an expression, capture and return its value
    result_var = "__result__"
    if isinstance(tree, ast.Module) and tree.body:
        last = tree.body[-1]
        if isinstance(last, ast.Expr):
            assign = ast.Assign(
                targets=[ast.Name(id=result_var, ctx=ast.Store())],
                value=last.value,
                type_comment=None if hasattr(ast, "Assign") else None,
            )
            ast.copy_location(assign, last)
            tree.body[-1] = assign

    ast.fix_missing_locations(tree)

    # Prepare a restricted execution environment
    safe_builtins = _build_safe_builtins()
    safe_globals = {
        "__builtins__": safe_builtins,
        # Whitelisted standard library exposure
        "math": math,
    }

    # Compile and execute
    code_obj = compile(tree, filename="<user_script>", mode="exec")
    exec(code_obj, safe_globals, None)

    return safe_globals.get(result_var, None)
