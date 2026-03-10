import ast
import builtins
import math
import logging


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


RESULT_VAR = "__result__"


def _parse_python_code(python_code: str) -> ast.Module:
    """
    Parse Python code to an AST module, converting SyntaxError to ValueError.
    """
    try:
        return ast.parse(python_code, mode="exec")
    except SyntaxError as e:
        raise ValueError(f"Invalid syntax: {e.msg}") from None


def _validate_ast_safety(tree: ast.AST) -> None:
    """
    Validate the AST for prohibited operations. Raises ValueError if unsafe.
    """
    _SafeAstChecker().visit(tree)


def _capture_last_expr_result(tree: ast.Module, result_var: str = RESULT_VAR) -> ast.Module:
    """
    Return a new AST module where the last Expr statement is replaced by an
    assignment to result_var so its value can be returned post-exec.
    """
    # Create a shallow copy of the module body to avoid mutating the input tree
    body = list(tree.body)
    if body and isinstance(body[-1], ast.Expr):
        last = body[-1]
        assign = ast.Assign(
            targets=[ast.Name(id=result_var, ctx=ast.Store())],
            value=last.value,
            type_comment=None,
        )
        ast.copy_location(assign, last)
        body[-1] = assign

    # Preserve type_ignores if present (Python 3.8+)
    type_ignores = getattr(tree, "type_ignores", [])
    new_module = ast.Module(body=body, type_ignores=type_ignores)
    ast.fix_missing_locations(new_module)
    return new_module


def _build_safe_environment():
    """
    Build and return a restricted globals dictionary for safe execution.
    """
    return {
        "__builtins__": _build_safe_builtins(),
        "math": math,
    }


def _execute_ast_module(module: ast.Module, result_var: str = RESULT_VAR):
    """
    Compile and execute the given AST module in a restricted environment.
    Returns the captured result if present, otherwise None.
    """
    code_obj = compile(module, filename="<user_script>", mode="exec")
    env = _build_safe_environment()
    exec(code_obj, env, None)
    return env.get(result_var, None)


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
    # Initialize logging for this function
    logger = logging.getLogger("execute_python_script")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False

    if not isinstance(python_code, str):
        logger.error("Invalid input type for python_code: %s", type(python_code).__name__)
        raise ValueError("Invalid input: python_code must be a string")

    logger.info("Received script for execution (length=%d)", len(python_code))
    logger.info("Script content:\n%s", python_code)

    try:
        # Parse and validate
        tree = _parse_python_code(python_code)
        _validate_ast_safety(tree)

        # Transform to capture last expression result
        module = _capture_last_expr_result(tree, RESULT_VAR)

        # Execute in sandbox and return the result
        result = _execute_ast_module(module, RESULT_VAR)
        logger.info("Script executed successfully. Result: %r", result)
        return result
    except ValueError as ve:
        logger.error("Script rejected or invalid: %s", ve, exc_info=True)
        raise
    except Exception as ex:
        logger.error("Runtime error during script execution: %s", ex, exc_info=True)
        raise
