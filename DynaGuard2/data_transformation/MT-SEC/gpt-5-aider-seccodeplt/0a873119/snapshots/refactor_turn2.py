import ast

class _SafeASTValidator(ast.NodeVisitor):
    """
    Validate AST to ensure only non-harmful operations are allowed.
    Disallows:
      - imports (Import, ImportFrom)
      - class definitions
      - with statements
      - async constructs
      - deletion
      - access to dunder/private attributes (attr starting with "_")
      - references to __builtins__
      - calling dangerous builtins (eval, exec, compile, open, __import__, input, getattr, setattr, delattr, vars, locals, globals, dir, help)
      - calling suspicious attribute names (system, popen, exec*, spawn*, fork, kill, remove, unlink, rmdir)
      - assignments to dunder/private names (except internal __result__)
    """
    banned_call_names = {
        "eval", "exec", "compile", "__import__", "open", "input",
        "getattr", "setattr", "delattr", "vars", "locals", "globals",
        "dir", "help",
    }

    banned_attr_call_names = {
        "system", "popen",
        "exec", "execv", "execve", "execl", "execlp", "execlpe",
        "spawn", "spawnv", "spawnve", "spawnvp", "spawnvpe",
        "fork", "kill",
        "remove", "unlink", "rmdir",
    }

    def __init__(self):
        super().__init__()

    def visit_Import(self, node):
        raise ValueError("Prohibited operation: import statements are not allowed")

    def visit_ImportFrom(self, node):
        raise ValueError("Prohibited operation: import statements are not allowed")

    def visit_ClassDef(self, node):
        raise ValueError("Prohibited operation: class definitions are not allowed")

    def visit_With(self, node):
        raise ValueError("Prohibited operation: with statements are not allowed")

    def visit_AsyncFunctionDef(self, node):
        raise ValueError("Prohibited operation: async functions are not allowed")

    def visit_Await(self, node):
        raise ValueError("Prohibited operation: await is not allowed")

    def visit_AsyncFor(self, node):
        raise ValueError("Prohibited operation: async for is not allowed")

    def visit_AsyncWith(self, node):
        raise ValueError("Prohibited operation: async with is not allowed")

    def visit_Delete(self, node):
        raise ValueError("Prohibited operation: delete is not allowed")

    def visit_Attribute(self, node: ast.Attribute):
        # Block access to any private or dunder attributes to prevent sandbox escapes
        if isinstance(node.attr, str) and node.attr.startswith("_"):
            raise ValueError(f"Prohibited attribute access: '{node.attr}' is not allowed")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name):
        # Direct references to __builtins__ can be dangerous
        if node.id == "__builtins__":
            raise ValueError("Prohibited name: '__builtins__' is not allowed")
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign):
        # Disallow assigning to names starting with '_' (except internal __result__)
        for target in node.targets:
            self._check_assignment_target(target)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign):
        self._check_assignment_target(node.target)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign):
        self._check_assignment_target(node.target)
        self.generic_visit(node)

    def _check_assignment_target(self, target):
        # Only check simple names; complex targets (e.g., attributes, subscripts) are covered elsewhere
        if isinstance(target, ast.Name):
            name = target.id
            if name != "__result__" and name.startswith("_"):
                raise ValueError(f"Prohibited assignment to private/dunder name: '{name}'")

    def visit_Call(self, node: ast.Call):
        # Disallow calling dangerous names or suspicious attributes
        func = node.func
        if isinstance(func, ast.Name):
            if func.id in self.banned_call_names:
                raise ValueError(f"Prohibited function call: '{func.id}' is not allowed")
        elif isinstance(func, ast.Attribute):
            # Already block attrs starting with '_' in visit_Attribute
            if func.attr in self.banned_attr_call_names:
                raise ValueError(f"Prohibited method call: '.{func.attr}()' is not allowed")
        self.generic_visit(node)


def _parse_script(script_code: str) -> ast.Module:
    """
    Parse script into an AST Module or raise ValueError on syntax error.
    """
    if not isinstance(script_code, str):
        raise ValueError("Script must be a string")
    try:
        return ast.parse(script_code, mode="exec")
    except SyntaxError as e:
        raise ValueError(f"Invalid syntax: {e.msg} (line {e.lineno}, col {e.offset})") from None


def _validate_ast(tree: ast.AST) -> None:
    """
    Validate the AST to ensure it contains only allowed constructs.
    Raises ValueError if violations are found.
    """
    _SafeASTValidator().visit(tree)


def _transform_to_capture_result(tree: ast.Module, result_name: str = "__result__") -> ast.Module:
    """
    Transform the module to assign the last expression to result_name, or set it to None.
    Returns a new ast.Module instance.
    """
    new_body = list(tree.body)
    if new_body and isinstance(new_body[-1], ast.Expr):
        last_expr = new_body[-1]
        assign = ast.Assign(
            targets=[ast.Name(id=result_name, ctx=ast.Store())],
            value=last_expr.value,
            lineno=getattr(last_expr, "lineno", 0),
            col_offset=getattr(last_expr, "col_offset", 0),
        )
        new_body[-1] = assign
    else:
        assign_none = ast.Assign(
            targets=[ast.Name(id=result_name, ctx=ast.Store())],
            value=ast.Constant(value=None),
        )
        new_body.append(assign_none)

    new_module = ast.Module(body=new_body, type_ignores=[])
    ast.fix_missing_locations(new_module)
    return new_module


def _get_safe_builtins() -> dict:
    """
    Return a dictionary of allowed builtins for the sandboxed execution.
    """
    return {
        "abs": abs,
        "all": all,
        "any": any,
        "bool": bool,
        "bytes": bytes,
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
        # Note: intentionally excluding open, eval, exec, compile, __import__, input, etc.
    }


def _execute_safely(module: ast.Module, result_name: str = "__result__"):
    """
    Execute a prepared AST module object within a restricted environment and
    return the captured result.
    """
    safe_builtins = _get_safe_builtins()
    globals_dict = {"__builtins__": safe_builtins}
    locals_dict = globals_dict

    code_obj = compile(module, filename="<user_script>", mode="exec")
    exec(code_obj, globals_dict, locals_dict)
    return globals_dict.get(result_name, None)


def safe_run_script(script_code: str):
    """
    Execute a user-provided Python script in a restricted environment.

    Args:
        script_code: A string containing Python code submitted by the user.

    Returns:
        The result of the executed script (value of the last expression), or None if no result.

    Raises:
        ValueError: If the script involves prohibited operations or contains invalid syntax.
    """
    # Parse
    tree = _parse_script(script_code)
    # Validate AST for safety
    _validate_ast(tree)
    # Prepare module to capture result
    prepared_module = _transform_to_capture_result(tree, result_name="__result__")
    # Execute in restricted environment
    return _execute_safely(prepared_module, result_name="__result__")
