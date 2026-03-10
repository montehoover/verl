import ast
import builtins as _builtins


class _SafeNodeVisitor(ast.NodeVisitor):
    UNSAFE_FUNCTIONS = {
        "open",
        "eval",
        "exec",
        "compile",
        "input",
        "__import__",
        "globals",
        "locals",
        "vars",
        "dir",
        "type",
        "help",
        "memoryview",
        "super",
        "getattr",
        "setattr",
        "delattr",
        "classmethod",
        "staticmethod",
        "property",
        "object",
        "bytes",
        "bytearray",
        "exit",
        "quit",
    }

    DISALLOWED_NAMES = {
        "__builtins__",
        "__loader__",
        "__spec__",
        "__package__",
        "__import__",
    }

    def visit_Attribute(self, node):
        raise ValueError("Attribute access is disallowed")

    def visit_Import(self, node):
        raise ValueError("Import statements are disallowed")

    def visit_ImportFrom(self, node):
        raise ValueError("Import statements are disallowed")

    def visit_Global(self, node):
        raise ValueError("global statements are disallowed")

    def visit_Nonlocal(self, node):
        raise ValueError("nonlocal statements are disallowed")

    def visit_With(self, node):
        raise ValueError("with statements are disallowed")

    def visit_Try(self, node):
        raise ValueError("try/except/finally statements are disallowed")

    def visit_Raise(self, node):
        raise ValueError("raise statements are disallowed")

    def visit_Delete(self, node):
        raise ValueError("del statements are disallowed")

    def visit_Lambda(self, node):
        raise ValueError("lambda expressions are disallowed")

    def visit_Await(self, node):
        raise ValueError("async/await is disallowed")

    def visit_Yield(self, node):
        raise ValueError("yield is disallowed")

    def visit_YieldFrom(self, node):
        raise ValueError("yield from is disallowed")

    def visit_ClassDef(self, node):
        raise ValueError("class definitions are disallowed")

    def visit_FunctionDef(self, node):
        # Allow function definitions but disallow decorators and async
        if node.decorator_list:
            raise ValueError("decorators are disallowed")
        if getattr(node, "returns", None) is not None:
            # Type annotations on return are expressions; conservative block
            raise ValueError("function return annotations are disallowed")
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node):
        raise ValueError("async functions are disallowed")

    def visit_Call(self, node):
        # Disallow calls to known-unsafe builtins by name
        func = node.func
        if isinstance(func, ast.Name):
            if func.id in self.UNSAFE_FUNCTIONS:
                raise ValueError(f"Disallowed function call: {func.id}")
        elif isinstance(func, ast.Attribute):
            # Attribute access already blocked, but keep explicit here
            raise ValueError("Attribute access is disallowed")
        # Disallow starargs/kwargs unpacking targets via attribute (implicitly handled)
        self.generic_visit(node)

    def visit_Name(self, node):
        # Disallow access to sensitive dunder names
        if node.id in self.DISALLOWED_NAMES:
            raise ValueError(f"Disallowed name: {node.id}")
        self.generic_visit(node)

    def visit_comprehension(self, node):
        if getattr(node, "is_async", False):
            raise ValueError("async comprehensions are disallowed")
        self.generic_visit(node)

    def visit_NamedExpr(self, node):
        # Walrus operator
        raise ValueError("assignment expressions are disallowed")


def run_user_script(user_script: str):
    """
    Execute a user-supplied Python script in a restricted environment.

    Args:
        user_script: str - The Python script provided by the user.

    Returns:
        The result of the script if any (value of the final expression), or None.

    Raises:
        ValueError: If the script contains disallowed operations or syntax errors.
        Other runtime exceptions produced by the script are propagated.
    """
    if not isinstance(user_script, str):
        raise ValueError("Script must be a string")

    try:
        tree = ast.parse(user_script, mode="exec")
    except SyntaxError as e:
        raise ValueError(f"Syntax error: {e}") from None

    # Validate AST for safety
    try:
        _SafeNodeVisitor().visit(tree)
    except ValueError:
        # Re-raise as-is for clarity
        raise
    except Exception as e:
        raise ValueError(f"Disallowed or unsafe operation: {e}") from None

    # Transform final expression to assignment to capture result
    result_var = "_script_result_"
    has_result_expr = False
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        last_expr_node = tree.body[-1]
        assign = ast.Assign(
            targets=[ast.Name(id=result_var, ctx=ast.Store())],
            value=last_expr_node.value,
        )
        ast.copy_location(assign, last_expr_node)
        tree.body[-1] = assign
        has_result_expr = True

    ast.fix_missing_locations(tree)

    # Build a restricted builtins dict
    SAFE_BUILTINS = {
        "abs",
        "all",
        "any",
        "bool",
        "chr",
        "complex",
        "dict",
        "divmod",
        "enumerate",
        "filter",
        "float",
        "format",
        "frozenset",
        "hash",
        "hex",
        "int",
        "isinstance",
        "issubclass",
        "iter",
        "len",
        "list",
        "map",
        "max",
        "min",
        "next",
        "oct",
        "ord",
        "pow",
        "print",
        "range",
        "repr",
        "reversed",
        "round",
        "set",
        "slice",
        "sorted",
        "str",
        "sum",
        "tuple",
        "zip",
    }
    safe_builtins = {name: getattr(_builtins, name) for name in SAFE_BUILTINS}

    # Prepare isolated globals/locals
    env = {
        "__builtins__": safe_builtins,
        "__name__": "__user__",
    }

    try:
        code = compile(tree, "<user_script>", "exec")
    except SyntaxError as e:
        raise ValueError(f"Syntax error: {e}") from None

    # Execute with identical globals/locals to keep scope simple
    exec(code, env, env)

    if has_result_expr:
        return env.get(result_var, None)
    return None
