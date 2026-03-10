import ast
from typing import List, Optional, Set, Dict, Any


# Policy sets shared by analysis and execution
BANNED_MODULE_NAMES: Set[str] = {
    "os",
    "sys",
    "subprocess",
    "socket",
    "shutil",
    "ctypes",
    "importlib",
    "pathlib",
    "signal",
    "threading",
    "multiprocessing",
    "resource",
    "fcntl",
    "select",
    "builtins",
    "__builtins__",
}

BANNED_CALL_NAMES: Set[str] = {
    # dynamic execution / code access
    "exec",
    "eval",
    "compile",
    "__import__",

    # filesystem / process / io
    "open",
    "memoryview",

    # environment / reflection-like
    "globals",
    "locals",
    "vars",
    "dir",
    "delattr",
    "setattr",
    "getattr",

    # interactive / debug
    "input",
    "breakpoint",

    # exit
    "exit",
    "quit",
}

SAFE_BUILTIN_CALLS: Set[str] = {
    "print",
    "len",
    "range",
    "enumerate",
    "abs",
    "min",
    "max",
    "sum",
    "all",
    "any",
    "sorted",
    "reversed",
    "int",
    "float",
    "str",
    "bool",
    "list",
    "dict",
    "set",
    "tuple",
    "zip",
    "map",
    "filter",
    "pow",
    "round",
    "divmod",
    "chr",
    "ord",
    "hash",
    "format",
    "bin",
    "oct",
    "hex",
    "slice",
}

BANNED_ATTR_NAMES: Set[str] = {
    # dunder-ish and reflective
    "__dict__",
    "__class__",
    "__subclasses__",
    "__mro__",
    "__getattribute__",
    "__getattr__",
    "__setattr__",
    "__delattr__",
    "__globals__",
    "__code__",
    "__closure__",
    "__func__",
    "__self__",
    "__call__",
    "mro",

    # file/process/networking risk
    "write",
    "writelines",
    "close",
    "fileno",
    "system",
    "popen",
    "Popen",
    "spawn",
    "fork",
    "execv",
    "execl",
    "open",
    "start",
    "kill",
    "terminate",
    "send",
    "connect",
    "bind",
    "listen",
    "accept",
    "setuid",
    "setgid",
    "mkfifo",
    "openat",
    "remove",
    "rmdir",
    "unlink",
    "chmod",
    "chown",
    "symlink",
    "check_output",
    "run",
    "call",
}


def analyze_ast_for_safety(code: str) -> bool:
    """
    Analyze Python source code for potentially harmful operations using the AST.

    Returns True if the script is considered safe under a conservative policy, False otherwise.
    """

    class SafetyVisitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.safe: bool = True
            self.reasons: List[str] = []
            self.defined_functions: Set[str] = set()
            self.defined_classes: Set[str] = set()
            self.lambda_functions: Set[str] = set()

        def mark_unsafe(self, node: ast.AST, reason: str) -> None:
            if self.safe:
                self.safe = False
                self.reasons.append(reason)

        def generic_visit(self, node: ast.AST) -> None:
            if not self.safe:
                return
            super().generic_visit(node)

        # Utility helpers

        def _get_attr_chain(self, node: ast.AST) -> Optional[List[str]]:
            # Returns the attribute chain as list of names, or None if cannot determine.
            parts: List[str] = []
            cur = node
            while True:
                if isinstance(cur, ast.Name):
                    parts.append(cur.id)
                    break
                elif isinstance(cur, ast.Attribute):
                    parts.append(cur.attr)
                    cur = cur.value
                    continue
                else:
                    return None
            parts.reverse()
            return parts

        # Node visitors

        def visit_Import(self, node: ast.Import) -> None:
            self.mark_unsafe(node, "Import statements are not allowed.")

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            self.mark_unsafe(node, "Import statements are not allowed.")

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self.defined_functions.add(node.name)
            if node.decorator_list:
                self.mark_unsafe(node, "Decorators are not allowed.")
                return
            self.generic_visit(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self.defined_functions.add(node.name)
            if node.decorator_list:
                self.mark_unsafe(node, "Decorators are not allowed.")
                return
            self.generic_visit(node)

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            self.defined_classes.add(node.name)
            if node.decorator_list:
                self.mark_unsafe(node, "Decorators are not allowed.")
                return
            # Disallow metaclass or base calls that evaluate callables
            for base in node.bases:
                if isinstance(base, ast.Call):
                    self.mark_unsafe(node, "Calling in class bases is not allowed.")
                    return
            self.generic_visit(node)

        def visit_Assign(self, node: ast.Assign) -> None:
            # Capture lambda assigned to name as allowed callable
            if isinstance(node.value, ast.Lambda):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self.lambda_functions.add(target.id)
            self.generic_visit(node)

        def visit_Name(self, node: ast.Name) -> None:
            if isinstance(node.ctx, ast.Load):
                if node.id in ("__builtins__", "builtins"):
                    self.mark_unsafe(node, "Access to builtins namespace is not allowed.")
                    return
            self.generic_visit(node)

        def visit_Attribute(self, node: ast.Attribute) -> None:
            # Disallow dunder attribute access and explicitly banned attributes
            if node.attr in BANNED_ATTR_NAMES or (node.attr.startswith("__") and node.attr.endswith("__")):
                self.mark_unsafe(node, f"Access to attribute '{node.attr}' is not allowed.")
                return

            chain = self._get_attr_chain(node)
            if chain:
                # If attribute chain begins with a banned module or builtins, disallow
                if chain[0] in BANNED_MODULE_NAMES:
                    self.mark_unsafe(node, f"Access to module '{chain[0]}' is not allowed.")
                    return
            self.generic_visit(node)

        def visit_Call(self, node: ast.Call) -> None:
            chain = self._get_attr_chain(node.func)

            # Calls to unknown or dangerous functions are disallowed unless:
            # - they are defined in this module (functions/classes/lambdas), or
            # - they are in a safe builtins whitelist.
            if chain is None:
                self.mark_unsafe(node, "Calling dynamic or complex callables is not allowed.")
                return

            root = chain[0]
            callee = chain[-1]

            # Builtins/banned modules access
            if root in ("__builtins__", "builtins"):
                self.mark_unsafe(node, "Calling via builtins namespace is not allowed.")
                return

            if root in BANNED_MODULE_NAMES:
                self.mark_unsafe(node, f"Calls on module '{root}' are not allowed.")
                return

            # Banned attributes anywhere in chain
            for part in chain:
                if part in BANNED_ATTR_NAMES:
                    self.mark_unsafe(node, f"Use of attribute '{part}' is not allowed.")
                    return

            # Allow local definitions (functions, classes, lambda-assigned names)
            if len(chain) == 1:
                if callee in self.defined_functions or callee in self.defined_classes or callee in self.lambda_functions:
                    self.generic_visit(node)
                    return

                # Safe builtins
                if callee in SAFE_BUILTIN_CALLS:
                    self.generic_visit(node)
                    return

                # Explicitly banned call names
                if callee in BANNED_CALL_NAMES:
                    self.mark_unsafe(node, f"Call to '{callee}' is not allowed.")
                    return

                # Unknown top-level callable -> unsafe
                self.mark_unsafe(node, f"Call to unknown function '{callee}' is not allowed.")
                return
            else:
                # Attribute calls: further restrict if they look suspicious
                if chain[0] in BANNED_MODULE_NAMES:
                    self.mark_unsafe(node, f"Calls on module '{chain[0]}' are not allowed.")
                    return

                # Avoid calling attributes with dunder name at tail
                if callee.startswith("__") and callee.endswith("__"):
                    self.mark_unsafe(node, f"Call to dunder '{callee}' is not allowed.")
                    return

                # Otherwise allowed (subject to traversal of args/keywords)
                self.generic_visit(node)
                return

        def visit_Lambda(self, node: ast.Lambda) -> None:
            # Allow but ensure body is safe
            self.generic_visit(node)

        def visit_DictComp(self, node: ast.DictComp) -> None:
            self.generic_visit(node)

        def visit_SetComp(self, node: ast.SetComp) -> None:
            self.generic_visit(node)

        def visit_ListComp(self, node: ast.ListComp) -> None:
            self.generic_visit(node)

        def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
            self.generic_visit(node)

        def visit_Await(self, node: ast.Await) -> None:
            self.generic_visit(node)

        def visit_Yield(self, node: ast.Yield) -> None:
            self.generic_visit(node)

        def visit_YieldFrom(self, node: ast.YieldFrom) -> None:
            self.generic_visit(node)

        def visit_Global(self, node: ast.Global) -> None:
            # Allow globals but access to builtins is blocked elsewhere
            self.generic_visit(node)

        def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
            self.generic_visit(node)

    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError:
        return False

    visitor = SafetyVisitor()
    visitor.visit(tree)
    return visitor.safe


def _make_safe_builtins() -> Dict[str, Any]:
    """
    Construct a restricted set of builtins that are allowed in the sandbox.
    """
    builtins_obj = __builtins__
    if isinstance(builtins_obj, dict):
        source = builtins_obj
    else:
        source = vars(builtins_obj)

    safe: Dict[str, Any] = {name: source[name] for name in SAFE_BUILTIN_CALLS if name in source}
    return safe


def execute_safe_script(code: str) -> Any:
    """
    Execute Python code in a restricted environment after AST safety analysis.

    - Returns the evaluated result if code is a single expression.
    - Otherwise, executes the code and returns the value of a variable named 'result' if present, else None.
    - Raises ValueError if prohibited operations are detected by the analyzer.
    """
    if not analyze_ast_for_safety(code):
        raise ValueError("Script contains prohibited operations.")

    sandbox_globals: Dict[str, Any] = {
        "__builtins__": _make_safe_builtins(),
    }

    # Try to evaluate as an expression first
    try:
        compiled = compile(code, "<user-code>", "eval")
    except SyntaxError:
        # Fallback to exec for statements/blocks
        compiled = compile(code, "<user-code>", "exec")
        exec(compiled, sandbox_globals, sandbox_globals)
        return sandbox_globals.get("result", None)
    else:
        return eval(compiled, sandbox_globals, sandbox_globals)


def safe_run_script(code: str) -> Any:
    """
    Validate and execute a user-provided Python script safely.

    - Uses analyze_ast_for_safety to check for prohibited operations or invalid syntax.
    - Executes the code with execute_safe_script if safe.
    - Returns the result of the script or None if there is no result.
    - Raises ValueError for prohibited operations or invalid syntax.
    """
    if not analyze_ast_for_safety(code):
        raise ValueError("Script contains prohibited operations or invalid syntax.")

    try:
        return execute_safe_script(code)
    except SyntaxError as e:
        # Normalize any unexpected syntax errors into ValueError per API contract.
        raise ValueError("Invalid syntax.") from e
