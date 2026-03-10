import ast
from typing import Set


BANNED_CALLS: Set[str] = {
    "eval",
    "exec",
    "compile",
    "__import__",
    "getattr",
    "setattr",
    "delattr",
    "globals",
    "locals",
    "vars",
    "dir",
    "open",
    "input",
    "exit",
    "quit",
    "breakpoint",
    "help",
}

BANNED_NAMES: Set[str] = {
    "__builtins__",
    "builtins",
    "os",
    "sys",
    "subprocess",
    "shutil",
    "signal",
    "importlib",
    "ctypes",
    "multiprocessing",
    "threading",
    "socket",
    "pickle",
    "marshal",
    "resource",
    "psutil",
    "pathlib",
}

BANNED_ATTRS: Set[str] = {
    "__dict__",
    "__class__",
    "__bases__",
    "__mro__",
    "__subclasses__",
    "__globals__",
    "__code__",
    "__closure__",
    "__getattribute__",
    "__getattr__",
    "__setattr__",
    "__delattr__",
    "__call__",
    "__import__",
    "f_back",
    "f_locals",
    "f_globals",
}

BANNED_ATTRIBUTE_CALLS: Set[str] = {
    "os.system",
    "os.popen",
    "os.execv",
    "os.execve",
    "os.spawn",
    "os.fork",
    "os.forkpty",
    "sys.exit",
    "subprocess.call",
    "subprocess.run",
    "subprocess.Popen",
    "shutil.rmtree",
    "shutil.move",
    "shutil.chown",
    "shutil.chmod",
}

BANNED_EXCEPTIONS: Set[str] = {"SystemExit", "KeyboardInterrupt"}


def get_full_attr_path(attr: ast.AST) -> str:
    parts = []
    cur = attr
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
    elif isinstance(cur, ast.Call):
        parts.append("<call>")
    else:
        parts.append(type(cur).__name__)
    return ".".join(reversed(parts))


class SafetyVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.safe = True

    def mark_unsafe(self) -> None:
        self.safe = False

    def visit_Import(self, node: ast.Import) -> None:
        self.mark_unsafe()

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self.mark_unsafe()

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func

        if isinstance(func, ast.Name):
            if func.id in BANNED_CALLS or func.id in BANNED_NAMES:
                self.mark_unsafe()
        elif isinstance(func, ast.Attribute):
            full = get_full_attr_path(func)
            base = full.split(".")[0]
            if base in BANNED_NAMES:
                self.mark_unsafe()
            if any(part in BANNED_ATTRS for part in full.split(".")):
                self.mark_unsafe()
            if full in BANNED_ATTRIBUTE_CALLS:
                self.mark_unsafe()

        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr in BANNED_ATTRS:
            self.mark_unsafe()

        base_name = None
        value = node.value
        if isinstance(value, ast.Name):
            base_name = value.id
        elif isinstance(value, ast.Attribute):
            base_full = get_full_attr_path(value)
            base_name = base_full.split(".")[0]

        if base_name and base_name in BANNED_NAMES:
            self.mark_unsafe()

        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in BANNED_NAMES:
            self.mark_unsafe()

    def visit_Raise(self, node: ast.Raise) -> None:
        exc_name = None
        if node.exc is not None:
            if isinstance(node.exc, ast.Name):
                exc_name = node.exc.id
            elif isinstance(node.exc, ast.Call):
                fn = node.exc.func
                if isinstance(fn, ast.Name):
                    exc_name = fn.id
                elif isinstance(fn, ast.Attribute):
                    exc_name = fn.attr
        if exc_name and exc_name in BANNED_EXCEPTIONS:
            self.mark_unsafe()
        self.generic_visit(node)


def analyze_ast_for_safety(code: str) -> bool:
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError:
        return False

    visitor = SafetyVisitor()
    visitor.visit(tree)
    return bool(visitor.safe)
