import ast
from typing import Dict, Optional, List


# Modules whose function/method calls are considered unsafe (prefix match)
BANNED_CALL_PREFIXES = {
    "os",
    "subprocess",
    "socket",
    "requests",
    "urllib.request",
    "http.client",
    "ftplib",
    "smtplib",
    "imaplib",
    "poplib",
    "ssl",
    "shutil",
}

# Built-in function names that are unsafe to call directly
BANNED_BUILTIN_CALLS = {
    "open",
    "eval",
    "exec",
    "__import__",
}

# Attribute method names that usually indicate file I/O when called
BANNED_ATTR_CALL_NAMES = {
    "open",
    "read_text",
    "read_bytes",
    "write_text",
    "write_bytes",
    "unlink",
    "mkdir",
    "rmdir",
    "rename",
    "replace",
    "rmtree",
    "remove",
    "popen",
    "system",
}


def _dotted_name_from_attr(node: ast.AST) -> Optional[str]:
    """
    Build a dotted name string from an Attribute/Name chain, if possible.
    Returns None if it encounters an expression that is not a simple Name/Attribute chain.
    """
    parts: List[str] = []
    n = node
    while isinstance(n, ast.Attribute):
        parts.append(n.attr)
        n = n.value
    if isinstance(n, ast.Name):
        parts.append(n.id)
    else:
        return None
    parts.reverse()
    return ".".join(parts)


def _replace_alias_root(dotted: str, aliases: Dict[str, str]) -> str:
    """
    Replace the root part of a dotted name if it is an alias to a module path.
    """
    if not dotted:
        return dotted
    root, *rest = dotted.split(".")
    if root in aliases:
        mapped = aliases[root]
        if rest:
            return ".".join([mapped] + rest)
        return mapped
    return dotted


class _SafetyVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.unsafe = False
        # Map alias -> module or attr path (e.g., "os", "urllib.request", "os.system")
        self.aliases: Dict[str, str] = {}

    def mark_unsafe(self) -> None:
        self.unsafe = True

    def visit_Import(self, node: ast.Import) -> None:
        # Record aliases for later resolution in calls
        for alias in node.names:
            name = alias.name  # fully qualified module path
            asname = alias.asname or name.split(".")[0]
            # If no "as", bind the top-level name to its top-level module
            if alias.asname:
                # alias maps to full dotted
                self.aliases[asname] = name
            else:
                # "import os.path" binds "os" name; for simplicity bind top-level to itself
                top = name.split(".")[0]
                self.aliases[top] = top
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        mod = node.module or ""
        for alias in node.names:
            bind = alias.asname or alias.name
            # Map alias to full path (module + '.' + name) if module is present
            full = f"{mod}.{alias.name}" if mod else alias.name
            self.aliases[bind] = full
        self.generic_visit(node)

    def _is_banned_prefix(self, dotted: str) -> bool:
        for p in BANNED_CALL_PREFIXES:
            if dotted == p or dotted.startswith(p + "."):
                return True
        return False

    def visit_Call(self, node: ast.Call) -> None:
        if self.unsafe:
            return

        func = node.func

        # Direct name call, e.g., open(), eval(), myalias()
        if isinstance(func, ast.Name):
            name = func.id
            if name in BANNED_BUILTIN_CALLS:
                self.mark_unsafe()
                return
            # If calling an alias, resolve what it points to
            if name in self.aliases:
                target = self.aliases[name]
                # If alias points to a banned module/function path
                if self._is_banned_prefix(target):
                    self.mark_unsafe()
                    return
                # If alias points to something ending with a banned method/function name
                last = target.rsplit(".", 1)[-1]
                if last in BANNED_ATTR_CALL_NAMES or last in BANNED_BUILTIN_CALLS:
                    self.mark_unsafe()
                    return

        # Attribute call, e.g., os.remove(), requests.get(), pathlib.Path(...).open()
        elif isinstance(func, ast.Attribute):
            dotted = _dotted_name_from_attr(func)
            last_attr = func.attr

            # If attribute couldn't be turned into a dotted name (e.g., (<expr>).open)
            # Still block by attribute name for known dangerous I/O methods.
            if dotted is None:
                if last_attr in BANNED_ATTR_CALL_NAMES:
                    self.mark_unsafe()
                    return
            else:
                # Resolve alias at root if present
                resolved = _replace_alias_root(dotted, self.aliases)
                # If the final attr is a banned I/O method, mark unsafe
                if last_attr in BANNED_ATTR_CALL_NAMES:
                    self.mark_unsafe()
                    return
                # If resolved dotted name falls under banned prefixes (network/OS/subprocess/etc.)
                if self._is_banned_prefix(resolved):
                    self.mark_unsafe()
                    return

        self.generic_visit(node)

    def visit_With(self, node: ast.With) -> None:
        # Explicitly inspect context managers like "with open(...)" which are Calls
        for item in node.items:
            ctx = item.context_expr
            if isinstance(ctx, ast.Call):
                self.visit_Call(ctx)
                if self.unsafe:
                    return
        self.generic_visit(node)

    def visit_Await(self, node: ast.Await) -> None:
        # Also analyze awaited calls for completeness
        self.generic_visit(node)


def analyze_snippet_safety(code: str) -> bool:
    """
    Analyze a Python code snippet for potentially harmful operations (e.g., file I/O or network access).
    Returns True if the snippet appears safe, False otherwise.
    """
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError:
        # Treat invalid code as unsafe to execute
        return False

    visitor = _SafetyVisitor()
    visitor.visit(tree)
    return not visitor.unsafe
