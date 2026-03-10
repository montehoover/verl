import ast
from typing import Set


def analyze_code_safety(code: str) -> bool:
    """
    Analyze a Python code snippet for potentially harmful operations using AST.
    Returns True if the code appears safe to execute; otherwise False.

    The analysis is conservative and blocks:
    - Imports of modules commonly used for system/file/network access or introspection.
    - Calls to dangerous builtins (open, exec, eval, compile, __import__, getattr/setattr/delattr, vars, dir, breakpoint).
    - Attribute access to dangerous operations (system, popen, remove, unlink, rmdir, chmod, chown, mkdir, makedirs, rename, replace, symlink, link, exec*, spawn, fork, kill, environ, getenv, putenv, startfile, open).
    - Any dunder attribute access (attributes starting with "__").
    - Relative imports.
    If parsing fails, the code is considered unsafe.
    """

    forbidden_modules: Set[str] = {
        "os",
        "sys",
        "subprocess",
        "shutil",
        "pathlib",
        "inspect",
        "ctypes",
        "importlib",
        "pkgutil",
        "site",
        "sysconfig",
        "socket",
        "ssl",
        "http",
        "urllib",
        "requests",
        "ftplib",
        "smtplib",
        "imaplib",
        "poplib",
        "telnetlib",
        "xmlrpc",
        "xmlrpc.client",
        "xmlrpc.server",
        "winreg",
        "plistlib",
        "pickle",
        "marshal",
        "dill",
        "cloudpickle",
        "resource",
        "multiprocessing",
        "threading",
        "signal",
        "atexit",
        "tempfile",
        "glob",
        "fnmatch",
        "getpass",
        "keyring",
        "builtins",
    }

    forbidden_call_names: Set[str] = {
        "open",
        "exec",
        "eval",
        "compile",
        "__import__",
        "getattr",
        "setattr",
        "delattr",
        "vars",
        "dir",
        "breakpoint",
        # Access to runtime namespaces can be sensitive
        "globals",
        "locals",
    }

    forbidden_attribute_names: Set[str] = {
        # process / system
        "system",
        "popen",
        "spawn",
        "fork",
        "execv",
        "execve",
        "execvp",
        "execl",
        "execlp",
        "kill",
        "killpg",
        "startfile",
        # filesystem mutation
        "remove",
        "unlink",
        "rmdir",
        "removedirs",
        "rename",
        "replace",
        "chmod",
        "chown",
        "lchown",
        "makedirs",
        "mkdir",
        "symlink",
        "link",
        "copy",
        "copy2",
        "copytree",
        "move",
        "rmtree",
        # environment / sensitive
        "environ",
        "getenv",
        "putenv",
        # IO / code exec
        "open",
        "eval",
        "exec",
        "walk",
        "chroot",
        "mknod",
        "mkfifo",
        "umask",
        "load",   # e.g., ctypes.CDLL(...).load or similar dangerous loaders
        # 'loads' and 'dumps' are intentionally not blocked because of json usage
    }

    class SafetyAnalyzer(ast.NodeVisitor):
        def __init__(self):
            self.safe = True

        def mark_unsafe(self):
            self.safe = False

        def visit(self, node):
            if not self.safe:
                return None
            return super().visit(node)

        def visit_Import(self, node: ast.Import):
            for alias in node.names:
                mod = alias.name
                root = mod.split(".")[0]
                if mod in forbidden_modules or root in forbidden_modules:
                    self.mark_unsafe()
                    return
            self.generic_visit(node)

        def visit_ImportFrom(self, node: ast.ImportFrom):
            # Block any relative import
            if node.level and node.level > 0:
                self.mark_unsafe()
                return
            mod = node.module or ""
            root = mod.split(".")[0] if mod else ""
            if mod in forbidden_modules or root in forbidden_modules:
                self.mark_unsafe()
                return
            self.generic_visit(node)

        def visit_Attribute(self, node: ast.Attribute):
            # Block any dunder attribute access as a strong sandboxing measure
            if node.attr.startswith("__"):
                self.mark_unsafe()
                return
            if node.attr in forbidden_attribute_names:
                self.mark_unsafe()
                return
            self.generic_visit(node)

        def visit_Call(self, node: ast.Call):
            # Check direct calls by name
            func = node.func
            if isinstance(func, ast.Name):
                if func.id in forbidden_call_names:
                    self.mark_unsafe()
                    return
            elif isinstance(func, ast.Attribute):
                # If calling an attribute like obj.system(), obj.open(), etc.
                if func.attr in forbidden_attribute_names:
                    self.mark_unsafe()
                    return
                # If directly calling module functions through attribute access where module is forbidden
                # Example: subprocess.run(...) would be caught at import time, but be defensive
                root_name = None
                value = func.value
                if isinstance(value, ast.Name):
                    root_name = value.id
                elif isinstance(value, ast.Attribute):
                    # Dive one level to get possible root module name
                    v = value
                    while isinstance(v, ast.Attribute):
                        v = v.value
                    if isinstance(v, ast.Name):
                        root_name = v.id
                if root_name in forbidden_modules:
                    self.mark_unsafe()
                    return
            self.generic_visit(node)

        def visit_Name(self, node: ast.Name):
            # Access to __builtins__ can expose powerful capabilities
            if node.id == "__builtins__":
                self.mark_unsafe()
                return
            self.generic_visit(node)

        def visit_Global(self, node: ast.Global):
            # Allow globals, but be conservative: too much mutation may be unsafe in shared envs
            self.generic_visit(node)

        def visit_Nonlocal(self, node: ast.Nonlocal):
            self.generic_visit(node)

    try:
        tree = ast.parse(code, mode="exec")
    except Exception:
        return False

    analyzer = SafetyAnalyzer()
    analyzer.visit(tree)
    return bool(analyzer.safe)
