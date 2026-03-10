import ast
import builtins
import io
from contextlib import redirect_stdout
from typing import Optional, Set

FORBIDDEN_MODULES: Set[str] = {
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

FORBIDDEN_CALL_NAMES: Set[str] = {
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

FORBIDDEN_ATTRIBUTE_NAMES: Set[str] = {
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
    "load",  # e.g., ctypes.CDLL(...).load or similar dangerous loaders
    # 'loads' and 'dumps' are intentionally not blocked because of json usage
}


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
                if mod in FORBIDDEN_MODULES or root in FORBIDDEN_MODULES:
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
            if mod in FORBIDDEN_MODULES or root in FORBIDDEN_MODULES:
                self.mark_unsafe()
                return
            self.generic_visit(node)

        def visit_Attribute(self, node: ast.Attribute):
            # Block any dunder attribute access as a strong sandboxing measure
            if node.attr.startswith("__"):
                self.mark_unsafe()
                return
            if node.attr in FORBIDDEN_ATTRIBUTE_NAMES:
                self.mark_unsafe()
                return
            self.generic_visit(node)

        def visit_Call(self, node: ast.Call):
            # Check direct calls by name
            func = node.func
            if isinstance(func, ast.Name):
                if func.id in FORBIDDEN_CALL_NAMES:
                    self.mark_unsafe()
                    return
            elif isinstance(func, ast.Attribute):
                # If calling an attribute like obj.system(), obj.open(), etc.
                if func.attr in FORBIDDEN_ATTRIBUTE_NAMES:
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
                if root_name in FORBIDDEN_MODULES:
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


def execute_safe_code(code: str) -> Optional[str]:
    """
    Execute a Python code snippet that has been deemed safe.
    Returns the combined stdout output as a string, or None if no output was produced
    or if a runtime error occurs. The function also rejects code that fails the
    analyze_code_safety check.
    """
    if not analyze_code_safety(code):
        return None

    # Restricted import that honors the same forbidden module list
    def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level and level > 0:
            raise ImportError("Relative imports are not allowed")
        mod = name or ""
        root = mod.split(".")[0] if mod else ""
        if mod in FORBIDDEN_MODULES or root in FORBIDDEN_MODULES:
            raise ImportError(f"Import of module '{name}' is not allowed")
        return builtins.__import__(name, globals, locals, fromlist, level)

    safe_builtins = {
        # common safe builtins
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
        "object": object,
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
        "super": super,
        # Required for class definitions
        "__build_class__": builtins.__build_class__,
        # Allow safe imports (AST already blocks forbidden ones)
        "__import__": _safe_import,
        # Constants
        "True": True,
        "False": False,
        "None": None,
        "NotImplemented": NotImplemented,
        "Ellipsis": Ellipsis,
    }

    # Provide an isolated global namespace
    exec_globals = {
        "__builtins__": safe_builtins,
        "__name__": "__main__",
        "__doc__": None,
        "__package__": None,
        "__spec__": None,
    }

    buf = io.StringIO()
    try:
        code_obj = compile(code, "<user_code>", "exec")
        with redirect_stdout(buf):
            exec(code_obj, exec_globals, exec_globals)
    except Exception:
        return None

    out = buf.getvalue()
    return out if out != "" else None


def execute_python_code(code: str) -> Optional[str]:
    """
    Validate code safety and execute it if safe.
    - Returns the stdout output as a string if any was produced.
    - Returns None if no output was produced or a runtime error occurred.
    - Raises ValueError if the code is unsafe or invalid.
    """
    if not isinstance(code, str):
        raise ValueError("Code must be a string")

    if not analyze_code_safety(code):
        raise ValueError("Provided code is unsafe or invalid")

    return execute_safe_code(code)
