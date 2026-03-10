import ast
import builtins
import contextlib
import io
from typing import Any, List, Optional


def parse_python_code(source: str) -> ast.AST:
    """
    Parse a string of Python code and return its AST if syntactically correct.

    Args:
        source: A string containing Python source code.

    Returns:
        An ast.AST object representing the parsed code.

    Raises:
        ValueError: If the code has syntax issues.
        TypeError: If 'source' is not a string.
    """
    if not isinstance(source, str):
        raise TypeError("source must be a string of Python code")

    try:
        return ast.parse(source, mode="exec")
    except SyntaxError as e:
        msg = e.msg or "invalid syntax"
        parts = []
        if e.lineno is not None:
            parts.append(f"line {e.lineno}")
        if e.offset is not None:
            parts.append(f"column {e.offset}")
        location = f" ({', '.join(parts)})" if parts else ""
        snippet = e.text.strip() if e.text else None
        if snippet:
            raise ValueError(f"{msg}{location}: {snippet}") from e
        raise ValueError(f"{msg}{location}") from e


# ---------------- Security checks ----------------

_PROHIBITED_IMPORT_PREFIXES = {
    # Network-related modules
    "socket",
    "ssl",
    "http",
    "urllib",
    "urllib3",
    "requests",
    "ftplib",
    "telnetlib",
    "smtplib",
    "poplib",
    "imaplib",
    "nntplib",
    "xmlrpc",
    "websocket",
    "websockets",
    "paramiko",
    "asyncio",
    # Process and shell
    "subprocess",
    # File-system related (for IO/manipulation)
    "os",
    "pathlib",
    "io",
    "shutil",
}

_PROHIBITED_CALLS = {
    # Directly prohibited builtins or helpers
    "open",         # file I/O
    "__import__",   # dynamic import
    "exec",         # code execution
    "eval",         # code execution
}

# Any call whose dotted name starts with one of these is prohibited
_PROHIBITED_CALL_PREFIXES = {
    # FS operations
    "os.",
    "pathlib.Path.open",
    "pathlib.Path.write_text",
    "pathlib.Path.write_bytes",
    "pathlib.Path.read_text",
    "pathlib.Path.read_bytes",
    "pathlib.Path.unlink",
    "pathlib.Path.rename",
    "pathlib.Path.replace",
    "shutil.",
    "io.open",
    # Network operations
    "socket",
    "requests",
    "urllib",
    "http.client",
    "ftplib",
    "telnetlib",
    "smtplib",
    "poplib",
    "imaplib",
    "websocket",
    "websockets",
    "paramiko",
    "asyncio.open_connection",
    "asyncio.start_server",
    # Process creation / shell
    "subprocess",
    "os.system",
    "os.popen",
}

# Attribute method names that imply file I/O or network ops
_PROHIBITED_FILE_METHODS = {
    "open",
    "unlink",
    "rename",
    "replace",
    "remove",
    "rmdir",
    "mkdir",
    "makedirs",
    "chmod",
    "chown",
    "write_text",
    "write_bytes",
    "read_text",
    "read_bytes",
}
_PROHIBITED_NETWORK_METHODS = {
    "connect",
    "bind",
    "listen",
    "accept",
    "send",
    "recv",
    "sendall",
    "create_connection",
    "open_connection",
    "get",
    "post",
    "put",
    "delete",
    "head",
    "patch",
    "request",
    "urlopen",
}


def _dotted_name(node: ast.AST) -> Optional[str]:
    """
    Attempt to reconstruct a dotted name from an AST node representing
    a function or attribute reference.
    """
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted_name(node.value)
        if base:
            return f"{base}.{node.attr}"
        return node.attr
    if isinstance(node, ast.Call):
        # For something like Path(...).open -> report "pathlib.Path.open"
        return _dotted_name(node.func)
    return None


class _ProhibitedActionChecker(ast.NodeVisitor):
    def __init__(self) -> None:
        self.issues: List[str] = []

    def _report(self, node: ast.AST, reason: str) -> None:
        lineno = getattr(node, "lineno", None)
        col = getattr(node, "col_offset", None)
        loc = f" at line {lineno}, column {col}" if lineno is not None and col is not None else ""
        self.issues.append(f"{reason}{loc}")

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            name = alias.name  # may be dotted
            for prefix in _PROHIBITED_IMPORT_PREFIXES:
                if name == prefix or name.startswith(prefix + "."):
                    self._report(node, f"Import of prohibited module '{name}'")
                    break
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        for prefix in _PROHIBITED_IMPORT_PREFIXES:
            if module == prefix or module.startswith(prefix + "."):
                self._report(node, f"Import from prohibited module '{module}'")
                break
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        name = _dotted_name(node.func)
        # Directly prohibited calls
        if name in _PROHIBITED_CALLS:
            self._report(node, f"Use of prohibited function '{name}'")
        # File I/O via any .open or builtin open
        if name == "open" or (isinstance(node.func, ast.Attribute) and node.func.attr == "open"):
            self._report(node, f"File I/O via '{name or 'open'}'")
        # Prefix-based checks
        if name:
            for prefix in _PROHIBITED_CALL_PREFIXES:
                if name == prefix.rstrip(".") or name.startswith(prefix):
                    self._report(node, f"Call to prohibited API '{name}'")
                    break
        # Method-name based checks
        if isinstance(node.func, ast.Attribute):
            meth = node.func.attr
            if meth in _PROHIBITED_FILE_METHODS:
                pretty = name or meth
                self._report(node, f"File system operation via '{pretty}'")
            if meth in _PROHIBITED_NETWORK_METHODS:
                pretty = name or meth
                self._report(node, f"Network operation via '{pretty}'")
        self.generic_visit(node)


def check_for_prohibited_actions(tree: ast.AST) -> bool:
    """
    Check an AST for prohibited actions (e.g., file I/O, network access).

    Args:
        tree: An ast.AST object previously produced by parse_python_code.

    Returns:
        True if the code is considered safe.

    Raises:
        TypeError: If 'tree' is not an AST.
        ValueError: If prohibited actions are detected.
    """
    if not isinstance(tree, ast.AST):
        raise TypeError("tree must be an ast.AST")
    checker = _ProhibitedActionChecker()
    checker.visit(tree)
    if checker.issues:
        raise ValueError("Prohibited actions detected: " + "; ".join(checker.issues))
    return True


# ---------------- Restricted execution ----------------

_SAFE_IMPORTS = {
    "math",
    "random",
    "statistics",
    "itertools",
    "functools",
    "operator",
    "string",
    "re",
    "collections",
}


def restricted_run_code(source: str) -> Optional[str]:
    """
    Parse, validate, and execute user code in a restricted environment.

    Args:
        source: Python source code to execute.

    Returns:
        The captured standard output as a string, or None if no output.

    Raises:
        ValueError: For syntax issues or prohibited actions.
        TypeError: If 'source' is not a string.
        Other exceptions from user code may propagate as-is.
    """
    tree = parse_python_code(source)
    check_for_prohibited_actions(tree)

    # Prepare a restricted set of builtins and a controlled importer
    def _restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
        root = name.split(".", 1)[0]
        if root not in _SAFE_IMPORTS:
            raise ImportError(f"Import of '{name}' is not allowed")
        return builtins.__import__(name, globals, locals, fromlist, level)

    safe_builtins = {
        "abs": abs,
        "all": all,
        "any": any,
        "bool": bool,
        "bytes": bytes,
        "chr": chr,
        "dict": dict,
        "enumerate": enumerate,
        "filter": filter,
        "float": float,
        "format": format,
        "hash": hash,
        "hex": hex,
        "int": int,
        "isinstance": isinstance,
        "issubclass": issubclass,
        "len": len,
        "list": list,
        "max": max,
        "min": min,
        "map": map,
        "next": next,
        "object": object,
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
        "__import__": _restricted_import,
        # Common exceptions
        "Exception": Exception,
        "ValueError": ValueError,
        "TypeError": TypeError,
    }

    globals_dict: dict = {"__builtins__": safe_builtins}
    locals_dict: dict = {}

    code_obj = compile(tree, filename="<user-code>", mode="exec")

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(code_obj, globals_dict, locals_dict)

    output = buf.getvalue()
    return output if output else None
