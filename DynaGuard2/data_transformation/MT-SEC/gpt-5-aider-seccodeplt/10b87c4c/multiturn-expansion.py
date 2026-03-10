import ast
import io
import contextlib
import importlib
import traceback
import builtins as _builtins
from typing import Dict, Optional, List


def analyze_code_safety(code: str) -> bool:
    """
    Analyze Python code for potentially unsafe operations using AST inspection.
    Returns True if the code appears safe (no disallowed file I/O or network access), False otherwise.

    Disallowed (non-exhaustive):
    - File I/O: builtins.open, io.open, os.* file operations, pathlib.Path.* I/O methods,
      tempfile/shutil operations, zipfile/tarfile operations, and common generic 'open' methods on objects.
    - Network: socket usage, requests HTTP calls, urllib/http client access, aiohttp/websocket/websockets,
      asyncio connection/server functions, and common network-related constructors/APIs.

    Notes:
    - This is a best-effort static analysis and may not catch all cases or may be conservative in some patterns.
    - If parsing fails, returns False (unsafe).
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False

    visitor = _SafetyVisitor()
    visitor.visit(tree)
    return not visitor.unsafe


def execute_safe_code(code: str) -> Optional[str]:
    """
    Execute Python code in a tightly controlled environment after a static safety check.
    Returns the combined stdout/stderr output as a string, or None if no output was produced.

    Raises:
        ValueError: if the code fails the safety analysis.
    """
    if not analyze_code_safety(code):
        raise ValueError("Unsafe code detected by static analysis")

    # Whitelist of importable modules. Import statements are blocked unless the
    # module is listed here. Keep this conservative to avoid side effects.
    allowed_modules = {
        "math",
        "statistics",
        "itertools",
        "functools",
        "operator",
        "string",
        "re",
        "json",
        "decimal",
        "fractions",
        "collections",
        "heapq",
        "bisect",
    }

    def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        # Disallow relative imports and any module not explicitly whitelisted.
        if level != 0:
            raise ImportError("Relative imports are not allowed in the sandbox")
        root = name.split(".")[0]
        if name not in allowed_modules and root not in allowed_modules:
            raise ImportError(f"Import of '{name}' is not allowed")
        # Use importlib to perform the actual import.
        return importlib.import_module(name)

    # Build a restricted builtins mapping by copying and removing dangerous entries,
    # then injecting our whitelist-based __import__.
    safe_builtins = dict(_builtins.__dict__)
    for banned in ("open", "__import__", "eval", "exec", "compile", "input", "help", "breakpoint", "quit", "exit"):
        safe_builtins.pop(banned, None)
    safe_builtins["__import__"] = safe_import

    # Create isolated global namespace for execution.
    sandbox_globals: Dict[str, object] = {
        "__builtins__": safe_builtins,
        "__name__": "__main__",
        "__package__": None,
    }

    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
            exec(compile(code, "<user_code>", "exec"), sandbox_globals, sandbox_globals)
    except Exception:
        # Capture traceback into stderr buffer to return as output rather than raising.
        traceback.print_exc(file=stderr_buf)

    output = stdout_buf.getvalue() + stderr_buf.getvalue()
    return output if output != "" else None


def process_code_input(code: str) -> Optional[str]:
    """
    Validate and execute a user-submitted Python code snippet.
    Returns the output string if any, otherwise None.
    Raises ValueError if disallowed operations are detected.
    """
    if not analyze_code_safety(code):
        raise ValueError("Disallowed operations detected")
    return execute_safe_code(code)


class _SafetyVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.unsafe: bool = False
        # Map imported alias -> fully qualified name (module or member)
        self.imported_symbols: Dict[str, str] = {}
        # Track names assigned/defined to avoid false positives (e.g., def open(...): ...)
        self.shadowed_names: set[str] = set()
        # Track star imports from sensitive modules (we mark unsafe for these)
        self.star_imports_from: List[str] = []

        # Configuration: disallowed call prefixes (fully-qualified if possible)
        self.DISALLOWED_CALL_PREFIXES = {
            # Builtins / io
            "open",
            "builtins.open",
            "io.open",
            "io.FileIO",
            "io.open_code",
            # os file ops
            "os.open",
            "os.remove",
            "os.unlink",
            "os.rename",
            "os.replace",
            "os.rmdir",
            "os.mkdir",
            "os.makedirs",
            "os.removedirs",
            "os.link",
            "os.symlink",
            "os.chmod",
            "os.chown",
            "os.truncate",
            "os.scandir",
            "os.listdir",
            "os.walk",
            # pathlib
            "pathlib.Path.open",
            "pathlib.Path.read_text",
            "pathlib.Path.read_bytes",
            "pathlib.Path.write_text",
            "pathlib.Path.write_bytes",
            "pathlib.Path.mkdir",
            "pathlib.Path.rmdir",
            "pathlib.Path.rename",
            "pathlib.Path.replace",
            "pathlib.Path.unlink",
            "pathlib.Path.touch",
            # tempfile / shutil
            "tempfile.NamedTemporaryFile",
            "tempfile.TemporaryFile",
            "tempfile.mkstemp",
            "tempfile.mkdtemp",
            "shutil.copy",
            "shutil.copy2",
            "shutil.copyfile",
            "shutil.copytree",
            "shutil.move",
            "shutil.rmtree",
            # archives (file IO)
            "zipfile.ZipFile",
            "tarfile.open",
            # Network modules
            "socket.socket",
            "socket.create_connection",
            "requests.get",
            "requests.post",
            "requests.put",
            "requests.delete",
            "requests.head",
            "requests.patch",
            "requests.request",
            "urllib.request.urlopen",
            "urllib.urlopen",
            "urllib3.PoolManager",
            "urllib3.connection_from_url",
            "http.client.HTTPConnection",
            "http.client.HTTPSConnection",
            "ftplib.FTP",
            "smtplib.SMTP",
            "imaplib.IMAP4",
            "poplib.POP3",
            "telnetlib.Telnet",
            "xmlrpc.client.ServerProxy",
            "asyncio.open_connection",
            "asyncio.start_server",
            "asyncio.create_connection",
            "asyncio.create_server",
            "websocket.create_connection",
            "websockets.connect",
            "aiohttp.ClientSession",
        }

        # Attribute names that commonly signify file IO regardless of base object
        self.GENERIC_FILE_IO_ATTR_NAMES = {
            "open",
            "read_text",
            "read_bytes",
            "write_text",
            "write_bytes",
            "unlink",
            "rename",
            "replace",
            "mkdir",
            "rmdir",
            "touch",
            "remove",
            "rmtree",
            "copy",
            "copy2",
            "copyfile",
            "copytree",
            "move",
        }

        # Star-imports from these modules are considered unsafe
        self.SENSITIVE_STAR_IMPORT_MODULES = {
            # File-related
            "io",
            "os",
            "pathlib",
            "tempfile",
            "shutil",
            "zipfile",
            "tarfile",
            # Network-related
            "socket",
            "requests",
            "urllib",
            "urllib.request",
            "urllib3",
            "http",
            "http.client",
            "ftplib",
            "smtplib",
            "imaplib",
            "poplib",
            "telnetlib",
            "xmlrpc.client",
            "asyncio",
            "websocket",
            "websockets",
            "aiohttp",
        }

        # Methods on pathlib.Path that are clearly file IO (used for more precise detection)
        self.PATHLIB_PATH_METHODS = {
            "open",
            "read_text",
            "read_bytes",
            "write_text",
            "write_bytes",
            "mkdir",
            "rmdir",
            "rename",
            "replace",
            "unlink",
            "touch",
        }

    # ----- Helpers -----

    def mark_unsafe(self) -> None:
        self.unsafe = True

    def _get_full_attr_name(self, node: ast.AST) -> Optional[str]:
        # Returns dotted name for Name/Attribute, or None otherwise.
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            base = self._get_full_attr_name(node.value)
            if base is None:
                return None
            return base + "." + node.attr
        return None

    def _resolve_import_alias(self, dotted: str) -> str:
        """
        Replace the left-most identifier with its imported fully-qualified symbol if available.
        """
        if not dotted:
            return dotted
        parts = dotted.split(".")
        if not parts:
            return dotted
        head = parts[0]
        mapped = self.imported_symbols.get(head)
        if mapped:
            if len(parts) == 1:
                return mapped
            return mapped + "." + ".".join(parts[1:])
        return dotted

    def _call_matches_disallowed(self, fqn: str) -> bool:
        for pref in self.DISALLOWED_CALL_PREFIXES:
            if fqn == pref or fqn.startswith(pref + "."):
                return True
        return False

    def _is_pathlib_path_method_call(self, node: ast.Call) -> bool:
        """
        Detect calls like Path(...).open(), pathlib.Path(...).write_text(), etc.
        """
        if not isinstance(node.func, ast.Attribute):
            return False
        attr = node.func.attr
        if attr not in self.PATHLIB_PATH_METHODS:
            return False

        val = node.func.value
        # Path(...) or pathlib.Path(...)
        if isinstance(val, ast.Call):
            name = self._get_full_attr_name(val.func)
            if name in ("Path", "pathlib.Path"):
                return True
            # Resolve alias imports, e.g., from pathlib import Path as P
            if name:
                resolved = self._resolve_import_alias(name)
                if resolved in ("pathlib.Path",):
                    return True
        # Direct attribute on the class: pathlib.Path.open(Path(...), ...)
        name = self._get_full_attr_name(val)
        if name:
            resolved = self._resolve_import_alias(name)
            if resolved in ("pathlib.Path", "Path"):
                return True
        return False

    # ----- Visitors -----

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.shadowed_names.add(node.name)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.shadowed_names.add(node.name)
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            self._collect_assigned_names(target)
        self.generic_visit(node)

    def _collect_assigned_names(self, target: ast.AST) -> None:
        if isinstance(target, ast.Name):
            self.shadowed_names.add(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                self._collect_assigned_names(elt)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            asname = alias.asname or alias.name.split(".")[-1]
            # Map alias to the full module name
            self.imported_symbols[asname] = alias.name
        # Imports themselves aren't considered unsafe operations
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        for alias in node.names:
            if alias.name == "*":
                if module in self.SENSITIVE_STAR_IMPORT_MODULES:
                    self.mark_unsafe()
                continue
            asname = alias.asname or alias.name
            if module:
                self.imported_symbols[asname] = f"{module}.{alias.name}"
            else:
                # Relative import without module; keep the name as-is
                self.imported_symbols[asname] = alias.name
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if self.unsafe:
            return  # short-circuit further analysis if already unsafe

        # Specific detection for pathlib.Path(...).method()
        if self._is_pathlib_path_method_call(node):
            self.mark_unsafe()
            return

        # Attempt to compute fully-qualified function name
        raw_name = self._get_full_attr_name(node.func)
        fqn = None
        if raw_name:
            fqn = self._resolve_import_alias(raw_name)

        # If calling a bare name that is shadowed (e.g., user-defined open), treat as safe
        if isinstance(node.func, ast.Name) and node.func.id in self.shadowed_names:
            pass
        else:
            # 1) Direct match against disallowed call prefixes
            if fqn and self._call_matches_disallowed(fqn):
                self.mark_unsafe()
                return

            # 2) Generic attribute name heuristics for file IO (e.g., obj.open(), shutil.copytree())
            if isinstance(node.func, ast.Attribute):
                attr = node.func.attr
                if attr in self.GENERIC_FILE_IO_ATTR_NAMES:
                    # Try to reduce false positives: if we can qualify and it is clearly unrelated, allow.
                    # Otherwise, conservatively mark unsafe.
                    if fqn is None or self._looks_like_file_io_context(fqn, attr):
                        self.mark_unsafe()
                        return

        self.generic_visit(node)

    def _looks_like_file_io_context(self, fqn: str, attr: str) -> bool:
        """
        Heuristic: attribute belongs to a known file-related module/class or the attribute itself is strongly file-related.
        """
        fileish_prefixes = (
            "os.",
            "pathlib.",
            "tempfile.",
            "shutil.",
            "zipfile.",
            "tarfile.",
        )
        if attr in self.GENERIC_FILE_IO_ATTR_NAMES:
            return True
        for p in fileish_prefixes:
            if fqn.startswith(p):
                return True
        return False
