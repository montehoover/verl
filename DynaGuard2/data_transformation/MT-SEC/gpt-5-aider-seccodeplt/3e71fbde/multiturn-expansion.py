import ast
import io
import contextlib
import builtins
from typing import Set, Dict, Optional


class _CodeSafetyAnalyzer(ast.NodeVisitor):
    def __init__(self) -> None:
        # Assume safe until a disallowed operation is found
        self.safe: bool = True

        # Modules that imply network access
        self._banned_network_modules: Set[str] = {
            "socket",
            "requests",
            "urllib",
            "urllib3",
            "http",
            "ftplib",
            "ssl",
            "paramiko",
            "telnetlib",
            "smtplib",
            "poplib",
            "imaplib",
            "websocket",
            "websockets",
        }

        # Modules that imply (or facilitate) file I/O
        self._banned_file_modules: Set[str] = {
            "pathlib",
            "shutil",
            "tempfile",
        }

        # For os.* we allow importing os but disallow specific function calls
        self._os_aliases: Set[str] = set()  # e.g., {"os", "o"} if "import os as o"
        self._module_aliases: Dict[str, str] = {}  # alias -> top-level module name

        # Calls to these names are considered file I/O (e.g., built-in open)
        self._banned_name_calls: Set[str] = {"open"}

        # Disallowed os.* functions (file system operations)
        self._banned_os_funcs: Set[str] = {
            "open",
            "close",
            "fdopen",
            "read",
            "write",
            "pwrite",
            "pread",
            "remove",
            "unlink",
            "rename",
            "replace",
            "mkdir",
            "makedirs",
            "rmdir",
            "removedirs",
            "chdir",
            "chmod",
            "chown",
            "lchown",
            "link",
            "symlink",
            "mkfifo",
            "mknod",
            "truncate",
            "scandir",
            "listdir",
            "walk",
            "stat",
            "lstat",
            "fstat",
            "ftruncate",
        }

        # Any direct reference to these modules (by alias) in a call is banned
        self._banned_direct_modules: Set[str] = (
            self._banned_network_modules | self._banned_file_modules
        )

    def _top_level_module(self, name: Optional[str]) -> str:
        if not name:
            return ""
        return name.split(".")[0]

    def _mark_unsafe(self) -> None:
        self.safe = False

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            full_name = alias.name
            top = self._top_level_module(full_name)

            # Track aliases for os to check os.* calls later
            if top == "os":
                alias_name = alias.asname or top
                self._os_aliases.add(alias_name)
                self._module_aliases[alias_name] = "os"
                # Importing os alone is allowed; specific calls are checked later.
                continue

            # Track alias mappings (e.g., import requests as r)
            alias_name = alias.asname or top
            self._module_aliases[alias_name] = top

            # Direct import of banned modules is unsafe (network or file helpers)
            if top in self._banned_direct_modules:
                self._mark_unsafe()

        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        top = self._top_level_module(module)

        if top == "os":
            # Only mark unsafe if importing specific os functions that do FS I/O
            for alias in node.names:
                name = alias.name
                if name in self._banned_os_funcs:
                    # Treat the imported name (or its alias) as a banned callable
                    self._banned_name_calls.add(alias.asname or name)
            # Importing from os (path, etc.) without banned funcs is allowed
            self.generic_visit(node)
            return

        if top == "builtins":
            # from builtins import open
            for alias in node.names:
                if alias.name == "open":
                    self._banned_name_calls.add(alias.asname or alias.name)
            self.generic_visit(node)
            return

        # Any import from banned modules is unsafe
        if top in self._banned_direct_modules:
            self._mark_unsafe()
            self.generic_visit(node)
            return

        # Track aliases to their modules for later attribute-call checks
        for alias in node.names:
            alias_name = alias.asname or alias.name
            self._module_aliases[alias_name] = top

        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        # Check for banned direct call names (e.g., open(...))
        func = node.func
        if isinstance(func, ast.Name):
            if func.id in self._banned_name_calls:
                self._mark_unsafe()

        # Check attribute calls like os.remove(...), requests.get(...), etc.
        elif isinstance(func, ast.Attribute):
            value = func.value
            if isinstance(value, ast.Name):
                base_name = value.id

                # If calling via an os alias, check if it's a banned os.* function
                if base_name in self._os_aliases and func.attr in self._banned_os_funcs:
                    self._mark_unsafe()

                # If calling via an alias to a banned module (non-os), disallow
                mod = self._module_aliases.get(base_name)
                if mod in self._banned_direct_modules:
                    self._mark_unsafe()

        self.generic_visit(node)


def analyze_code_safety(code: str) -> bool:
    """
    Analyze Python source code to determine whether it attempts disallowed operations
    such as file I/O or network access.

    Returns True if the code appears safe, False otherwise.
    """
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError:
        # Treat syntactically invalid code as unsafe to execute
        return False

    analyzer = _CodeSafetyAnalyzer()
    analyzer.visit(tree)
    return analyzer.safe


def execute_safe_code(code: str) -> Optional[str]:
    """
    Execute Python code in a restricted environment and capture stdout.

    - Verifies code safety using analyze_code_safety.
    - Executes with a curated set of safe builtins only.
    - Disables importing and other side-effectful builtins.
    - Returns the exact stdout produced as a string, or None if no output.
    """
    if not analyze_code_safety(code):
        raise ValueError("Provided code is not safe to execute.")

    # Curate a minimal set of safe builtins (no open, __import__, eval, exec, etc.)
    allowed_builtin_names = {
        "abs",
        "all",
        "any",
        "ascii",
        "bin",
        "bool",
        "bytes",
        "callable",
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
        "object",
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
        "type",
        "zip",
        "Exception",
        "BaseException",
        "__build_class__",  # allow class definitions
    }
    safe_builtins: Dict[str, object] = {name: getattr(builtins, name) for name in allowed_builtin_names if hasattr(builtins, name)}

    # No importing allowed in executed code
    safe_builtins["__import__"] = lambda *args, **kwargs: (_ for _ in ()).throw(ImportError("Importing is disabled"))

    # Prepare isolated globals for execution
    exec_globals: Dict[str, object] = {
        "__builtins__": safe_builtins,
        "__name__": "__main__",
    }

    output_buffer = io.StringIO()
    with contextlib.redirect_stdout(output_buffer):
        exec(compile(code, "<user_code>", "exec"), exec_globals, None)

    output = output_buffer.getvalue()
    return output if output != "" else None


def run_code_snippet(code: str) -> Optional[str]:
    """
    Check safety of the provided Python code snippet and execute it in a restricted
    environment if safe. Returns the captured stdout as a string, or None if there
    was no output. Raises ValueError if the code is unsafe.
    """
    if not analyze_code_safety(code):
        raise ValueError("Provided code is not safe to execute.")
    return execute_safe_code(code)
