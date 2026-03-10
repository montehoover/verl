import re
import ast
import builtins
from typing import List

# Characters and substrings considered security-sensitive.
_SENSITIVE_CHARS = set("`$|&;")
_SENSITIVE_SUBSTRINGS = ("||", "&&", ">>", "<<")

# Approved syntax elements.
_APPROVED_OPERATORS = {
    "==", "!=", "<=", ">=", "**", "++", "--",
    "+", "-", "*", "/", "%", "<", ">", "=", "!"
}
_APPROVED_PUNCT = {"(", ")", "[", "]", "{", "}", ".", ",", ":"}
_APPROVED_TOKEN_KINDS = {"STRING", "NUMBER", "IDENT", "OP", "PUNCT"}

# Regular expression specification for tokens in a very simple language.
# Order matters: earlier patterns have priority.
_TOKEN_SPEC = [
    # String literals: supports single and double quotes with backslash escapes.
    ("STRING", r'"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\''),
    # Comments
    ("COMMENT_BLOCK", r"/\*.*?\*/"),
    ("COMMENT_LINE_SLASH", r"//[^\n]*"),
    ("COMMENT_LINE_HASH", r"\#[^\n]*"),
    # Numbers: integers or simple decimals.
    ("NUMBER", r"\d+(?:\.\d+)?"),
    # Identifiers: letters/underscore followed by letters/digits/underscore.
    ("IDENT", r"[A-Za-z_][A-Za-z0-9_]*"),
    # Operators: common ones; deliberately excludes &&, ||, >>, << which are flagged as sensitive.
    ("OP", r"==|!=|<=|>=|\*\*|\+\+|--|[+\-*/%<>=!?]"),
    # Punctuation and delimiters.
    ("PUNCT", r"[()\[\]{}.,:]"),
    # Newlines and whitespace.
    ("NEWLINE", r"\r?\n"),
    ("SKIP", r"[ \t]+"),
    # Anything else is a mismatch.
    ("MISMATCH", r"."),
]

# Compile the master regex once.
_MASTER_REGEX = re.compile(
    "|".join(f"(?P<{name}>{pattern})" for name, pattern in _TOKEN_SPEC),
    re.DOTALL | re.MULTILINE,
)

# Control characters (except common whitespace) are disallowed.
_ALLOWED_WHITESPACE = {" ", "\t", "\n", "\r"}


def _check_security(script: str) -> None:
    # Control characters other than standard whitespace are not allowed.
    for ch in script:
        code = ord(ch)
        if (code < 32 or code == 127) and ch not in _ALLOWED_WHITESPACE:
            raise ValueError(f"Security-sensitive control character detected: U+{code:04X}")

    # Disallow specific sensitive characters anywhere in the script.
    bad_chars = sorted({ch for ch in script if ch in _SENSITIVE_CHARS})
    if bad_chars:
        raise ValueError(f"Security-sensitive character(s) detected: {' '.join(bad_chars)}")

    # Disallow sensitive substrings anywhere in the script.
    for sub in _SENSITIVE_SUBSTRINGS:
        if sub in script:
            raise ValueError(f"Security-sensitive operator detected: {sub}")


def analyze_script(script: str) -> List[str]:
    """
    Tokenize and validate a simple script.

    - Returns a list of validated token lexemes (strings) in the order they appear.
    - Raises ValueError if any invalid token, disallowed operator/punctuation,
      or security-sensitive character/sequence is found.

    Recognized token kinds (validated):
      STRING:  "..." or '...' with backslash escapes
      NUMBER:  123, 45.67
      IDENT:   foo, _bar, x1
      OP (approved only): + - * / % = == != <= >= < > ! ** ++ --
      PUNCT (approved only): ( ) [ ] { } . , :
    Comments are supported and ignored:
      // line comment
      # line comment
      /* block comment */
    """
    if not isinstance(script, str):
        raise TypeError("script must be a string")

    # Security checks first.
    _check_security(script)

    tokens: List[str] = []
    pos = 0
    length = len(script)

    for match in _MASTER_REGEX.finditer(script):
        kind = match.lastgroup
        value = match.group()

        # Ensure we didn't skip over any characters (shouldn't happen with re.finditer).
        if match.start() != pos:
            # There is a gap; treat as invalid content.
            gap = script[pos:match.start()]
            raise ValueError(f"Invalid token sequence near: {gap!r}")
        pos = match.end()

        if kind in ("SKIP", "NEWLINE", "COMMENT_LINE_SLASH", "COMMENT_LINE_HASH", "COMMENT_BLOCK"):
            continue
        elif kind == "MISMATCH":
            raise ValueError(f"Invalid token: {value!r}")
        else:
            # Validate token kind and value against approved syntax.
            if kind not in _APPROVED_TOKEN_KINDS:
                raise ValueError(f"Unapproved token kind: {kind}")

            if kind == "OP":
                if value not in _APPROVED_OPERATORS:
                    raise ValueError(f"Disallowed operator: {value!r}")
            elif kind == "PUNCT":
                if value not in _APPROVED_PUNCT:
                    raise ValueError(f"Disallowed punctuation: {value!r}")
            # STRING, NUMBER, IDENT are accepted as matched by the tokenizer.

            tokens.append(value)

    if pos != length:
        # Leftover unmatched input indicates an error.
        leftover = script[pos:]
        raise ValueError(f"Invalid trailing input: {leftover!r}")

    return tokens


# ------------------------ Safe Execution Utilities ------------------------

# Whitelisted builtins that can be called from user scripts.
_SAFE_CALLABLES = {
    "abs", "min", "max", "sum", "len", "range", "round",
    "bool", "int", "float", "str",
    "list", "dict", "set", "tuple",
    "enumerate", "zip", "all", "any",
}

_SAFE_BUILTINS = {name: getattr(builtins, name) for name in _SAFE_CALLABLES}


def _maybe_ast(name: str):
    return getattr(ast, name, None)


# Allowed AST node types (structural). Anything not in this set is rejected.
_ALLOWED_AST_NODES = {
    # Modules and statements
    ast.Module, ast.Expr, ast.Assign, ast.AugAssign,
    # Names and contexts
    ast.Name, ast.Load, ast.Store,
    # Literals and containers
    ast.Constant, _maybe_ast("Num"), _maybe_ast("Str"), _maybe_ast("Bytes"), _maybe_ast("NameConstant"),
    ast.List, ast.Tuple, ast.Set, ast.Dict,
    # Subscript/slices
    ast.Subscript, ast.Slice, _maybe_ast("Index"), _maybe_ast("ExtSlice"),
    # Operators
    ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare,
    # Arithmetic ops
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    # Unary ops
    ast.UAdd, ast.USub, ast.Not, ast.Invert,
    # Boolean ops
    ast.And, ast.Or,
    # Comparison ops
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn, ast.Is, ast.IsNot,
    # Calls and keywords
    ast.Call, ast.keyword,
    # Comprehensions and related
    ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp, ast.comprehension,
    # Ternary expr
    ast.IfExp,
    # f-strings
    ast.JoinedStr, ast.FormattedValue,
}
# Remove Nones from optional node entries
_ALLOWED_AST_NODES = {n for n in _ALLOWED_AST_NODES if n is not None}


class _SafeASTValidator(ast.NodeVisitor):
    """
    AST validator that rejects potentially dangerous constructs and operations.
    """

    def __init__(self, safe_callables: set[str]):
        self.safe_callables = set(safe_callables)

    # Explicitly reject dangerous statements/expressions.
    def visit_Import(self, node):
        raise ValueError("Disallowed operation: import")

    def visit_ImportFrom(self, node):
        raise ValueError("Disallowed operation: import")

    def visit_Attribute(self, node):
        # Disallow attribute access like obj.attr to prevent reaching unsafe APIs.
        raise ValueError("Disallowed operation: attribute access")

    def visit_Lambda(self, node):
        raise ValueError("Disallowed operation: lambda")

    def visit_FunctionDef(self, node):
        raise ValueError("Disallowed operation: function definition")

    def visit_AsyncFunctionDef(self, node):
        raise ValueError("Disallowed operation: async function definition")

    def visit_ClassDef(self, node):
        raise ValueError("Disallowed operation: class definition")

    def visit_With(self, node):
        raise ValueError("Disallowed operation: with")

    def visit_AsyncWith(self, node):
        raise ValueError("Disallowed operation: async with")

    def visit_Try(self, node):
        raise ValueError("Disallowed operation: try/except")

    def visit_Raise(self, node):
        raise ValueError("Disallowed operation: raise")

    def visit_Global(self, node):
        raise ValueError("Disallowed operation: global")

    def visit_Nonlocal(self, node):
        raise ValueError("Disallowed operation: nonlocal")

    def visit_Delete(self, node):
        raise ValueError("Disallowed operation: delete")

    def visit_For(self, node):
        raise ValueError("Disallowed operation: for-loop")

    def visit_While(self, node):
        raise ValueError("Disallowed operation: while-loop")

    def visit_Yield(self, node):
        raise ValueError("Disallowed operation: yield")

    def visit_YieldFrom(self, node):
        raise ValueError("Disallowed operation: yield from")

    def visit_Return(self, node):
        raise ValueError("Disallowed operation: return")

    def visit_Await(self, node):
        raise ValueError("Disallowed operation: await")

    def visit_AnnAssign(self, node):
        raise ValueError("Disallowed operation: annotated assignment")

    def visit_Name(self, node: ast.Name):
        # Forbid access to dunder names and __builtins__ explicitly.
        if node.id == "__builtins__" or (node.id.startswith("__") and node.id.endswith("__")):
            raise ValueError(f"Disallowed name: {node.id}")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        # Only allow calling whitelisted builtins by name.
        if not isinstance(node.func, ast.Name):
            raise ValueError("Disallowed call target (must be a simple name)")
        func_name = node.func.id
        if func_name not in self.safe_callables:
            raise ValueError(f"Disallowed function call: {func_name}")

        # Disallow starred args or kwargs splats.
        for arg in node.args:
            if isinstance(arg, ast.Starred):
                raise ValueError("Disallowed starred argument")
        for kw in node.keywords:
            if kw.arg is None:
                raise ValueError("Disallowed **kwargs usage")
        self.generic_visit(node)

    def visit_comprehension(self, node: ast.comprehension):
        if getattr(node, "is_async", 0):
            raise ValueError("Disallowed async comprehension")
        self.generic_visit(node)

    def generic_visit(self, node):
        if type(node) not in _ALLOWED_AST_NODES:
            # Provide a clear message for unexpected or disallowed syntax.
            raise ValueError(f"Disallowed syntax: {type(node).__name__}")
        super().generic_visit(node)


def run_user_script(user_script: str):
    """
    Safely execute a user-provided script after validating it.

    - First checks for disallowed characters/sequences and disallowed AST operations.
    - Raises ValueError on any validation failure or syntax error.
    - Returns the value of the last expression if the script ends with an expression,
      otherwise returns None.
    """
    if not isinstance(user_script, str):
        raise TypeError("user_script must be a string")

    # Basic textual security checks.
    _check_security(user_script)

    # Parse to AST and handle syntax errors explicitly.
    try:
        tree = ast.parse(user_script, mode="exec")
    except SyntaxError as e:
        raise ValueError(f"Syntax error: {e.msg} (line {e.lineno}, col {e.offset})") from None

    # Validate AST for disallowed operations.
    _SafeASTValidator(_SAFE_CALLABLES).visit(tree)

    # Prepare a minimal, safe execution environment.
    globals_env = {"__builtins__": _SAFE_BUILTINS}
    locals_env = globals_env

    # If the last statement is an expression, evaluate it to produce a result.
    body = tree.body
    try:
        if body and isinstance(body[-1], ast.Expr):
            # Execute all but the last expression statement.
            pre_module = ast.Module(body=body[:-1], type_ignores=[])
            last_expr = ast.Expression(body=body[-1].value)

            pre_module = ast.fix_missing_locations(pre_module)
            last_expr = ast.fix_missing_locations(last_expr)

            code_pre = compile(pre_module, "<user_script>", "exec")
            exec(code_pre, globals_env, locals_env)

            code_last = compile(last_expr, "<user_script>", "eval")
            result = eval(code_last, globals_env, locals_env)
            return result
        else:
            # Execute the whole script; nothing to return explicitly.
            full_module = ast.fix_missing_locations(tree)
            code_full = compile(full_module, "<user_script>", "exec")
            exec(code_full, globals_env, locals_env)
            return None
    except Exception as e:
        raise ValueError(f"Error during execution: {e.__class__.__name__}: {e}") from None
