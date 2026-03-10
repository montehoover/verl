import re
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
