import ast
import io
import tokenize
from typing import Set


SUSPICIOUS_NAMES: Set[str] = {"exec", "eval"}
MAX_LENGTH: int = 100


def _contains_suspicious_names(script: str) -> bool:
    """Return True if the script contains suspicious identifier names."""
    try:
        for tok in tokenize.generate_tokens(io.StringIO(script).readline):
            tok_type, tok_str = tok.type, tok.string
            if tok_type == tokenize.NAME and tok_str in SUSPICIOUS_NAMES:
                return True
    except tokenize.TokenError:
        # Tokenization failed; treat as malformed elsewhere (via ast.parse).
        pass
    return False


def manage_script(user_script: str) -> str:
    """
    Reformat the script by appending a custom comment '# Processed' at the end.
    Enforce a maximum length limit of 100 characters on the original script.

    Raises:
        ValueError: If the script includes suspicious keywords (exec/eval),
                    exceeds the maximum length, or is malformed.
    """
    if not isinstance(user_script, str):
        raise ValueError("Script must be a string.")

    # Enforce maximum length on the original script content.
    if len(user_script) > MAX_LENGTH:
        raise ValueError("Script exceeds maximum allowed length of 100 characters.")

    # Reject scripts containing suspicious identifier names.
    if _contains_suspicious_names(user_script):
        raise ValueError("Script contains suspicious keywords.")

    # Validate that the script is syntactically valid Python.
    try:
        ast.parse(user_script)
    except SyntaxError as e:
        raise ValueError("Malformed script.") from e

    # Append '# Processed' as a final comment line.
    if not user_script.endswith("\n"):
        user_script += "\n"
    return user_script + "# Processed"
