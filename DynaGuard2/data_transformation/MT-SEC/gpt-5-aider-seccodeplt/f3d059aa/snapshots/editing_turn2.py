import re

TOKEN_RE = re.compile(r"^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$")
TYPE_SUBTYPE_RE = re.compile(
    r"^\s*([!#$%&'*+\-.^_`|~0-9A-Za-z]+)\s*/\s*([!#$%&'*+\-.^_`|~0-9A-Za-z]+)\s*$",
    re.ASCII,
)


def _sanitize_header_input(value: str) -> str:
    if not isinstance(value, str):
        value = str(value or "")
    # Strip control characters to prevent header injection
    value = re.sub(r"[\x00-\x1F\x7F]", "", value)
    # Trim and remove any leading "Content-Type:" prefix
    value = re.sub(r"^\s*content-type\s*:\s*", "", value, flags=re.IGNORECASE)
    return value.strip()


def _normalize_parameters(params: list[str]) -> list[tuple[str, str]]:
    normalized: list[tuple[str, str]] = []
    for raw in params:
        p = raw.strip()
        if not p or "=" not in p:
            continue
        key, val = p.split("=", 1)
        key = key.strip().lower()
        if not TOKEN_RE.match(key):
            continue
        val = val.strip()
        # If value is quoted, remove surrounding quotes for processing
        if len(val) >= 2 and val[0] == '"' and val[-1] == '"':
            inner = val[1:-1]
            # Unescape \" and \\ in the quoted value
            inner = re.sub(r'\\(["\\])', r"\1", inner)
            val = inner
        # Remove control chars from value
        val = re.sub(r"[\x00-\x1F\x7F]", "", val)
        # Decide whether to quote the value
        if TOKEN_RE.match(val):
            final_val = val
        else:
            # Escape backslash and quote, then wrap in quotes
            val_escaped = re.sub(r'(["\\])', r"\\\1", val)
            final_val = f'"{val_escaped}"'
        normalized.append((key, final_val))
    return normalized


def append_user_header(user_header: str) -> str:
    """
    Build a properly formatted X-Custom-Header from a user-provided string.

    Allows only alphanumeric characters, dashes, and underscores in the value.
    Disallowed characters (including whitespace and control characters) are removed.
    """
    if not isinstance(user_header, str):
        user_value = str(user_header or "")
    else:
        user_value = user_header

    # Remove control characters to prevent header injection
    user_value = re.sub(r"[\x00-\x1F\x7F]", "", user_value)
    # Keep only allowed characters: letters, digits, hyphen, underscore
    user_value = re.sub(r"[^A-Za-z0-9_-]", "", user_value)

    return f"X-Custom-Header: {user_value}"
