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
    Build a properly formatted Content-Type header from a user-provided MIME type string.

    Accepts either a bare MIME type (e.g., "application/json") or a full header line
    (e.g., "Content-Type: application/json; charset=UTF-8"). Normalizes type/subtype
    to lowercase, parameter keys to lowercase, trims whitespace, quotes parameter
    values when necessary, and strips control characters to prevent header injection.
    Falls back to "application/octet-stream" if parsing fails.
    """
    raw = _sanitize_header_input(user_header)
    if not raw:
        return "Content-Type: application/octet-stream"

    # Split into media type and parameters
    parts = raw.split(";")
    media = parts[0].strip()
    params = parts[1:]

    m = TYPE_SUBTYPE_RE.match(media)
    if not m:
        return "Content-Type: application/octet-stream"

    media_type = f"{m.group(1).lower()}/{m.group(2).lower()}"

    normalized_params = _normalize_parameters(params)
    if normalized_params:
        param_str = "".join(f"; {k}={v}" for k, v in normalized_params)
    else:
        param_str = ""

    return f"Content-Type: {media_type}{param_str}"
