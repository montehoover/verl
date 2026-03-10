import re

def _percent_decode(text: str) -> str:
    """
    Decode percent-encoded bytes (%XX) in a best-effort manner, up to two passes.
    Avoids raising if malformed sequences are present.
    """
    def repl(m: re.Match) -> str:
        try:
            return chr(int(m.group(1), 16))
        except Exception:
            return m.group(0)

    # Perform up to two passes to handle cascaded encodings like %25%33%41
    for _ in range(2):
        new_text = re.sub(r'%([0-9a-fA-F]{2})', repl, text)
        if new_text == text:
            break
        text = new_text
    return text


def _decode_numeric_char_refs(text: str) -> str:
    """
    Decode numeric HTML character references like &#58; and &#x3A; in a best-effort way.
    """
    def repl(m: re.Match) -> str:
        num = m.group(1)
        try:
            if num.lower().startswith('x'):
                return chr(int(num[1:], 16))
            return chr(int(num, 10))
        except Exception:
            return m.group(0)

    return re.sub(r'&#(x[0-9a-fA-F]+|\d+);', repl, text)


def contains_dangerous_scheme(user_input: str) -> bool:
    """
    Determine whether the input string contains a risky URL scheme.

    Dangerous schemes include:
      - javascript:
      - vbscript:
      - file:
      - data: (except valid image data URLs, e.g., data:image/png;base64,....)

    Args:
        user_input: The input string to inspect.

    Returns:
        True if a dangerous URL scheme is found (ignoring valid image data URLs), otherwise False.

    Notes:
        - The function is resilient to common obfuscations (whitespace/control chars and percent/numeric-entity encodings).
        - The function doesn't explicitly raise exceptions, but regex-related errors (re.error) could occur if regex
          patterns are malformed.
    """
    # Coerce to string defensively and normalize case
    text = "" if user_input is None else str(user_input)

    # Best-effort decoding of common encodings/obfuscations
    text = _percent_decode(text)
    text = _decode_numeric_char_refs(text)
    lowered = text.lower()

    # Remove whitespace and ASCII control chars to defeat obfuscations like "java\nscript:"
    normalized = re.sub(r'[\x00-\x1f\x7f\s]+', '', lowered)

    # 1) Handle data: scheme specially — allow only valid image data URLs (data:image/...,...).
    # Find potential data URLs (we require a comma per RFC 2397).
    for m in re.finditer(r'(?<![a-z0-9+.\-])data:([^,]*),', normalized):
        mediatype = m.group(1)  # e.g., "", "image/png;base64", "text/plain;charset=utf-8"
        # Empty mediatype defaults to text/plain;charset=US-ASCII, which is NOT an image.
        if not mediatype.startswith('image/'):
            return True  # Dangerous data: usage
        # If it starts with image/, we treat it as a valid image data URL and ignore it.

    # If there are other data: occurrences without a comma (malformed), treat as dangerous.
    if re.search(r'(?<![a-z0-9+.\-])data:(?![^,]*,)', normalized):
        return True

    # 2) Detect classic dangerous executable schemes.
    if re.search(r'(?<![a-z0-9+.\-])(javascript|vbscript|file):', normalized):
        return True

    return False
