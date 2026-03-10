import logging
from datetime import datetime, timezone
from html import escape

html_template: str = "<html><body><h1>Welcome!</h1><p>{user_content}</p></body></html>"

# Module logger setup with a NullHandler to avoid configuring global logging implicitly.
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

_BIDI_CONTROL_CODEPOINTS = {
    0x061C,  # ARABIC LETTER MARK
    0x200E,  # LEFT-TO-RIGHT MARK
    0x200F,  # RIGHT-TO-LEFT MARK
    0x202A,  # LEFT-TO-RIGHT EMBEDDING
    0x202B,  # RIGHT-TO-LEFT EMBEDDING
    0x202C,  # POP DIRECTIONAL FORMATTING
    0x202D,  # LEFT-TO-RIGHT OVERRIDE
    0x202E,  # RIGHT-TO-LEFT OVERRIDE
    0x2066,  # LEFT-TO-RIGHT ISOLATE
    0x2067,  # RIGHT-TO-LEFT ISOLATE
    0x2068,  # FIRST STRONG ISOLATE
    0x2069,  # POP DIRECTIONAL ISOLATE
}

def _remove_unsafe_controls(text: str) -> str:
    """
    Remove unsafe control characters and bidi controls that could be abused for spoofing.
    Allows common whitespace controls: tab, newline, carriage return.
    """
    if not text:
        return ""
    allowed_whitespace = {"\t", "\n", "\r"}
    cleaned_chars = []
    for ch in text:
        cp = ord(ch)
        if ch in allowed_whitespace:
            cleaned_chars.append(ch)
        elif cp == 0x7F:
            # DEL control
            continue
        elif cp < 0x20:
            # Other C0 controls
            continue
        elif cp in _BIDI_CONTROL_CODEPOINTS:
            # Strip bidi override/isolates/marks
            continue
        else:
            cleaned_chars.append(ch)
    return "".join(cleaned_chars)

def _sanitize_user_input(user_input: str) -> str:
    """
    Pure function that cleans and escapes user-provided input to prevent script injection.
    """
    raw = user_input if user_input is not None else ""
    cleaned = _remove_unsafe_controls(raw)
    return escape(cleaned, quote=True)

def _build_html_response(user_content: str, template: str = html_template) -> str:
    """
    Pure function that inserts the sanitized content into the HTML template.
    """
    return template.format(user_content=user_content)

def _log_render_attempt(raw_input: str, safe_content: str) -> None:
    """
    Log details of the render attempt, including a timestamp and whether
    the raw input appears suspicious. Uses repr() to mitigate log injection.
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    lowered = (raw_input or "").lower()
    suspicious_patterns = (
        "<script",
        "javascript:",
        "onerror",
        "onload",
        "onfocus",
        "onmouseover",
        "<iframe",
        "<img",
        "<svg",
        "href=",
        "style=",
        "srcset=",
        "data:",
    )
    suspicious = any(pat in lowered for pat in suspicious_patterns)
    level = logging.WARNING if suspicious else logging.INFO
    logger.log(
        level,
        "render_html_output: time=%s raw=%s safe=%s suspicious=%s",
        timestamp,
        repr(raw_input if raw_input is not None else ""),
        repr(safe_content),
        suspicious,
    )

def render_html_output(user_input: str) -> str:
    """
    Construct an HTML response by safely embedding user-provided content
    into the predefined html_template.

    Args:
        user_input: Content provided by the user to be embedded.

    Returns:
        The resulting HTML response with user input safely escaped to
        prevent script injection.
    """
    safe_content = _sanitize_user_input(user_input)
    _log_render_attempt(user_input, safe_content)
    return _build_html_response(safe_content)
