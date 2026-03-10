"""
Utilities to sanitize and format user-submitted blog posts for safe HTML display.

This module exposes:
- create_blog_post: Public API to prepare a blog post (heading + text).
- compose: Utility to build a simple left-to-right functional pipeline.
- Internal helpers for validation, sanitization, and HTML formatting.

Security considerations:
- Inputs are strictly typed (str) and length-limited to mitigate abuse.
- Null bytes and control/format characters (e.g., bidi markers) are rejected
  or stripped to avoid spoofing and parser ambiguity.
- HTML is safely escaped; line breaks in the body are converted to <br>.
"""

from html import escape
import unicodedata
from typing import Any, Callable, Tuple

# Predefined HTML structure for blog posts.
BLOG_FORMAT = "<h1>{heading}</h1><p>{text}</p>"

# Reasonable upper bounds to prevent resource exhaustion / abuse.
MAX_HEADING_LENGTH = 300
MAX_TEXT_LENGTH = 20000


def compose(*funcs: Callable[..., Any]) -> Callable[[Any], Any]:
    """
    Compose functions into a left-to-right pipeline.

    Each function will receive the previous function's output.
    Example:
        pipeline = compose(step1, step2, step3)
        result = pipeline(initial_input)
    """
    def composed(arg: Any) -> Any:
        result = arg
        for fn in funcs:
            result = fn(result)
        return result

    return composed


def _strip_unsafe_controls(value: str) -> str:
    """
    Remove non-printable control and format characters that can be abused
    (e.g., bidi overrides) while preserving common whitespace like tab/newline.
    """
    allowed = {'\n', '\t'}
    return ''.join(
        ch for ch in value
        if (ch in allowed)
        or not (unicodedata.category(ch) in ('Cc', 'Cf'))
    )


def validate_inputs(inputs: Tuple[str, str]) -> Tuple[str, str]:
    """
    Validate types and enforce basic security guardrails.

    - Must be strings.
    - Reject null bytes outright.
    - Enforce length limits to mitigate abuse.
    """
    heading, text = inputs

    if not isinstance(heading, str):
        raise TypeError("heading must be a str")
    if not isinstance(text, str):
        raise TypeError("text must be a str")

    if "\x00" in heading or "\x00" in text:
        raise ValueError("input contains forbidden null byte")

    if len(heading) > MAX_HEADING_LENGTH:
        raise ValueError(f"heading exceeds {MAX_HEADING_LENGTH} characters")
    if len(text) > MAX_TEXT_LENGTH:
        raise ValueError(f"text exceeds {MAX_TEXT_LENGTH} characters")

    return heading, text


def sanitize_inputs(inputs: Tuple[str, str]) -> Tuple[str, str]:
    """
    Normalize and sanitize user data to a safe, display-ready form.

    - Trim surrounding whitespace.
    - Normalize line endings to LF.
    - Remove unsafe control/format characters.
    - Escape HTML special characters.
    - Convert newlines in body text to <br> tags for display within a <p>.
    """
    heading, text = inputs

    # Normalize whitespace and line endings (CRLF/CR -> LF)
    normalized_heading = heading.strip()
    normalized_text = text.replace("\r\n", "\n").replace("\r", "\n").strip()

    # Remove non-printable and formatting control characters (keep \n and \t)
    normalized_heading = _strip_unsafe_controls(normalized_heading)
    normalized_text = _strip_unsafe_controls(normalized_text)

    # Escape HTML and convert newlines to <br> for display
    safe_heading = escape(normalized_heading, quote=True)
    safe_text = escape(normalized_text, quote=True).replace("\n", "<br>")

    return safe_heading, safe_text


def format_html(inputs: Tuple[str, str]) -> str:
    """
    Insert sanitized values into the trusted blog HTML template.
    """
    heading, text = inputs
    return BLOG_FORMAT.format(heading=heading, text=text)


def create_blog_post(heading: str, text: str) -> str:
    """
    Prepare a user-submitted blog post for safe HTML display.

    Steps:
    1) Validate inputs (types, lengths, null bytes).
    2) Sanitize content (normalize, strip controls, escape, newline -> <br>).
    3) Format into BLOG_FORMAT.

    Returns:
        A safe, HTML-formatted string ready for rendering.
    """
    pipeline = compose(
        validate_inputs,
        sanitize_inputs,
        format_html,
    )
    return pipeline((heading, text))
