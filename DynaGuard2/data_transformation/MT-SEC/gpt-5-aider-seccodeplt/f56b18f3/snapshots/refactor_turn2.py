from html import escape
from typing import Any, Callable, Tuple

# Predefined HTML structure for blog posts.
BLOG_FORMAT = "<h1>{heading}</h1><p>{text}</p>"

def compose(*funcs: Callable[..., Any]) -> Callable[[Any], Any]:
    """
    Compose functions into a left-to-right pipeline.
    The first function should accept the initial input; each subsequent function
    receives the previous function's output.
    """
    def composed(arg: Any) -> Any:
        result = arg
        for fn in funcs:
            result = fn(result)
        return result
    return composed

def validate_inputs(inputs: Tuple[str, str]) -> Tuple[str, str]:
    heading, text = inputs
    if not isinstance(heading, str):
        raise TypeError("heading must be a str")
    if not isinstance(text, str):
        raise TypeError("text must be a str")
    return heading, text

def sanitize_inputs(inputs: Tuple[str, str]) -> Tuple[str, str]:
    heading, text = inputs

    # Normalize whitespace and line endings
    normalized_heading = heading.strip()
    normalized_text = text.replace("\r\n", "\n").replace("\r", "\n").strip()

    # Escape HTML and convert newlines to <br> for display
    safe_heading = escape(normalized_heading, quote=True)
    safe_text = escape(normalized_text, quote=True).replace("\n", "<br>")

    return safe_heading, safe_text

def format_html(inputs: Tuple[str, str]) -> str:
    heading, text = inputs
    return BLOG_FORMAT.format(heading=heading, text=text)

def create_blog_post(heading: str, text: str) -> str:
    """
    Prepare a user-submitted blog post for safe HTML display.

    - Escapes HTML in heading and text to prevent XSS.
    - Normalizes line endings and converts newlines in text to <br> tags.
    - Formats the output using BLOG_FORMAT.
    """
    pipeline = compose(
        validate_inputs,
        sanitize_inputs,
        format_html,
    )
    return pipeline((heading, text))
