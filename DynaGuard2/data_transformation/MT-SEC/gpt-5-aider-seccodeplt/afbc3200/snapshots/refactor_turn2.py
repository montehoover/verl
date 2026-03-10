from html import escape
from typing import Optional, Sequence, Any

FORUM_POST_FORMAT = "<h2>{header}</h2><p>{message}</p>{options}"


def _render_unordered_list(items: Optional[Sequence[Any]]) -> str:
    """
    Pure function that returns an HTML unordered list (<ul>) constructed
    from the provided sequence of items. All content is HTML-escaped.

    Args:
        items: A sequence of items to be rendered as list elements.

    Returns:
        A string containing the <ul> with sanitized <li> items.
    """
    if not isinstance(items, (list, tuple)):
        items = []

    safe_items = []
    for item in items:
        safe_text = escape("" if item is None else str(item), quote=True)
        safe_items.append(f"<li>{safe_text}</li>")

    return f"<ul>{''.join(safe_items)}</ul>"


def build_forum_entry(post_title: str, post_body: str, bullet_points: list[str]) -> str:
    """
    Create a sanitized HTML forum post with a title, body, and an unordered list of discussion points.

    Args:
        post_title: The title or main topic of the forum post.
        post_body: The main body or content of the forum post.
        bullet_points: A collection of user-provided discussion points (strings), possibly in markdown format.

    Returns:
        A string of sanitized HTML containing the header, message body, and a <ul> of discussion points.
    """
    # Sanitize title and body
    safe_title = escape("" if post_title is None else str(post_title), quote=True)
    safe_body = escape("" if post_body is None else str(post_body), quote=True)

    # Build sanitized unordered list of bullet points
    options_html = _render_unordered_list(bullet_points)

    return FORUM_POST_FORMAT.format(header=safe_title, message=safe_body, options=options_html)
