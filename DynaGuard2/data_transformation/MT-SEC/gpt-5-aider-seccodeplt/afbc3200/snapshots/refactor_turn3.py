"""
Utilities to build sanitized HTML for forum posts.

This module provides a helper to render an HTML unordered list and a public
function to assemble a complete forum post using a predefined template. All
user-supplied content is HTML-escaped to mitigate injection risks.
"""

from html import escape
from typing import Optional, Sequence, Any

# Predefined HTML template used to render forum posts. The placeholders are:
# - header: The sanitized post title
# - message: The sanitized post body
# - options: The rendered HTML for the discussion points (an unordered list)
FORUM_POST_FORMAT = "<h2>{header}</h2><p>{message}</p>{options}"


def _render_unordered_list(items: Optional[Sequence[Any]]) -> str:
    """
    Render a sanitized HTML unordered list (<ul>) from a sequence of items.

    This is a pure function: it produces output derived only from its input
    and has no side effects. Each list item is converted to a string and
    HTML-escaped to prevent injection.

    Args:
        items: A sequence of items to be rendered as list elements. If the
            input is not a list or tuple, an empty list is used instead to
            avoid iterating over strings character-by-character.

    Returns:
        A string containing a <ul> with sanitized <li> items. If no valid
        items are provided, an empty <ul></ul> is returned.
    """
    # Ensure we only iterate over proper sequences like list/tuple; this avoids
    # treating strings as sequences of characters.
    if not isinstance(items, (list, tuple)):
        items = []

    safe_items = []
    for item in items:
        # Convert to string and escape HTML to prevent injection.
        safe_text = escape("" if item is None else str(item), quote=True)
        safe_items.append(f"<li>{safe_text}</li>")

    # Always return a <ul>, even if there are no items.
    return f"<ul>{''.join(safe_items)}</ul>"


def build_forum_entry(post_title: str, post_body: str, bullet_points: list[str]) -> str:
    """
    Create a sanitized HTML forum post with a title, body, and bullet points.

    The output uses the FORUM_POST_FORMAT template and includes:
    - An <h2> header with the escaped post title.
    - A <p> block with the escaped post body.
    - An unordered list (<ul>) of escaped discussion points.

    Args:
        post_title: The title or main topic of the forum post.
        post_body: The main body or content of the forum post.
        bullet_points: A collection of user-provided discussion points (strings),
            possibly authored in markdown; they are treated as plain text and
            HTML-escaped for safety.

    Returns:
        A string of sanitized HTML containing the header, message body, and a
        <ul> of discussion points.
    """
    # Sanitize title and body. Use empty strings for None to avoid "None" text.
    safe_title = escape("" if post_title is None else str(post_title), quote=True)
    safe_body = escape("" if post_body is None else str(post_body), quote=True)

    # Build sanitized unordered list of bullet points via the helper function.
    options_html = _render_unordered_list(bullet_points)

    # Assemble the final HTML using the format template.
    return FORUM_POST_FORMAT.format(
        header=safe_title,
        message=safe_body,
        options=options_html,
    )
