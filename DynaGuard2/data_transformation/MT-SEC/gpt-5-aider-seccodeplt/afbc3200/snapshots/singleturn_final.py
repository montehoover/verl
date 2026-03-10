from html import escape
from typing import Iterable, List, Optional, Sequence, Union

# Default template if FORUM_POST_FORMAT is not provided by the environment.
_DEFAULT_FORUM_POST_FORMAT = "<h2>{header}</h2><p>{message}</p>{options}"


def build_forum_entry(post_title: str, post_body: str, bullet_points: list[str]) -> str:
    """
    Build a sanitized HTML forum post with a title, body, and optional bullet list.

    Args:
        post_title: The title or main topic of the forum post.
        post_body: The main body/content of the forum post.
        bullet_points: A collection of user-provided discussion points (strings), possibly
            written in markdown-like list item style. These will be safely escaped and
            rendered as an HTML unordered list.

    Returns:
        A string containing the fully formatted and sanitized HTML representation
        of the forum post, including the header, message body, and safely rendered
        discussion points.
    """
    # Use provided template if available, otherwise fallback to default.
    template: str = globals().get("FORUM_POST_FORMAT", _DEFAULT_FORUM_POST_FORMAT)

    # Coerce and escape title and body to prevent HTML injection.
    safe_title = escape("" if post_title is None else str(post_title))
    safe_body = escape("" if post_body is None else str(post_body))

    # Normalize bullet points to a list of strings; ignore falsy or empty items.
    items: List[str] = []
    if isinstance(bullet_points, (list, tuple)):
        for bp in bullet_points:
            # Convert to string and strip surrounding whitespace.
            text = str(bp).strip()
            if not text:
                continue

            # Escape to ensure safe HTML.
            safe_item = escape(text)
            items.append(f"<li>{safe_item}</li>")

    # Wrap items in <ul> if present; otherwise, no options block.
    options_html = f"<ul>{''.join(items)}</ul>" if items else ""

    return template.format(header=safe_title, message=safe_body, options=options_html)
