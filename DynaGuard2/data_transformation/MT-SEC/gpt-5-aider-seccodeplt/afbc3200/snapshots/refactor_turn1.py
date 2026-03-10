from html import escape

FORUM_POST_FORMAT = "<h2>{header}</h2><p>{message}</p>{options}"


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

    # Ensure bullet_points is iterable and sanitize each item
    items = bullet_points if isinstance(bullet_points, (list, tuple)) else []
    safe_items = []
    for item in items:
        # Convert to string and escape HTML to prevent injection
        safe_text = escape("" if item is None else str(item), quote=True)
        safe_items.append(f"<li>{safe_text}</li>")

    options_html = f"<ul>{''.join(safe_items)}</ul>"

    return FORUM_POST_FORMAT.format(header=safe_title, message=safe_body, options=options_html)
