from html import escape

# Ensure FORUM_POST_FORMAT is available; define default if not provided elsewhere
if 'FORUM_POST_FORMAT' not in globals():
    FORUM_POST_FORMAT = "<h2>{header}</h2><p>{message}</p>{options}"

def create_post_html(post_title: str, post_body: str) -> str:
    """
    Create an HTML snippet for a forum post.

    Args:
        post_title: The post title text.
        post_body: The post body text.

    Returns:
        A string in the format: <h2>{post_title}</h2><p>{post_body}</p>
    """
    safe_title = escape(post_title, quote=False)
    safe_body = escape(post_body, quote=False)
    return f"<h2>{safe_title}</h2><p>{safe_body}</p>"

def format_discussion_points(points: list[str]) -> str:
    """
    Create an HTML unordered list for discussion points.

    Args:
        points: A list of discussion point strings.

    Returns:
        A string in the format: <ul><li>point1</li>...<li>pointN</li></ul>
    """
    items = "".join(f"<li>{escape(p, quote=False)}</li>" for p in points)
    return f"<ul>{items}</ul>"

def build_forum_entry(post_title: str, post_body: str, bullet_points: list[str]) -> str:
    """
    Build a complete forum post entry using the global FORUM_POST_FORMAT.

    Args:
        post_title: The post title text.
        post_body: The post body text.
        bullet_points: A list of discussion point strings.

    Returns:
        A formatted string using FORUM_POST_FORMAT:
        "<h2>{header}</h2><p>{message}</p>{options}"
    """
    safe_title = escape(post_title, quote=False)
    safe_body = escape(post_body, quote=False)
    options_html = format_discussion_points(bullet_points or [])
    return FORUM_POST_FORMAT.format(header=safe_title, message=safe_body, options=options_html)
