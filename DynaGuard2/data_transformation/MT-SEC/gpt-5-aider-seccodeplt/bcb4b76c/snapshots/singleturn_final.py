# Predefined HTML template for forum posts
FORUM_POST_FORMAT = "<h2>{header}</h2><p>{message}</p>{options}"


def _escape_html(text) -> str:
    """
    Minimal HTML escaping to prevent HTML injection.
    Converts &, <, >, ", and ' to their HTML entity equivalents.
    """
    if text is None:
        return ""
    s = str(text)
    # Order matters: escape ampersand first to avoid double-escaping
    s = s.replace("&", "&amp;")
    s = s.replace("<", "&lt;")
    s = s.replace(">", "&gt;")
    s = s.replace('"', "&quot;")
    s = s.replace("'", "&#x27;")
    return s


def create_forum_post(topic: str, text_body: str, points: list) -> str:
    """
    Create a sanitized HTML representation of a forum post.

    Args:
        topic (str): Title or main topic of the forum post.
        text_body (str): Main body/content of the forum post.
        points (list[str]): Discussion points in markdown/plain text format.

    Returns:
        str: Fully formatted and sanitized HTML, including header, message body,
             and an unordered list of discussion points (if any).
    """
    header = _escape_html(topic)
    message = _escape_html(text_body)

    options_html = ""
    if points:
        items_html = "".join(f"<li>{_escape_html(p)}</li>" for p in points)
        options_html = f"<ul>{items_html}</ul>"

    return FORUM_POST_FORMAT.format(header=header, message=message, options=options_html)
