import html

FORUM_POST_FORMAT: str = "<h2>{header}</h2><p>{message}</p>{options}"


def create_forum_post(topic: str, text_body: str, points: list[str]) -> str:
    """
    Return a sanitized HTML string for a forum post using the global FORUM_POST_FORMAT.
    The discussion points are rendered as an HTML unordered list.
    """
    safe_title = html.escape(topic, quote=True)
    safe_message = html.escape(text_body, quote=True)

    options_html = ""
    if points:
        items = "".join(f"<li>{html.escape(str(p), quote=True)}</li>" for p in points)
        options_html = f"<ul>{items}</ul>"

    return FORUM_POST_FORMAT.format(header=safe_title, message=safe_message, options=options_html)
