import html

FORUM_POST_FORMAT = "<h2>{header}</h2><p>{message}</p>{options}"

def create_forum_post(topic: str, text_body: str, points: list[str]) -> str:
    """
    Creates a fully formatted and sanitized HTML string for a forum post.

    Args:
        topic: The topic/title of the post.
        text_body: The main content/message of the post.
        points: A list of discussion points (strings).

    Returns:
        A string containing the formatted HTML for the forum post.
    """
    sanitized_topic = html.escape(topic)
    sanitized_text_body = html.escape(text_body)

    options_html = ""
    if points:
        options_html = "<ul>\n"
        for point in points:
            options_html += f"  <li>{html.escape(point)}</li>\n"
        options_html += "</ul>"

    return FORUM_POST_FORMAT.format(
        header=sanitized_topic,
        message=sanitized_text_body,
        options=options_html
    )
