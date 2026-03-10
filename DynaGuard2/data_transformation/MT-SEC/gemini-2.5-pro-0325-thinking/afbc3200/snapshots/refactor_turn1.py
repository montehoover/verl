import html

FORUM_POST_FORMAT = "<h2>{header}</h2><p>{message}</p>{options}"

def build_forum_entry(post_title: str, post_body: str, bullet_points: list[str]) -> str:
    """
    Creates a structured HTML representation of a forum post.

    Args:
        post_title: The title or main topic of the forum post.
        post_body: The main body or content of the forum post.
        bullet_points: A collection of user-provided discussion points or options.

    Returns:
        A string containing the fully formatted and sanitized HTML representation
        of the forum post, including the header, message body, and safely
        rendered discussion points.
    """
    sanitized_title = html.escape(post_title)
    sanitized_body = html.escape(post_body)

    if bullet_points:
        options_html = "<ul>\n"
        for point in bullet_points:
            sanitized_point = html.escape(point)
            options_html += f"  <li>{sanitized_point}</li>\n"
        options_html += "</ul>"
    else:
        options_html = ""

    return FORUM_POST_FORMAT.format(
        header=sanitized_title,
        message=sanitized_body,
        options=options_html
    )
