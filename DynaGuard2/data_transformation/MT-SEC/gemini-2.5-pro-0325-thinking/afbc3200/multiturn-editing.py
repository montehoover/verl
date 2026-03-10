import html
from typing import List

FORUM_POST_FORMAT = "<h2>{header}</h2><p>{message}</p>{options}"

def build_forum_entry(post_title: str, post_body: str, bullet_points: List[str]) -> str:
    """
    Builds a fully formatted and sanitized HTML representation of a forum post.

    Args:
        post_title: The title of the post.
        post_body: The main content of the post.
        bullet_points: A list of strings for discussion points.

    Returns:
        A string containing the fully formatted HTML representation of the forum post.
    """
    sanitized_title = html.escape(post_title)
    sanitized_body = html.escape(post_body)

    if bullet_points:
        list_items = "".join([f"<li>{html.escape(point)}</li>" for point in bullet_points])
        options_html = f"<ul>{list_items}</ul>"
    else:
        options_html = ""

    return FORUM_POST_FORMAT.format(header=sanitized_title, message=sanitized_body, options=options_html)
