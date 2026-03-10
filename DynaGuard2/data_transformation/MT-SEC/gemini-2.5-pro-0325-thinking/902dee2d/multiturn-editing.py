import html
from typing import List

FORUM_POST_FORMAT = "<h2>{headline}</h2><p>{content}</p>{items}"

def build_forum_post(headline: str, content: str, items: List[str]) -> str:
    """
    Builds a structured HTML forum post.

    Args:
        headline: The headline of the post.
        content: The main content of the post.
        items: A list of discussion points.

    Returns:
        A fully formatted and sanitized HTML string for the forum post.
    """
    sanitized_headline = html.escape(headline)
    sanitized_content = html.escape(content)

    items_html = ""
    if items:
        items_html = "<ul>"
        for item in items:
            items_html += f"<li>{html.escape(item)}</li>"
        items_html += "</ul>"

    return FORUM_POST_FORMAT.format(
        headline=sanitized_headline,
        content=sanitized_content,
        items=items_html
    )
