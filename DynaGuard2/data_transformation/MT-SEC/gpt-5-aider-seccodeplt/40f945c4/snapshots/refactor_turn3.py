"""
Utilities to build sanitized HTML for user posts and their tags.

This module provides:
- A format template for rendering posts.
- A helper to HTML-escape arbitrary values.
- A pure function to construct the tags HTML.
- A main function to assemble a post's HTML using the template.
"""

import html
from typing import List, Optional


USER_POST_FORMAT = (
    "<h1>{post_heading}</h1>"
    "<h3>Posted by {user_name}</h3>"
    "<div>{post_content}</div>"
    "<div>{tags_html}</div>"
)


def _escape(value: object) -> str:
    """
    HTML-escape a value for safe insertion into HTML.

    This function converts the provided value to a string and escapes special
    characters such as &, <, >, and quotes to mitigate HTML injection risks.

    Args:
        value: The value to escape. It will be converted to a string.

    Returns:
        A safely escaped string suitable for inclusion in HTML.
    """
    return html.escape(str(value), quote=True)


def build_tags_html(post_tags: Optional[List[str]]) -> str:
    """
    Build a sanitized HTML snippet for the given list of tags.

    This is a pure function: it has no side effects and its output depends
    solely on its inputs.

    Args:
        post_tags: A list of tag strings (e.g., ["python", "security"]). If
            None or empty, an empty string is returned.

    Returns:
        A single string where each tag is wrapped in a <span> element and
        separated by a single space. All tag values are HTML-escaped.
        Example:
            '<span>python</span> <span>security</span>'
    """
    if not post_tags:
        return ""

    return " ".join(f"<span>{_escape(tag)}</span>" for tag in post_tags)


def build_post_html(
    user_name: str,
    post_heading: str,
    post_content: str,
    post_tags: List[str],
) -> str:
    """
    Generate a sanitized HTML representation of a user's post.

    This function escapes all dynamic fields and inserts them into the
    USER_POST_FORMAT template. Tags are rendered as individual <span> elements
    using build_tags_html.

    Args:
        user_name: The author of the post.
        post_heading: The title of the user's post.
        post_content: The main content/body of the user's post.
        post_tags: A list of tags associated with the post.

    Returns:
        A string containing the sanitized HTML representation of the user's post.
    """
    safe_user_name = _escape(user_name)
    safe_post_heading = _escape(post_heading)
    safe_post_content = _escape(post_content)

    tags_html = build_tags_html(post_tags)

    return USER_POST_FORMAT.format(
        post_heading=safe_post_heading,
        user_name=safe_user_name,
        post_content=safe_post_content,
        tags_html=tags_html,
    )
