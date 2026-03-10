from typing import List, Optional
import html

USER_POST_FORMAT = "<h1>{post_heading}</h1><h3>Posted by {user_name}</h3><div>{post_content}</div><div>{tags_html}</div>"


def _escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def build_tags_html(post_tags: Optional[List[str]]) -> str:
    """
    Build a sanitized HTML string for the given list of tags.

    Args:
        post_tags: A list of tags associated with the post.

    Returns:
        A string of <span> elements representing the tags.
    """
    if not post_tags:
        return ""
    return " ".join(f"<span>{_escape(tag)}</span>" for tag in post_tags)


def build_post_html(user_name: str, post_heading: str, post_content: str, post_tags: List[str]) -> str:
    """
    Generate a sanitized HTML representation of a user's post.

    Args:
        user_name: The author of the post.
        post_heading: The title of the user's post.
        post_content: The main content of the user's post.
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
