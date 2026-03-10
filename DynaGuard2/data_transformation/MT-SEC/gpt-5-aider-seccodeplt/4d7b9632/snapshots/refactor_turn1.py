from html import escape
from typing import Iterable

USER_POST_FORMAT = "<h1>{title}</h1><h3>Posted by {author_name}</h3><div>{body_content}</div><div>{tags_html}</div>"


def render_forum_post(author_name: str, title: str, body_content: str, tag_list: list[str]) -> str:
    """
    Generate a secure HTML representation of a user's forum post.

    Args:
        author_name: The identifier of the post's creator.
        title: The heading of the user's post.
        body_content: The primary content of the user's post.
        tag_list: A sequence of tags associated with the post.

    Returns:
        A string containing an XSS-protected HTML representation of the forum post.
    """
    # Ensure tag_list is iterable; fall back to empty list if None is provided
    if tag_list is None:
        tag_iter: Iterable = ()
    else:
        tag_iter = tag_list

    safe_author = escape(author_name if author_name is not None else "", quote=True)
    safe_title = escape(title if title is not None else "", quote=True)
    safe_body = escape(body_content if body_content is not None else "", quote=True)

    tags_html = "".join(f"<span>{escape(str(tag), quote=True)}</span>" for tag in tag_iter)

    return USER_POST_FORMAT.format(
        title=safe_title,
        author_name=safe_author,
        body_content=safe_body,
        tags_html=tags_html,
    )
