import html
from typing import Sequence

# Fallback in case USER_POST_FORMAT is not provided by the environment.
try:
    USER_POST_FORMAT  # type: ignore  # noqa: F401
except NameError:
    USER_POST_FORMAT = "<h1>{title}</h1><h3>Posted by {author_name}</h3><div>{body_content}</div><div>{tags_html}</div>"


def render_forum_post(author_name: str, title: str, body_content: str, tag_list: Sequence[str]) -> str:
    """
    Generate a secure HTML representation of a user's forum post.

    Args:
        author_name: The identifier of the post's creator.
        title: The heading of the user's post.
        body_content: The primary content of the user's post.
        tag_list: A sequence of tags associated with the post.

    Returns:
        A string containing XSS-protected HTML for the forum post.
    """
    escaped_author = html.escape(author_name, quote=True)
    escaped_title = html.escape(title, quote=True)
    escaped_body = html.escape(body_content, quote=True)

    tags_html = " ".join(f"<span>{html.escape(str(tag), quote=True)}</span>" for tag in tag_list)

    return USER_POST_FORMAT.format(
        title=escaped_title,
        author_name=escaped_author,
        body_content=escaped_body,
        tags_html=tags_html,
    )
