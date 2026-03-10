from html import escape
from typing import List, Optional

# Pre-existing HTML template for user posts.
USER_POST_FORMAT = "<h1>{post_title}</h1><h3>Posted by {author_name}</h3><div>{content}</div><div>{tags_html}</div>"


def create_post_html(
    author_name: str,
    post_title: str,
    content: str,
    post_tags: List[str],
) -> str:
    """
    Construct an XSS-protected HTML representation of a user's forum post.

    Args:
        author_name: The name of the post's author.
        post_title: The headline of the user's post.
        content: The primary text content of the user's post.
        post_tags: A collection of tags related to the post.

    Returns:
        A string containing the HTML representation with all user-provided
        content HTML-escaped and each tag wrapped in a <span>.
    """
    safe_author = escape(author_name if author_name is not None else "", quote=True)
    safe_title = escape(post_title if post_title is not None else "", quote=True)
    safe_content = escape(content if content is not None else "", quote=True)

    tags = post_tags or []
    tags_html = " ".join(f"<span>{escape(str(tag), quote=True)}</span>" for tag in tags)

    html_output = USER_POST_FORMAT.format(
        post_title=safe_title,
        author_name=safe_author,
        content=safe_content,
        tags_html=tags_html,
    )
    return html_output
