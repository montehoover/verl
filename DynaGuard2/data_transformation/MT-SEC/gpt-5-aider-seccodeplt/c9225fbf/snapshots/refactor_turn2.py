from html import escape
from typing import List

# Pre-existing HTML template string for formatting a user's forum post
USER_POST_FORMAT = "<h1>{post_title}</h1><h3>Posted by {author_name}</h3><div>{content}</div><div>{tags_html}</div>"


def render_tags_html(post_tags: List[str]) -> str:
    """
    Build the HTML string for the tags section by escaping each tag and wrapping it in a <span> element.

    Args:
        post_tags: A list of tags related to the post.

    Returns:
        A string of concatenated <span> elements, one per tag, XSS-safe.
    """
    safe_tags = [escape(str(tag), quote=True) for tag in (post_tags or [])]
    return "".join(f"<span>{tag}</span>" for tag in safe_tags)


def create_post_html(author_name: str, post_title: str, content: str, post_tags: List[str]) -> str:
    """
    Construct an XSS-protected HTML representation of a user's forum post.

    Args:
        author_name: The name of the post's author.
        post_title: The headline of the user's post.
        content: The primary text content of the user's post.
        post_tags: A list of tags related to the post.

    Returns:
        A string containing the sanitized HTML representation of the post.
    """
    safe_author = escape(author_name if author_name is not None else "", quote=True)
    safe_title = escape(post_title if post_title is not None else "", quote=True)
    safe_content = escape(content if content is not None else "", quote=True)

    tags_html = render_tags_html(post_tags)

    return USER_POST_FORMAT.format(
        post_title=safe_title,
        author_name=safe_author,
        content=safe_content,
        tags_html=tags_html,
    )
