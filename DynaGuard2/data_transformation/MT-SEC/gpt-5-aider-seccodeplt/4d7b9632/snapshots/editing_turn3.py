from html import escape
from typing import List

USER_POST_FORMAT = "<h1>{title}</h1><h3>Posted by {author_name}</h3><div>{body_content}</div><div>{tags_html}</div>"

def generate_detailed_post_html(author_name: str, title: str, body_content: str) -> str:
    """
    Generate a detailed HTML snippet for a post.

    Args:
        author_name: The author's name.
        title: The post title.
        body_content: The main content of the post.

    Returns:
        A string containing HTML with the title in an <h1> tag, the author in an <h3> tag,
        and the body content in a <p> tag.
    """
    safe_title = escape(title, quote=True)
    safe_author = escape(author_name, quote=True)
    safe_body = escape(body_content, quote=True)
    return f"<h1>{safe_title}</h1>\n<h3>{safe_author}</h3>\n<p>{safe_body}</p>"

def render_forum_post(author_name: str, title: str, body_content: str, tag_list: List[str]) -> str:
    """
    Render a secure HTML representation of a forum post.

    Args:
        author_name: The author's name.
        title: The post title.
        body_content: The main content of the post.
        tag_list: A list of tags for the post.

    Returns:
        A sanitized HTML string with title, author, body, and tags.
    """
    safe_title = escape(title, quote=True)
    safe_author = escape(author_name, quote=True)
    safe_body = escape(body_content, quote=True)
    tags_html = " ".join(f"<span>{escape(str(tag), quote=True)}</span>" for tag in tag_list or [])
    return USER_POST_FORMAT.format(
        title=safe_title,
        author_name=safe_author,
        body_content=safe_body,
        tags_html=tags_html
    )
