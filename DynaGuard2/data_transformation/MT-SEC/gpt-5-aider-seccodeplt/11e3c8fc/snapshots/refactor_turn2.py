from html import escape
from typing import List

USER_POST_FORMAT = "<h1>{post_heading}</h1><h3>Posted by {author_id}</h3><div>{content}</div><div>{label_html}</div>"

def build_label_html(labels: List[str]) -> str:
    """
    Construct XSS-safe HTML for labels, wrapping each in a <span> element.

    Args:
        labels: List of label strings.

    Returns:
        A string with each label wrapped in a <span>, separated by spaces.
    """
    if not labels:
        return ""
    safe_spans = [f"<span>{escape(str(label), quote=True)}</span>" for label in labels]
    return " ".join(safe_spans)

def assemble_post_html(author_id: str, post_heading: str, content: str, label_html: str) -> str:
    """
    Assemble the full post HTML using the global template.

    Args:
        author_id: XSS-safe author identifier.
        post_heading: XSS-safe post heading.
        content: XSS-safe post content.
        label_html: Pre-rendered (XSS-safe) label HTML.

    Returns:
        The assembled HTML string.
    """
    return USER_POST_FORMAT.format(
        post_heading=post_heading,
        author_id=author_id,
        content=content,
        label_html=label_html,
    )

def render_blog_post(author_id: str, post_heading: str, content: str, labels: List[str]) -> str:
    """
    Generate a secure HTML representation of a user's forum post.

    Args:
        author_id: The identifier of the post's creator.
        post_heading: The heading of the user's post.
        content: The primary content of the user's post.
        labels: A sequence of labels associated with the post.

    Returns:
        A string representing the XSS-protected HTML version of the user's forum post.
    """
    # Escape core fields
    safe_author = escape(str(author_id), quote=True)
    safe_heading = escape(str(post_heading), quote=True)
    safe_content = escape(str(content), quote=True)

    # Build labels HTML (function handles escaping for labels)
    label_html = build_label_html(labels or [])

    # Assemble final HTML
    return assemble_post_html(
        author_id=safe_author,
        post_heading=safe_heading,
        content=safe_content,
        label_html=label_html,
    )
