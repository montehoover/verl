from html import escape
from typing import List

USER_POST_FORMAT = "<h1>{post_heading}</h1><h3>Posted by {author_id}</h3><div>{content}</div><div>{label_html}</div>"

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
    safe_author = escape(str(author_id), quote=True)
    safe_heading = escape(str(post_heading), quote=True)
    safe_content = escape(str(content), quote=True)

    safe_labels = []
    if labels is not None:
        for label in labels:
            safe_labels.append(f"<span>{escape(str(label), quote=True)}</span>")

    label_html = " ".join(safe_labels)

    return USER_POST_FORMAT.format(
        post_heading=safe_heading,
        author_id=safe_author,
        content=safe_content,
        label_html=label_html,
    )
