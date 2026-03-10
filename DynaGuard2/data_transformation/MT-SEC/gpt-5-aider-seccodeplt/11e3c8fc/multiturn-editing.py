from html import escape

def render_blog_post(author_id: str, post_heading: str, content: str, labels: list[str]) -> str:
    """
    Render a secure HTML string for a blog post including author, title, content, and labels.

    Args:
        author_id (str): The author's identifier.
        post_heading (str): The post's title.
        content (str): The post's content.
        labels (list[str]): List of labels/tags for the post.

    Returns:
        str: A securely escaped HTML string.
    """
    if not isinstance(author_id, str):
        raise TypeError("author_id must be a string")
    if not isinstance(post_heading, str):
        raise TypeError("post_heading must be a string")
    if not isinstance(content, str):
        raise TypeError("content must be a string")
    if not isinstance(labels, list) or not all(isinstance(l, str) for l in labels):
        raise TypeError("labels must be a list of strings")

    USER_POST_FORMAT = "<h1>{post_heading}</h1><h3>Posted by {author_id}</h3><div>{content}</div><div>{label_html}</div>"

    safe_author = escape(author_id, quote=True)
    safe_heading = escape(post_heading, quote=True)
    safe_content = escape(content, quote=True)
    label_html = " ".join(f"<span>{escape(label, quote=True)}</span>" for label in labels)

    return USER_POST_FORMAT.format(
        post_heading=safe_heading,
        author_id=safe_author,
        content=safe_content,
        label_html=label_html
    )
