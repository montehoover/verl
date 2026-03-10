import html

USER_POST_FORMAT = "<h1>{post_heading}</h1><h3>Posted by {author_id}</h3><div>{content}</div><div>{label_html}</div>"

def render_blog_post(author_id: str, post_heading: str, content: str, labels: list[str]) -> str:
    """
    Generates a secure HTML representation of a user's forum post.

    Args:
        author_id: The identifier of the post's creator.
        post_heading: The heading of the user's post.
        content: The primary content of the user's post.
        labels: A sequence of labels associated with the post.

    Returns:
        A string representing the XSS-protected HTML version of the user's forum post.
    """
    escaped_heading = html.escape(post_heading)
    escaped_author_id = html.escape(author_id)
    escaped_content = html.escape(content)

    escaped_labels_html = []
    for label in labels:
        escaped_label = html.escape(label)
        escaped_labels_html.append(f"<span>{escaped_label}</span>")
    
    label_html_str = "".join(escaped_labels_html)

    return USER_POST_FORMAT.format(
        post_heading=escaped_heading,
        author_id=escaped_author_id,
        content=escaped_content,
        label_html=label_html_str
    )
