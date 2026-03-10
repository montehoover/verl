import html

USER_POST_FORMAT = "<h1>{post_heading}</h1><h3>Posted by {author_id}</h3><div>{content}</div><div>{label_html}</div>"

def render_blog_post(author_id: str, post_heading: str, content: str, labels: list[str]) -> str:
    """
    Generates a secure HTML string for a blog post.

    Args:
        author_id: The ID of the post author.
        post_heading: The heading/title of the post.
        content: The main content of the post.
        labels: A list of strings representing tags for the post.

    Returns:
        A secure HTML string representation of the blog post.
    """
    # Escape all inputs to prevent XSS
    safe_post_heading = html.escape(post_heading)
    safe_author_id = html.escape(author_id)
    safe_content = html.escape(content)
    
    # Process labels into HTML spans
    safe_labels_html = "".join([f"<span>{html.escape(label)}</span> " for label in labels]).strip()
    
    return USER_POST_FORMAT.format(
        post_heading=safe_post_heading,
        author_id=safe_author_id,
        content=safe_content,
        label_html=safe_labels_html
    )
