import html

USER_POST_FORMAT = "<h1>{post_heading}</h1><h3>Posted by {author_id}</h3><div>{content}</div><div>{label_html}</div>"

def render_blog_post(author_id, post_heading, content, labels):
    """
    Generates a secure HTML representation of a user's forum post.
    
    Args:
        author_id: str, the identifier of the post's creator
        post_heading: str, the heading of the user's post
        content: str, the primary content of the user's post
        labels: list(str), a sequence of labels associated with the post
    
    Returns:
        str: XSS-protected HTML version of the user's forum post
    """
    # Escape all user inputs to prevent XSS
    safe_author_id = html.escape(author_id)
    safe_post_heading = html.escape(post_heading)
    safe_content = html.escape(content)
    
    # Create label HTML with each label in its own span
    label_spans = []
    for label in labels:
        safe_label = html.escape(label)
        label_spans.append(f"<span>{safe_label}</span>")
    
    # Join all label spans with a space
    label_html = " ".join(label_spans)
    
    # Format the final HTML using the template
    return USER_POST_FORMAT.format(
        post_heading=safe_post_heading,
        author_id=safe_author_id,
        content=safe_content,
        label_html=label_html
    )
