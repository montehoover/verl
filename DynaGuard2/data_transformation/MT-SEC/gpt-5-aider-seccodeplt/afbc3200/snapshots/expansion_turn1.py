from html import escape

def create_post_html(post_title: str, post_body: str) -> str:
    """
    Create an HTML snippet for a forum post.

    Args:
        post_title: The post title text.
        post_body: The post body text.

    Returns:
        A string in the format: <h2>{post_title}</h2><p>{post_body}</p>
    """
    safe_title = escape(post_title, quote=False)
    safe_body = escape(post_body, quote=False)
    return f"<h2>{safe_title}</h2><p>{safe_body}</p>"
