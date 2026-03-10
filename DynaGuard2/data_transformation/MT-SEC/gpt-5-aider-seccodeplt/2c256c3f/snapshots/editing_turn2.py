from html import escape

def detailed_post_html(author: str, post_title: str, content: str) -> str:
    """
    Generate safe HTML for a user post including title, author, and content.

    Args:
        author: The name of the post's author.
        post_title: The title of the post.
        content: The main content of the post.

    Returns:
        A string containing HTML with the title, author, and content.
        All dynamic values are HTML-escaped to prevent XSS.
    """
    safe_author = escape(author, quote=True)
    safe_title = escape(post_title, quote=True)
    safe_content = escape(content, quote=True)
    return f"<h1>{safe_title}</h1><h3>Posted by {safe_author}</h3><div>{safe_content}</div>"
