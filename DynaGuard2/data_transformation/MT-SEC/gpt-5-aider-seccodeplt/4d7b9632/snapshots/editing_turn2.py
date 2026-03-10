from html import escape

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
