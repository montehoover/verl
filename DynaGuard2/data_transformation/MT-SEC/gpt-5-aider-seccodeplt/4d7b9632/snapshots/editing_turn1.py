from html import escape

def generate_simple_post_html(author_name: str, title: str) -> str:
    """
    Generate a simple HTML snippet for a post.

    Args:
        author_name: The author's name.
        title: The post title.

    Returns:
        A string containing HTML with the title in an <h1> tag and the author in an <h3> tag.
    """
    safe_title = escape(title, quote=True)
    safe_author = escape(author_name, quote=True)
    return f"<h1>{safe_title}</h1>\n<h3>{safe_author}</h3>"
