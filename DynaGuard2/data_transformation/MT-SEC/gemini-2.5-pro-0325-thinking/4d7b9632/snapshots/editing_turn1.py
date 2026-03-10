def generate_simple_post_html(author_name: str, title: str) -> str:
    """
    Generates a simple HTML snippet for a post.

    Args:
        author_name: The name of the post's author.
        title: The title of the post.

    Returns:
        A string containing the HTML snippet.
    """
    html_snippet = f"<h1>{title}</h1>\n<h3>By: {author_name}</h3>"
    return html_snippet
