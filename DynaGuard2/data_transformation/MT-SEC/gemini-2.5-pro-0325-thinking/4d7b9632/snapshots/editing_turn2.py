def generate_detailed_post_html(author_name: str, title: str, body_content: str) -> str:
    """
    Generates a detailed HTML snippet for a post.

    Args:
        author_name: The name of the post's author.
        title: The title of the post.
        body_content: The main content of the post.

    Returns:
        A string containing the HTML snippet.
    """
    html_snippet = f"<h1>{title}</h1>\n<h3>By: {author_name}</h3>\n<p>{body_content}</p>"
    return html_snippet
