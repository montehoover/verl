import html

def detailed_post_html(author: str, post_title: str, content: str) -> str:
    """
    Generates detailed HTML for a post, including title, author, and content.

    Args:
        author: The name of the post author.
        post_title: The title of the post.
        content: The main content of the post.

    Returns:
        A string containing HTML for the post title, author, and content.
        The content is HTML-escaped to prevent XSS attacks.
    """
    escaped_content = html.escape(content)
    return f"<h1>{post_title}</h1><h3>Posted by {author}</h3><div>{escaped_content}</div>"
