def basic_post_html(author: str, post_title: str) -> str:
    """
    Generates basic HTML for a post title and author.

    Args:
        author: The name of the post author.
        post_title: The title of the post.

    Returns:
        A string containing HTML for the post title and author.
    """
    return f"<h1>{post_title}</h1><h3>Posted by {author}</h3>"
